import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleProjWithDerivatives(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 3, hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, patch_size * patch_size, hidden_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout * 0.5)
    
    def _build_spectral_channels(self, x_flat):
        x_smooth = F.avg_pool1d(x_flat.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

        d1 = x_smooth[:, 1:] - x_smooth[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]

        d1 = F.pad(d1, (0, 1), mode='replicate')
        d2 = F.pad(d2, (1, 1), mode='replicate')

        with torch.no_grad():
            x_scale = x_flat.std(dim=1, keepdim=True) + 1e-6
            d1_scale = d1.std(dim=1, keepdim=True) + 1e-6
            d2_scale = d2.std(dim=1, keepdim=True) + 1e-6

        d1 = d1 * (x_scale / d1_scale)
        d2 = d2 * (x_scale / d2_scale)
        return torch.stack([x_flat, d1, d2], dim=1)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        BN = B * N
        
        x_flat = x.reshape(BN, C)
        x_3ch = self._build_spectral_channels(x_flat)
        x_3ch = x_3ch.reshape(BN, C * 3)
        
        x_proj = self.input_proj(x_3ch).reshape(B, N, -1) + self.pos_encoding
        return self.dropout(x_proj)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, patch_size, kernel_size=3, expansion=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pw_expand = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw_contract = nn.Linear(dim * expansion, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        H = W = self.patch_size
        residual = x
        
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_2d = self.dw_conv(x_2d)
        x = x_2d.reshape(B, C, N).transpose(1, 2)
        
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        x = self.dropout(x)
        
        return residual + x

class VAEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, patch_size, dropout=0.5):
        super(VAEncoder, self).__init__()
        self.encoder_block = ConvNeXtBlock(hidden_dim, patch_size, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = self.encoder_block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

class GraphAwareDecoder(nn.Module):
    def __init__(self, latent_dim, patch_size=5, target_features=64, dropout=0.5):
        super(GraphAwareDecoder, self).__init__()
        self.node_size = patch_size * patch_size
        self.target_features = target_features

        self.register_buffer('temperature', torch.tensor(1.2))

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, target_features),
            nn.LayerNorm(target_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.register_buffer(
            'upper_tri_mask', 
            torch.triu(torch.ones(self.node_size, self.node_size), diagonal=1).bool()
        )
        self.register_buffer(
            'eye_mask', 
            torch.eye(self.node_size).bool()
        )
        self.query_proj = nn.Linear(target_features, target_features//4)
        self.key_proj = nn.Linear(target_features, target_features//4)
        
    def create_attention_adjacency(self, node_features, hard=True):
        temp = torch.clamp(self.temperature, min=1e-5)
        q = F.normalize(self.query_proj(node_features), dim=-1)
        k = F.normalize(self.key_proj(node_features), dim=-1)
        scores = (torch.bmm(q, k.transpose(1,2)) + torch.bmm(k, q.transpose(1,2))) / 2
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        scores = scores.masked_fill(self.eye_mask, -1e9)

        if self.training:
            eps = 1e-10
            logits = torch.stack([scores, -scores], dim=-1)
            eps = 1e-10
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
            perturbed = (logits + gumbel_noise) / temp
            y_soft = perturbed.softmax(dim=-1)
            
            if hard:
                index = y_soft.max(dim=-1, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
            adjacency_full = ret[..., 0]
            upper = adjacency_full * self.upper_tri_mask
            adjacency = upper + upper.transpose(-1, -2)
        else:
            adjacency = (scores > 0).float()
            adjacency = adjacency * (1 - self.eye_mask.float())
            
        return adjacency

    def forward(self, z):
        node_features = self.projector(z)
        adj_matrices = self.create_attention_adjacency(node_features)
        return adj_matrices, node_features

class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, use_symmetric_norm=True):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True) 
        self.bn = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.use_symmetric_norm = use_symmetric_norm

    def forward(self, node_features, adj_matrices):
        B, N, _ = adj_matrices.shape
        device = adj_matrices.device

        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        adj_hat = adj_matrices + eye
        
        if self.use_symmetric_norm:
            degree = adj_hat.sum(dim=2, keepdim=True)
            degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
            adj_normalized = degree_inv_sqrt * adj_hat * degree_inv_sqrt.transpose(1, 2)
        else:
            degree = adj_hat.sum(dim=2, keepdim=True)
            adj_normalized = adj_hat / (degree + 1e-6)

        support = torch.bmm(adj_normalized, node_features) 
        output = self.fc(support)

        output = self.bn(output)
        output = self.act(output)
        output = self.dropout(output)

        return output

class PrototypeLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, scale=10.0, center_idx=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scale = scale
        self.center_idx = center_idx
        self.linear = nn.Linear(input_dim, num_classes, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)
        self.register_buffer('active_mask', torch.ones(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def set_prototype_weights(self, prototypes, center_idx=None, initialized_mask=None):
        center_idx = center_idx if center_idx is not None else self.center_idx
        assert center_idx is not None, "center_idx must be provided"
        assert prototypes.dim() == 3 and prototypes.shape[0] == self.num_classes \
            and prototypes.shape[2] == self.input_dim, \
            f"shape mismatch {tuple(prototypes.shape)} vs ({self.num_classes}, N, {self.input_dim})"

        centers = prototypes[:, center_idx, :] 
        normed = F.normalize(centers, p=2, dim=1).to(self.linear.weight.device)
        
        w = self.linear.weight.data
        if initialized_mask is not None:
            initialized_mask = initialized_mask.to(w.device)
            w[initialized_mask] = normed[initialized_mask]
            w[~initialized_mask] = 0.0
            self.active_mask.copy_(initialized_mask)
        else:
            w.copy_(normed)
            self.active_mask.fill_(True)

    def forward(self, features):
        feat = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.linear.weight, p=2, dim=1)
        logits = F.linear(feat, w) * self.scale
        return logits.masked_fill(~self.active_mask, float('-inf'))

class AVAGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, patch_size, target_features, dropout=0.5):
        super(AVAGCN, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.target_features = target_features
        self.center_idx = patch_size * patch_size // 2

        self.spatial_spectral = SimpleProjWithDerivatives(input_dim, hidden_dim, patch_size, dropout=dropout)
        self.encoder = VAEncoder(hidden_dim, latent_dim, patch_size, dropout)
        self.decoder = GraphAwareDecoder(latent_dim, patch_size, target_features, dropout=dropout)

        self.context_gcn1 = GCNBlock(target_features, target_features, dropout)
        self.context_gcn2 = GCNBlock(target_features, target_features, dropout)

        self.classifier = PrototypeLinearClassifier(target_features, num_classes=num_classes, center_idx=self.center_idx)

    def return_output_contextGCN(self, x, deterministic=True):
        x = self.spatial_spectral(x)
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar, deterministic)
        adj_matrices, node_features = self.decoder(z)
        gcn1_features = self.context_gcn1(node_features, adj_matrices)
        gcn2_features = self.context_gcn2(gcn1_features, adj_matrices)
        return {
            "result": gcn2_features,
            "gcn1_output": gcn1_features,
            'node_features': node_features,
            "adj_matrices": adj_matrices,
            "mu": mu,
            'logvar': logvar,
            }

    def forward(self, x, deterministic=True):   
        x = self.spatial_spectral(x)
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar, deterministic)
        adj_matrices, node_features = self.decoder(z)
        gcn1_features = self.context_gcn1(node_features, adj_matrices)
        gcn2_features = self.context_gcn2(gcn1_features, adj_matrices)

        center_features = gcn2_features[:, self.center_idx, :]

        predictions = self.classifier(center_features)       
        return {
                'predictions': predictions, 
                "gcn2_output": gcn2_features,
                "gcn1_output": gcn1_features,
                'node_features': node_features, 
                'adj_matrices': adj_matrices, 
                'center_features': center_features,
                'mu': mu, 
                'logvar': logvar,
        }

class BYOLHead(nn.Module):
    def __init__(self, feature_dim, proj_hidden_dim=None, proj_out_dim=None):
        super().__init__()
        proj_hidden_dim = proj_hidden_dim or feature_dim * 2
        proj_out_dim = proj_out_dim or feature_dim
        
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_out_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )
    
    def project(self, x):
        return self.projector(x)
    
    def predict(self, x):
        return self.predictor(x)

class NNBank(nn.Module):
    def __init__(self, feature_dim, queue_size=2048):
        super().__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        
        self.register_buffer('queue', torch.zeros(queue_size, feature_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_filled', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('num_stored', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def enqueue(self, features):
        features_norm = F.normalize(features, dim=-1, p=2)
        batch_size = features_norm.size(0)
        
        ptr = int(self.queue_ptr.item())
        end_ptr = ptr + batch_size
        
        if end_ptr <= self.queue_size:
            self.queue[ptr:end_ptr] = features_norm
        else:
            first_part = self.queue_size - ptr
            self.queue[ptr:] = features_norm[:first_part]
            self.queue[:batch_size - first_part] = features_norm[first_part:]
            self.queue_filled[0] = True
        
        self.queue_ptr[0] = end_ptr % self.queue_size
        if not self.queue_filled.item():
            new_stored = min(self.queue_size, end_ptr)
            self.num_stored[0] = new_stored
            if end_ptr >= self.queue_size:
                self.queue_filled[0] = True
                self.num_stored[0] = self.queue_size
        else:
            self.num_stored[0] = self.queue_size
    
    @torch.no_grad()
    def find_nearest(self, queries):
        n = int(self.num_stored.item())
        if n < 1:
            return F.normalize(queries, dim=-1, p=2)
        
        queries_norm = F.normalize(queries, dim=-1, p=2)
        valid_queue = self.queue[:n]
        
        sim = torch.mm(queries_norm, valid_queue.T)
        nearest_idx = sim.argmax(dim=1)
        
        return valid_queue[nearest_idx]
    
    @torch.no_grad()
    def find_top_k_nearest(self, queries, k=5, temperature=0.1):
        n = int(self.num_stored.item())
        if n < k:
            return self.find_nearest(queries)
        
        queries_norm = F.normalize(queries, dim=-1, p=2)
        valid_queue = self.queue[:n]
        sim = torch.mm(queries_norm, valid_queue.T)
        
        topk_sim, topk_idx = sim.topk(k, dim=1)
        weights = F.softmax(topk_sim / temperature, dim=1)
        
        topk_features = valid_queue[topk_idx]
        weighted = (weights.unsqueeze(-1) * topk_features).sum(dim=1)
        return F.normalize(weighted, dim=-1, p=2)
    
    def is_ready(self, min_size=None):
        min_size = min_size or (self.queue_size // 4)
        return self.num_stored.item() >= min_size
    
    def reset(self):
        self.queue.zero_()
        self.queue_ptr.zero_()
        self.queue_filled.zero_()
        self.num_stored.zero_()
