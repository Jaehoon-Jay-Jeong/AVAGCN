import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Regularization_Loss(nn.Module):
    def __init__(self, kl_weight=0.1, density_weight=1.0, density_target=(0.35, 0.55)):
        super().__init__()
        self.alpha = kl_weight
        self.beta = density_weight
        self.density_target = density_target

    def kl_loss(self, logvar, mu):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    def density_loss(self, adj_matrices):
        current_density = torch.mean(adj_matrices)
        density_min = self.density_target[0]
        density_max = self.density_target[1]

        if current_density < density_min:
            diff = density_min - current_density
            penalty = diff ** 2 * 100.0
        elif current_density > density_max:
            diff = current_density - density_max
            penalty = diff ** 2 * 100.0
        else:
            penalty = torch.tensor(0.0, device=current_density.device)
        return penalty, current_density

    def forward(self, logvar, mu, adj_matrices):
        kl_loss = self.kl_loss(logvar, mu)
        density_loss, current_density = self.density_loss(adj_matrices)

        total = self.alpha * kl_loss + self.beta * density_loss
        return total, kl_loss.detach(), density_loss.detach(), current_density.detach()

class RuleBasedGraphAggregator(nn.Module):
    def __init__(self, node_size):
        super().__init__()
        self.patch_size = int(np.sqrt(node_size))
        self.center_idx = node_size // 2
        self.register_buffer('node_importance', self._init_node_importance())

    def _init_node_importance(self):
        num_nodes = self.patch_size * self.patch_size
        return self._simple_weights(num_nodes)

    def _simple_weights(self, num_nodes):
        weights = torch.ones(num_nodes) / num_nodes
        if self.center_idx < num_nodes:
            center_weight = 0.3
            weights[self.center_idx] = center_weight
            remaining_weight = 1.0 - center_weight
            weights[weights != center_weight] = remaining_weight / (num_nodes - 1)
        return weights

    def forward(self, node_features):
        batch_size, num_nodes, feature_dim = node_features.shape
        device = node_features.device
        weights = self.node_importance[:num_nodes].to(device)
        weights = weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, feature_dim)
        graph_features = (node_features * weights).sum(dim=1)
        return F.normalize(graph_features, p=2, dim=1)
    
class GraphPrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, momentum=0.95,
                 node_level_mining=4,
                 num_classes=16, node_size=81, feature_dim=256,
                 neg_weight=1.0,
                 ):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.node_mining = node_level_mining

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.node_size = node_size
        self.center_idx = node_size // 2

        self.neg_weight = neg_weight
        self.register_buffer('training_step', torch.tensor(0))

        self.register_buffer('prototypes', torch.zeros(num_classes, node_size, feature_dim))
        self.register_buffer('prototype_initialized', torch.zeros(num_classes, dtype=torch.bool))

        self.aggregator = RuleBasedGraphAggregator(node_size)

    def ensure_device_consistency(self, features):
        device = features.device
        if self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)
            self.prototype_initialized = self.prototype_initialized.to(device)
            self.training_step = self.training_step.to(device)
            self.aggregator = self.aggregator.to(device)
        return device

    def normalize_graph_features(self, node_features):
        if node_features.dim() == 3:
            return F.normalize(node_features, p=2, dim=2)
        else:
            return F.normalize(node_features, p=2, dim=1)

    def _get_active_proto_centers(self, active_class_ids):
        proto_centers = self.prototypes[active_class_ids][:, self.center_idx, :]
        return F.normalize(proto_centers, p=2, dim=1)

    def smart_initialize_prototypes(self, features, labels):
        device = features.device
        with torch.no_grad():
            for label in torch.unique(labels):
                label_idx = label.item()
                if not self.prototype_initialized[label_idx]:
                    class_mask = labels == label
                    class_count = class_mask.sum().item()
                    if class_count >= 1:
                        class_features = features[class_mask]
                        class_prototype = (class_features[0] if class_count == 1
                                           else class_features.mean(dim=0))
                        noise = torch.randn_like(class_prototype) * 0.01
                        class_prototype = class_prototype + noise
                        class_prototype = self.normalize_graph_features(class_prototype)
                        self.prototypes[label_idx] = class_prototype.to(device)
                        self.prototype_initialized[label_idx] = True

    def update_prototypes(self, features, labels):
        device = features.device

        with torch.no_grad():
            for label in torch.unique(labels):
                label_idx = label.item()
                class_mask = labels == label
                class_features = features[class_mask]

                if not self.prototype_initialized[label_idx]:
                    class_prototype = (class_features[0] if class_features.shape[0] == 1
                                       else class_features.mean(dim=0))
                    self.prototypes[label_idx] = self.normalize_graph_features(class_prototype).to(device)
                    self.prototype_initialized[label_idx] = True
                else:
                    current_proto = self.prototypes[label_idx].to(device)

                    labeled_mean = self.normalize_graph_features(class_features.mean(dim=0))

                    new_prototype = self.momentum * current_proto + (1 - self.momentum) * labeled_mean
                    new_prototype = self.normalize_graph_features(new_prototype)
                    self.prototypes[label_idx] = new_prototype.to(device)

    def _compute_adjacency_masked_node_loss(self, features, labels,
                                            adj_matrices, active_class_ids,
                                            temperature):
        _, num_nodes, feature_dim = features.shape
        device = features.device
        num_active = len(active_class_ids)

        label_to_active_idx = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
        label_to_active_idx[active_class_ids] = torch.arange(num_active, device=device)

        active_label_indices = label_to_active_idx[labels]
        valid_mask = active_label_indices >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        valid_features = features[valid_mask]
        valid_labels = active_label_indices[valid_mask]
        valid_adj = adj_matrices[valid_mask]
        valid_batch_size = valid_features.shape[0]

        hop1 = valid_adj[:, self.center_idx, :]
        soft_weights = hop1.detach()

        center_mask = torch.zeros(num_nodes, device=device)
        center_mask[self.center_idx] = 1.0
        soft_weights = soft_weights * (1.0 - center_mask) + center_mask

        active_prototypes = self.prototypes[active_class_ids]
        active_prototypes_flat = active_prototypes.reshape(-1, feature_dim)

        similarities = torch.einsum('bnf,mf->bnm', valid_features, active_prototypes_flat)

        batch_idx_v = torch.arange(valid_batch_size, device=device)[:, None]
        node_idx_n = torch.arange(num_nodes, device=device)[None, :]
        
        positive_indices = valid_labels[:, None] * num_nodes + node_idx_n 
        positive_sims = similarities[batch_idx_v, node_idx_n, positive_indices]

        proto_indices_arange = torch.arange(active_prototypes_flat.shape[0], device=device)
        proto_indices_3d = proto_indices_arange[None, None, :]
        
        same_class_start = valid_labels[:, None, None] * num_nodes
        same_class_end = same_class_start + num_nodes
        
        cross_class_mask = (proto_indices_3d < same_class_start) | (proto_indices_3d >= same_class_end)
        cross_class_mask = cross_class_mask.expand(valid_batch_size, num_nodes, -1)

        same_class_base = valid_labels[:, None, None] * num_nodes
        all_node_offsets = torch.arange(num_nodes, device=device)[None, None, :]
        same_class_indices_full = (same_class_base + all_node_offsets).expand(
            valid_batch_size, num_nodes, num_nodes
        )

        self_exclude_mask = torch.ones(valid_batch_size, num_nodes, num_nodes,
                                        dtype=torch.bool, device=device)
        node_range = torch.arange(num_nodes, device=device)
        self_exclude_mask[:, node_range, node_range] = False

        same_class_sims_for_mining = torch.gather(similarities, 2, same_class_indices_full)
        same_class_sims_for_mining = same_class_sims_for_mining.masked_fill(
            ~self_exclude_mask, float('-inf')
        )

        k_same = min(self.node_mining, num_nodes - 1)
        if k_same > 0:
            _, same_class_top_k_local = same_class_sims_for_mining.topk(k_same, dim=2)

            same_class_top_k_global = torch.gather(
                same_class_indices_full, 2, same_class_top_k_local
            )
        else:
            same_class_top_k_global = torch.empty(
                valid_batch_size, num_nodes, 0, dtype=torch.long, device=device
            )

        same_class_keep_mask = torch.zeros(
            valid_batch_size, num_nodes, active_prototypes_flat.shape[0],
            dtype=torch.bool, device=device
        )
        if k_same > 0:
            batch_idx_3d = batch_idx_v.unsqueeze(2).expand(-1, num_nodes, k_same)
            node_idx_3d = node_idx_n.unsqueeze(2).expand(valid_batch_size, -1, k_same)
            same_class_keep_mask[batch_idx_3d, node_idx_3d, same_class_top_k_global] = True

        keep_negative_mask = cross_class_mask | same_class_keep_mask
        masked_negative_sims = similarities.masked_fill(~keep_negative_mask, float('-inf'))
        pos_sims_3d = positive_sims.unsqueeze(2)
        all_logits = torch.cat([pos_sims_3d, masked_negative_sims], dim=2)
        
        log_sum_exp = torch.logsumexp(all_logits / temperature, dim=2)
        node_losses = -(positive_sims / temperature - log_sum_exp)

        weighted_losses = node_losses * soft_weights

        weight_sum_per_sample = soft_weights.sum(dim=1).clamp_min(1e-6)
        per_sample_loss = weighted_losses.sum(dim=1) / weight_sum_per_sample

        center_loss = torch.tensor(0.0, device=device)
        if num_active > 2:
            proto_centers = self._get_active_proto_centers(active_class_ids)
            center_feats = valid_features[:, self.center_idx, :]
            if self.training:
                center_feats = F.dropout(center_feats, p=0.1, training=True)
            center_feats = F.normalize(center_feats, p=2, dim=1)
            center_proto_sims = torch.mm(center_feats, proto_centers.T)

            center_logits = center_proto_sims / temperature
            bidx = torch.arange(valid_batch_size, device=device)
            center_infonce = -(center_logits[bidx, valid_labels]
                            - torch.logsumexp(center_logits, dim=1))

            hard_pen = self._compute_correct_hard_penalty(center_proto_sims, valid_labels)
            center_loss = (center_infonce + hard_pen).mean()

        return per_sample_loss.mean(), center_loss

    def _compute_correct_hard_penalty(self, similarities, positive_indices):
        valid_batch_size, _ = similarities.shape
        device = similarities.device

        batch_range = torch.arange(valid_batch_size, device=device)
        negative_mask = torch.ones_like(similarities, dtype=torch.bool)
        negative_mask[batch_range, positive_indices] = False

        positive_sims = similarities[batch_range, positive_indices]

        negative_sims = similarities.masked_fill(~negative_mask, float('-inf'))
        closest_wrong = negative_sims.max(dim=1)[0]

        margin = 0.3
        penalties = torch.relu(margin - (positive_sims - closest_wrong))
        return penalties

    def compute_vectorized_batch_loss(self, features, labels, adj_matrices):
        initialized_mask = self.prototype_initialized
        if initialized_mask.sum() < 2:
            return torch.tensor(0.0, device=self.prototypes.device, requires_grad=True), {}
 
        active_class_ids = torch.where(initialized_mask)[0]
        current_temp = self.temperature
 
        node_loss, center_loss = self._compute_adjacency_masked_node_loss(
            features, labels, adj_matrices, active_class_ids, current_temp
        )

        total_loss = node_loss + center_loss
 
        return total_loss, {
            'loss_node': node_loss.item(),
            'loss_center': center_loss.item(),
            'loss_labeled_total': total_loss.item(),
        }

    def _negative_expansion_loss(self, labeled_feats, labels, unlabeled_feats,
                                 active_class_ids, temperature):
        device = labeled_feats.device
        num_active = len(active_class_ids)

        g_labeled = self.aggregator(labeled_feats)
        g_unlabeled = self.aggregator(unlabeled_feats)

        active_prototypes = self.prototypes[active_class_ids]
        active_graph_protos = self.aggregator(active_prototypes)

        label_to_active = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
        label_to_active[active_class_ids] = torch.arange(num_active, device=device)
        pos_idx = label_to_active[labels]
        valid = pos_idx >= 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        valid_g = g_labeled[valid]
        valid_pos_idx = pos_idx[valid]
        V = valid_g.shape[0]

        all_keys = torch.cat([active_graph_protos, g_unlabeled.detach()], dim=0)

        sims = torch.mm(valid_g, all_keys.T) / temperature

        batch_idx = torch.arange(V, device=device)
        pos_sims = sims[batch_idx, valid_pos_idx]

        loss = -(pos_sims - torch.logsumexp(sims, dim=1)).mean()

        return loss

    def forward(self, context_features, labels, adj_matrices,
                unlabeled_feats=None,
                proto_update=True,
                n_pseudo=0,
                clean_gcn2 = None
                ): 
        features = self.normalize_graph_features(context_features)
        device = self.ensure_device_consistency(features)
        if labels.device != device:
            labels = labels.to(device)
        if not self.prototype_initialized.all():
            self.smart_initialize_prototypes(features.detach(), labels)
        active_class_ids = torch.where(self.prototype_initialized)[0]
        current_temp = self.temperature

        labeled_loss, labeled_stats = self.compute_vectorized_batch_loss(
            features, labels, adj_matrices
        )

        stats = {
            'unlabeled_active': False,
            'neg_expansion_loss': 0.0,
            'unlabeled_total': 0.0,
            'warmup_scale': 0.0,
        }

        unlabeled_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        use_unlabeled = (unlabeled_feats is not None
                         and self.prototype_initialized.all()
                         and len(active_class_ids) >= 2)

        if use_unlabeled:
            unlabeled_feats_norm = self.normalize_graph_features(unlabeled_feats)
            if unlabeled_feats_norm.device != device:
                unlabeled_feats_norm = unlabeled_feats_norm.to(device)

            stats['unlabeled_active'] = True


            neg_loss = self._negative_expansion_loss(
                features, labels, unlabeled_feats_norm,
                active_class_ids, current_temp
            )
            unlabeled_loss = unlabeled_loss + neg_loss * self.neg_weight
            stats['neg_expansion_loss'] = neg_loss.item()

            stats['unlabeled_total'] = unlabeled_loss.item()
        stats.update(labeled_stats)

        total_loss = labeled_loss + unlabeled_loss

        if proto_update:
            if n_pseudo > 0:
                clean_gcn2 = clean_gcn2.detach()[:-n_pseudo]
                proto_labels = labels[:-n_pseudo]
            else:
                clean_gcn2 = clean_gcn2.detach()
                proto_labels = labels
            self.update_prototypes(clean_gcn2, proto_labels)
            self.training_step += 1

        return total_loss, stats

    def get_prototype_stats(self):
        initialized_count = self.prototype_initialized.sum().item()
        if initialized_count > 0:
            stats = {
                "initialized_classes": initialized_count,
                "total_classes": self.num_classes,
                "current_temperature": self.temperature,
                "training_step": self.training_step.item(),
            }
            if initialized_count > 1:
                with torch.no_grad():
                    active_class_ids = torch.where(self.prototype_initialized)[0]
                    
                    center_protos = self._get_active_proto_centers(active_class_ids)
                    
                    similarities = torch.mm(center_protos, center_protos.T)
                    
                    mask = ~torch.eye(initialized_count, dtype=torch.bool, device=similarities.device)
                    all_sims = similarities[mask]
                    
                    stats["avg_inter_class_similarity"] = all_sims.mean().item()
                    stats["max_inter_class_similarity"] = all_sims.max().item()
                    stats["min_inter_class_similarity"] = all_sims.min().item()
            return stats
        else:
            return {
                "initialized_classes": 0,
                "total_classes": self.num_classes,
                "training_step": self.training_step.item(),
            }
        