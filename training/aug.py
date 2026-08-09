import torch
import numpy as np
import random
import torch.nn.functional as F

class HyperspectralPatchAugmentation:
    def __init__(self, 
                 spatial_size: int = 11, 
                 num_bands: int = 200, 
                 device: str = 'cuda'):
        self.spatial_size = spatial_size
        self.num_bands = num_bands
        self.device = device
        self.center = spatial_size // 2

        band_idx = torch.linspace(0, 1, num_bands, device=device)
        self.noise_profile = 1.0 + 2.0 * (2 * band_idx - 1).pow(2)
        
        y_coord = torch.arange(spatial_size, device=device).float() - self.center
        x_coord = torch.arange(spatial_size, device=device).float() - self.center

        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing='ij')
        dist = torch.sqrt(y_grid**2 + x_grid**2)
        
        self.dist_map = dist / (dist.max() + 1e-6)
        
        self.dist_map_4d = self.dist_map.view(1, spatial_size, spatial_size, 1)
        
        self.center_protect = torch.ones(spatial_size, spatial_size, device=device)
        self.center_protect[self.center, self.center] = 0.0 
        
        self.cumulative_probs = {
            'weak':   (['none', 'weak'], [0.3, 0.7]),
            'strong': (['none', 'weak', 'medium', 'strong'], [0.2, 0.2, 0.2, 0.4]),
        }
        
        self.aug_params = {
            'weak': {
                'sensor_noise': {'prob': 0.5, 'base_std': 0.003},
            },
            'medium': {
                'sensor_noise':            {'prob': 0.3, 'base_std': 0.008},
                'illumination_curve':      {'prob': 0.4, 'strength': 0.05},
                'distance_weighted_noise': {'prob': 0.3, 'max_std': 0.02},
                'context_cutmix':          {'prob': 0.8, 'beta': 0.3, 'same_class_ratio': 0.5},
            },
            'strong': {
                'sensor_noise':            {'prob': 0.4, 'base_std': 0.012},
                'illumination_curve':      {'prob': 0.5, 'strength': 0.08},
                'distance_weighted_noise': {'prob': 0.4, 'max_std': 0.035},
                'context_cutmix':          {'prob': 0.9, 'beta': 0.5, 'same_class_ratio': 0.0},
            },
        }

    def _get_batch_mask(self, batch_size, prob):
        return (torch.rand(batch_size, 1, 1, 1, device=self.device) < prob).float()

    def sensor_noise(self, x, base_std, apply_mask):
        noise = torch.randn_like(x) * (base_std * self.noise_profile.view(1, 1, 1, -1))
        x_aug = x + noise
        return apply_mask * x_aug + (1 - apply_mask) * x
    
    def illumination_curve(self, x, strength, apply_mask):
        B = x.size(0)
        band_idx = torch.linspace(-0.5, 0.5, self.num_bands, device=self.device)
        slope = torch.empty(B, 1, 1, 1, device=self.device).uniform_(-strength, strength)
        curve = 1.0 + slope * band_idx.view(1, 1, 1, -1)
        x_aug = x * curve
        return apply_mask * x_aug + (1 - apply_mask) * x
    
    def distance_weighted_noise(self, x, max_std, apply_mask):
        noise_std = self.dist_map_4d * max_std
        noise = torch.randn_like(x) * noise_std
        x_aug = x + noise
        return apply_mask * x_aug + (1 - apply_mask) * x

    def context_cutmix(self, x, labels=None, beta=1.0, same_class_ratio=0.0):
        batch_size, H, W, C = x.size()
        if batch_size < 2:
            return x
        
        shift = random.randint(1, batch_size - 1)
        indices = torch.arange(batch_size, device=self.device).roll(shifts=shift)
        
        if same_class_ratio > 0 and labels is not None:
            for i in range(batch_size):
                if random.random() < same_class_ratio:
                    candidates = (labels == labels[i]).nonzero(as_tuple=True)[0]
                    candidates = candidates[candidates != i]
                    if len(candidates) > 0:
                        indices[i] = candidates[random.randint(0, len(candidates) - 1)]
        
        x_donor = x[indices]
        
        lam = np.random.beta(beta, beta)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = max(1, int(W * cut_rat))
        cut_h = max(1, int(H * cut_rat))
        
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        
        mix_mask = torch.zeros(H, W, device=self.device)
        mix_mask[y1:y2, x1:x2] = 1.0 
        
        mix_mask = mix_mask * self.center_protect
        mix_mask = mix_mask.view(1, H, W, 1)
        
        return x * (1.0 - mix_mask) + x_donor * mix_mask
    
    def __call__(self, x, mode='medium', labels=None):
        batch_size = x.shape[0]
        valid_mask = (torch.abs(x).sum(dim=-1, keepdim=True) > 1e-6).float()
        
        if mode.startswith('pretrain'):
            actual_mode = mode
        elif mode in self.cumulative_probs:
            modes, probs = self.cumulative_probs[mode]
            actual_mode = np.random.choice(modes, p=probs)
        else:
            actual_mode = mode

        if actual_mode == 'none':
            return x

        flip_h = torch.randint(0, 2, (batch_size, 1, 1, 1), device=x.device).float()
        x = flip_h * torch.flip(x, dims=[2]) + (1 - flip_h) * x
        valid_mask = flip_h * torch.flip(valid_mask, dims=[2]) + (1 - flip_h) * valid_mask

        flip_v = torch.randint(0, 2, (batch_size, 1, 1, 1), device=x.device).float()
        x = flip_v * torch.flip(x, dims=[1]) + (1 - flip_v) * x
        valid_mask = flip_v * torch.flip(valid_mask, dims=[1]) + (1 - flip_v) * valid_mask

        k_v = torch.randint(0, 4, (batch_size,), device=x.device)
        if not torch.all(k_v == 0):
            x_rot1 = torch.rot90(x, 1, dims=[1, 2])
            x_rot2 = torch.rot90(x, 2, dims=[1, 2])
            x_rot3 = torch.rot90(x, 3, dims=[1, 2])
            
            m_rot1 = torch.rot90(valid_mask, 1, dims=[1, 2])
            m_rot2 = torch.rot90(valid_mask, 2, dims=[1, 2])
            m_rot3 = torch.rot90(valid_mask, 3, dims=[1, 2])

            combined_x = torch.stack([x, x_rot1, x_rot2, x_rot3])
            combined_m = torch.stack([valid_mask, m_rot1, m_rot2, m_rot3])
            
            batch_indices = torch.arange(batch_size, device=x.device)
            x = combined_x[k_v, batch_indices]
            valid_mask = combined_m[k_v, batch_indices]

        params = self.aug_params.get(actual_mode, self.aug_params['medium'])
        
        if 'sensor_noise' in params:
            p = params['sensor_noise']['prob']
            mask = self._get_batch_mask(batch_size, p)
            x = self.sensor_noise(x, params['sensor_noise']['base_std'], mask)
        
        if 'illumination_curve' in params:
            p = params['illumination_curve']['prob']
            mask = self._get_batch_mask(batch_size, p)
            x = self.illumination_curve(x, params['illumination_curve']['strength'], mask)

        if 'context_cutmix' in params and random.random() < params['context_cutmix']['prob']:
            x = self.context_cutmix(
                x, labels=labels,
                beta=params['context_cutmix']['beta'],
                same_class_ratio=params['context_cutmix']['same_class_ratio']
            )

        if 'distance_weighted_noise' in params:
            p = params['distance_weighted_noise']['prob']
            mask = self._get_batch_mask(batch_size, p)
            x = self.distance_weighted_noise(x, params['distance_weighted_noise']['max_std'], mask)

        x = x * valid_mask
        
        return x

    def create_boundary_cutmix_unlabeled(self, x, beta=0.5):
        """unlabeled-cutmix mode"""
        batch_size, H, W, C = x.size()
        if batch_size < 2:
            return x
        
        shift = random.randint(1, batch_size - 1)
        indices = torch.arange(batch_size, device=self.device).roll(shifts=shift)
        x_donor = x[indices]
        
        lam = np.clip(np.random.beta(beta, beta), 0.16, 0.64)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = max(2, int(W * cut_rat))
        cut_h = max(2, int(H * cut_rat))
        
        center_y, center_x = H // 2, W // 2
        cy = center_y + random.randint(-min(2, H//4), min(2, H//4))
        cx = center_x + random.randint(-min(2, W//4), min(2, W//4))
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, x1 + cut_w)
        y2 = min(H, y1 + cut_h)
        
        mix_mask = torch.zeros(H, W, device=self.device)
        mix_mask[y1:y2, x1:x2] = 1.0
        mix_mask = mix_mask * self.center_protect
        mix_mask = mix_mask.view(1, H, W, 1)
        
        return x * (1.0 - mix_mask) + x_donor * mix_mask
    