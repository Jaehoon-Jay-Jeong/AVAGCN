import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score

import os
import copy
import gc
import time
from collections import defaultdict

from model.AVAGCN import BYOLHead, NNBank
from .aug import HyperspectralPatchAugmentation
from .criterion import Regularization_Loss
from .criterion import GraphPrototypeContrastiveLoss

from dataset.dataset import CLASSIFICATION_PALETTE

import logging
log = logging.getLogger(__name__)
print = log.info

class BYOLTrainer:
    def __init__(self, target_feature=200, device='cuda', queue_size=2048, center_idx=60, nn_byol_weight = 0.3,nn_byol_warmup_epochs=50, nn_byol_top_k=1):
        self.target_feature = target_feature
        self.device = device
        self.queue_size = queue_size
        self.center_idx = center_idx
        self.nn_byol_warmup_epochs = nn_byol_warmup_epochs
        self.nn_byol_top_k = nn_byol_top_k
        self.nn_byol_weight = nn_byol_weight

        self.byol_weak = "weak"
        self.byol_strong = "strong"

        self.byol_head_student = BYOLHead(
            feature_dim=target_feature,
            proj_hidden_dim=target_feature * 2,
            proj_out_dim=target_feature,
        ).to(self.device)

        self.nn_bank = NNBank(
            feature_dim=target_feature,
            queue_size=queue_size,
        ).to(self.device)

        self.byol_head_teacher = None

    def _byol_module_reset(self):
        target_feature = self.target_feature
        self.byol_head_student = BYOLHead(
            feature_dim=target_feature,
            proj_hidden_dim=target_feature * 2,
            proj_out_dim=target_feature,
        ).to(self.device)

        self.nn_bank = NNBank(
            feature_dim=target_feature,
            queue_size=self.queue_size,
        ).to(self.device)

        self.byol_head_teacher = None

    def byol_loss(self, online_pred, target_proj):
        online_norm = F.normalize(online_pred, dim=-1, p=2)
        target_norm = F.normalize(target_proj, dim=-1, p=2)
        return 2 - 2 * (online_norm * target_norm).sum(dim=-1).mean()

    def compute_byol_losses_from_feats(self, s_strong_c, s_weak_c, t_weak_c, t_strong_c,
                                       current_epoch, round_idx=0):
        device = s_strong_c.device
        if self.byol_head_teacher is None:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        s_strong_pred = self.byol_head_student.predict(
            self.byol_head_student.project(s_strong_c)
        )
        s_weak_pred = self.byol_head_student.predict(
            self.byol_head_student.project(s_weak_c)
        )

        with torch.no_grad():
            t_weak_proj = self.byol_head_teacher.project(t_weak_c)
            t_strong_proj = self.byol_head_teacher.project(t_strong_c)

        loss_1 = self.byol_loss(s_strong_pred, t_weak_proj)
        loss_2 = self.byol_loss(s_weak_pred, t_strong_proj)
        byol_loss_val = 0.5 * (loss_1 + loss_2)

        in_warmup = (round_idx == 0) and (
            not self.nn_bank.is_ready() or current_epoch < self.nn_byol_warmup_epochs
        )
        if in_warmup:
            with torch.no_grad():
                self.nn_bank.enqueue(t_weak_proj)
            nn_byol_loss_val = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            with torch.no_grad():
                if self.nn_byol_top_k == 1:
                    nn_target = self.nn_bank.find_nearest(t_weak_proj)
                else:
                    nn_target = self.nn_bank.find_top_k_nearest(
                        t_weak_proj, k=self.nn_byol_top_k, temperature=0.1
                    )
                self.nn_bank.enqueue(t_weak_proj)
            nn_byol_loss_val = self.byol_loss(s_strong_pred, nn_target)

        return byol_loss_val, nn_byol_loss_val

class SemiSupervisedTrainer:
    def __init__(self, model, DATASET, PATCH_SIZE, device='cuda', num_classes=16,
                 target_feature=200, raw_bands=200, BATCH_SIZE=80,
                 mt_loss_weight = 0.5,
                 nn_byol_warmup_epochs=50, nn_byol_top_k=1, byol_weight=0.5, nn_byol_weight=0.3):
        self.model = model.to(device)
        self.DATASET = DATASET
        self.PATCH_SIZE = PATCH_SIZE
        self.device = device
        self.num_classes = num_classes
        self.raw_bands = raw_bands
        self.target_feature = target_feature
        self.node_size = PATCH_SIZE * PATCH_SIZE
        self.center_idx = self.node_size // 2
        self.BATCH_SIZE = BATCH_SIZE

        self.aug = HyperspectralPatchAugmentation(spatial_size=self.PATCH_SIZE,num_bands=self.raw_bands,device=self.device)

        self.byol_trainer = BYOLTrainer(target_feature=self.target_feature, 
                                        device=self.device, 
                                        center_idx=self.center_idx, 
                                        nn_byol_weight = nn_byol_weight,
                                        nn_byol_warmup_epochs=nn_byol_warmup_epochs,
                                        nn_byol_top_k=nn_byol_top_k)

        self.pseudo_pool_data = None
        self.pseudo_pool_labels = None

        self.best_val_acc = 0.0
        
        self.mt_loss_weight = mt_loss_weight
        self.byol_weight = byol_weight
        self.nn_byol_weight = nn_byol_weight

        self.recon_head = nn.Linear(target_feature, raw_bands).to(self.device)

    def _recon_head_reset(self, device):
        self.recon_head = nn.Linear(self.target_feature, self.raw_bands).to(device)

    @torch.no_grad()
    def _generate_pseudo_pool_consistent(self, unlabeled_loader, conf_threshold,
                                        per_class_cap, active_class_ids):
        if not hasattr(self, 'teacher_model') or self.teacher_model is None:
            teacher_scorer = self.model
        else:
            teacher_scorer = self.teacher_model

        teacher_scorer.eval()
        self.model.eval()
        proto_centers = self.contrastive_loss._get_active_proto_centers(active_class_ids)

        candidates = defaultdict(list)
        n_pass_agree = 0
        n_pass_conf = 0
        n_total = 0

        pseudo_temperature = 0.1

        for data, _ in unlabeled_loader:
            data = data.to(self.device)
            n_total += data.shape[0]

            view = data.clone()

            result_t = teacher_scorer.return_output_contextGCN(view, deterministic=True)
            result_s = self.model.return_output_contextGCN(view, deterministic=True)

            feat_t = self.contrastive_loss.normalize_graph_features(
                result_t['result'][:, self.center_idx, :]
            )
            feat_s = self.contrastive_loss.normalize_graph_features(
                result_s['result'][:, self.center_idx, :]
            )

            probs_t = F.softmax(feat_t @ proto_centers.T / pseudo_temperature, dim=1)
            probs_s = F.softmax(feat_s @ proto_centers.T / pseudo_temperature, dim=1)

            conf_t, pred_t = probs_t.max(dim=1)
            conf_s, pred_s = probs_s.max(dim=1)

            agree = (pred_t == pred_s)
            avg_conf = (conf_t + conf_s) / 2
            confident = avg_conf > conf_threshold

            mask = agree & confident
            n_pass_agree += agree.sum().item()
            n_pass_conf += mask.sum().item()

            if not mask.any():
                continue

            sel_data = data[mask].cpu()
            sel_score = avg_conf[mask].cpu()
            sel_pseudo = pred_t[mask].cpu()

            for i in range(sel_data.shape[0]):
                c_active = sel_pseudo[i].item()
                c_orig = active_class_ids[c_active].item()
                candidates[c_orig].append((sel_score[i].item(), sel_data[i]))

        print(f"  Filter pass: agreement={n_pass_agree}/{n_total}, "
              f"+avg_conf>{conf_threshold}={n_pass_conf}")

        selected_data, selected_labels = [], []
        per_class_count = {}
        for c_orig, items in candidates.items():
            items.sort(key=lambda x: -x[0])
            items = items[:per_class_cap]
            per_class_count[c_orig] = len(items)
            for _, d in items:
                selected_data.append(d)
                selected_labels.append(c_orig)

        if not selected_data:
            return None, None, per_class_count

        pseudo_data = torch.stack(selected_data, dim=0).to(self.device)
        pseudo_labels = torch.tensor(selected_labels, dtype=torch.long, device=self.device)
        teacher_scorer.train()
        self.model.train()
        return pseudo_data, pseudo_labels, per_class_count

    def _compute_mean_teacher_loss(self, unlabeled_data, aug, active_class_ids):
        with torch.no_grad():
            teacher_result = self.teacher_model.return_output_contextGCN(
                unlabeled_data, deterministic=False
            )
            teacher_center = F.normalize(
                teacher_result['result'][:, self.center_idx, :], p=2, dim=1
            )

            proto_centers = self.contrastive_loss._get_active_proto_centers(active_class_ids)
            teacher_logits = torch.mm(teacher_center, proto_centers.T) / 0.1
            teacher_prob = F.softmax(teacher_logits, dim=1)
            
            confidence = teacher_prob.max(dim=1).values
            conf_mask = confidence > 0.7

        if not conf_mask.any():
            return torch.tensor(0.0, device=unlabeled_data.device)

        aug_data = aug.create_boundary_cutmix_unlabeled(unlabeled_data) # unlabeled-cutmix view
        student_result = self.model.return_output_contextGCN(aug_data, deterministic=False)
        student_center = F.normalize(
            student_result['result'][:, self.center_idx, :], p=2, dim=1
        )
        student_logits = torch.mm(student_center, proto_centers.T) / 0.1
        student_log_prob = F.log_softmax(student_logits, dim=1)

        loss = F.kl_div(
            student_log_prob[conf_mask],
            teacher_prob[conf_mask].detach(),
            reduction='batchmean'
        )
        return loss

    def R_Round_iterative_training(self,
                                    train_loader, val_loader, test_loader,
                                    unlabeled_loader,
                                    val_interval=1,
                                    num_rounds=3,
                                    round0_epochs=400,
                                    round_epochs=200,
                                    conf_thresholds=(0.95, 0.9, 0.85),
                                    per_class_caps=(20, 40, 80),
                                    kl_weight=0.1,
                                    density_target=(0.35,0.55)):
        print(f" R-Round Iterative training: {num_rounds} rounds")

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        train_start = time.perf_counter()
        history = []

        for r in range(num_rounds):
            print(f"\n{'='*60}\n Round {r}/{num_rounds-1}\n{'='*60}")
            epochs = round0_epochs if r == 0 else round_epochs

            best_val_acc = self.contrastive_learning(
                train_loader=train_loader,
                val_loader=val_loader,
                # test_loader=test_loader,
                unlabeled_loader=unlabeled_loader,
                val_interval=val_interval,
                epochs=epochs,
                round_idx= r,
                kl_weight=kl_weight,
                density_target=density_target
            )

            history.append({
                'round': r,
                'val_acc': best_val_acc,
            })

            print(f" Round {r} done: val_acc={best_val_acc:.4f}")

            if r == num_rounds - 1:
                break

            active_class_ids = torch.where(self.contrastive_loss.prototype_initialized)[0]
            conf_th = conf_thresholds[min(r, len(conf_thresholds)-1)]
            cap = per_class_caps[min(r, len(per_class_caps)-1)]
            print(f" Generating pseudo pool: avg_conf>{conf_th}, cap={cap}")

            pseudo_data, pseudo_labels, per_class_count = self._generate_pseudo_pool_consistent(
                unlabeled_loader, conf_th, cap, active_class_ids
            )
            
            if pseudo_data is not None:
                self.pseudo_pool_data = pseudo_data
                self.pseudo_pool_labels = pseudo_labels
                print(f"  Pseudo pool: {len(pseudo_data)} samples")
                print(f"  Per-class: {sorted(per_class_count.items())}")
            else:
                print("[pseudo datas is None]")

        print(f"\n{'='*60}\n R-Round iterative training history:")
        for h in history:
            print(f"  Round {h['round']}: val={h['val_acc']:.4f}")
        print(f"{'='*60}")

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        train_time = time.perf_counter() - train_start
        print(f" Total train time: {train_time:.2f}s")

        self._load_checkpoint()
        self.unfreeze_final_layers()

        self.model.classifier.set_prototype_weights(
            self.contrastive_loss.prototypes,
            center_idx=self.center_idx,
            initialized_mask=self.contrastive_loss.prototype_initialized,
        )

        final_path = os.path.join(self.DATASET, 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, final_path)
        print(f"  Final model (with prototype classifier) saved: {final_path}")

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        test_start = time.perf_counter()

        oa, aa, kappa, per_class_acc_dict = self.compute_test_metrics(test_loader)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        test_time = time.perf_counter() - test_start
        print(f" Test time: {test_time:.2f}s")

        self.round_dict = {
            "test_OA": oa, "test_AA": aa, "test_Kappa": kappa,
            "test_per_class_accuracy": per_class_acc_dict, "val_acc": best_val_acc,
            "train_time_sec": train_time, "test_time_sec": test_time,
        }

        return history

    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher_model.parameters(), 
                                    self.model.parameters()):
            t_param.data.mul_(self.teacher_momentum).add_(
                s_param.data, alpha=1 - self.teacher_momentum
            )
        if self.byol_trainer.byol_head_teacher is not None:
            for t_param, s_param in zip(self.byol_trainer.byol_head_teacher.parameters(),
                                        self.byol_trainer.byol_head_student.parameters()):
                t_param.data.mul_(self.teacher_momentum).add_(
                    s_param.data, alpha=1 - self.teacher_momentum
                )

    def _load_checkpoint(self):
        ckpt_path = os.path.join(self.DATASET, 'round_model.pth')
        if os.path.exists(ckpt_path):
            print(f"  Loading best state for round continuation: {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.device)
                
            self.model.load_state_dict(ckpt['model_state_dict'])
            
            self.contrastive_loss.load_state_dict(ckpt['contrastive_loss_state_dict'])

            if ckpt.get('teacher_model_state_dict') is not None:
                if not hasattr(self, 'teacher_model') or self.teacher_model is None:
                    self.teacher_model = copy.deepcopy(self.model)
                    for p in self.teacher_model.parameters():
                        p.requires_grad = False
                self.teacher_model.load_state_dict(ckpt['teacher_model_state_dict'])
                self.teacher_momentum = 0.999

    def _module_reset(self):
        self.teacher_model = None
        self.byol_trainer._byol_module_reset()

    def recon_loss(self, recon_output, original_patch):
        target = original_patch.reshape(-1, self.node_size, self.raw_bands)
        return F.mse_loss(recon_output, target)

    def contrastive_learning(self,
                            train_loader, val_loader, unlabeled_loader=None, val_interval=1,
                            epochs=150,
                            round_idx=0,
                            lr=2e-4,
                            weight_decay=2e-2,
                            contrastive_temperature=0.1,
                            contrastive_momentum=0.95,
                            node_level_mining=4,
                            kl_weight=0.1,
                            density_target=(0.35,0.55),
                            warmup_temp=1.2,
                            final_temp=0.2,
                            temp_decay_epochs=300,
                            ):
        print(f" Round {round_idx}: Prototype-Driven Contrastive Learning Start!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if round_idx != 0:
            self._module_reset()
            self._recon_head_reset(self.device)

        if round_idx == 0 or not hasattr(self, 'contrastive_loss'):
            self.contrastive_loss = GraphPrototypeContrastiveLoss(temperature=contrastive_temperature, 
                                                                  momentum=contrastive_momentum,
                                                                  node_level_mining=node_level_mining,
                                                                  num_classes=self.num_classes, 
                                                                  node_size=self.node_size, 
                                                                  feature_dim=self.target_feature)
            
            self.Regularization_Loss_criterion = Regularization_Loss(kl_weight=kl_weight,density_target=density_target)
            self.freeze_final_layers()

        self.optimizer = torch.optim.AdamW([
            {'params': self.model.spatial_spectral.parameters(),'lr': lr, 'weight_decay': weight_decay},
            {'params': self.model.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.model.decoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.model.context_gcn1.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.model.context_gcn2.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.byol_trainer.byol_head_student.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.recon_head.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        ])
    
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=200, T_mult=3, eta_min=15e-6)

        if unlabeled_loader:
            unlabeled_iter = iter(unlabeled_loader)

        mt_activate_cond = False

        for epoch in range(epochs):
            if epoch < temp_decay_epochs:
                current_temp = warmup_temp - (warmup_temp - final_temp) * (epoch / temp_decay_epochs)
            else:
                current_temp = final_temp
            self.model.decoder.temperature.data.fill_(current_temp)

            self.model.train()
            self.contrastive_loss.train()
            self.byol_trainer.byol_head_student.train()
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                data_clean, labels = data.to(self.device), labels.to(self.device)
                
                if self.pseudo_pool_data is not None and len(self.pseudo_pool_data) > 0:
                    pool_size = len(self.pseudo_pool_data)
                    n_add = min(self.BATCH_SIZE // 2, pool_size)
                    idx = torch.randint(0, pool_size, (n_add,), device=self.device)
                    pseudo_d = self.pseudo_pool_data[idx]
                    pseudo_l = self.pseudo_pool_labels[idx]
                    data_clean = torch.cat([data_clean, pseudo_d], dim=0)
                    labels = torch.cat([labels, pseudo_l], dim=0)
                    n_pseudo_in_batch = n_add
                else:
                    n_pseudo_in_batch = 0

                try:
                    unlabeled_data_raw, _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_data_raw, _ = next(unlabeled_iter)
                unlabeled_data_raw = unlabeled_data_raw.to(self.device)

                data = self.aug(data_clean, mode="strong", labels=labels)
                
                labeled_result = self.model.return_output_contextGCN(data, deterministic=False)
                mu, logvar = labeled_result['mu'], labeled_result['logvar']
                adj_matrices, node_feats = labeled_result['adj_matrices'], labeled_result['node_features']
                gcn2 = labeled_result['result']
                
                clean_result = self.model.return_output_contextGCN(data_clean, deterministic=False)
                clean_mu, clean_logvar = clean_result['mu'], clean_result['logvar']
                clean_node_features, clean_adj_matrices = clean_result['node_features'], clean_result['adj_matrices']
                clean_gcn2 = clean_result['result']

                unlabeled_weak = self.aug(unlabeled_data_raw.clone(), mode='weak', labels=None)
                unlabeled_strong = self.aug(unlabeled_data_raw.clone(), mode='strong', labels=None)

                unlabeled_result = self.model.return_output_contextGCN(unlabeled_weak, deterministic=False)
                student_strong_result = self.model.return_output_contextGCN(unlabeled_strong, deterministic=False)

                unlabeled_mu, unlabeled_logvar = unlabeled_result['mu'], unlabeled_result['logvar']
                unlabeled_adj_matrices, unlabeled_node_feats = unlabeled_result['adj_matrices'], unlabeled_result['node_features']
                unlabeled_gcn2 = unlabeled_result['result']

                strong_unlabeled_mu, strong_unlabeled_logvar = student_strong_result['mu'], student_strong_result['logvar']
                strong_unlabeled_adj_matrices, strong_unlabeled_node_feats = student_strong_result['adj_matrices'], student_strong_result['node_features']

                stacked_node_features = torch.cat([node_feats, clean_node_features, unlabeled_node_feats, strong_unlabeled_node_feats], dim=0)
                stacked_data = torch.cat([data, data_clean, unlabeled_weak, unlabeled_strong], dim=0)

                recon_loss = self.recon_loss(self.recon_head(stacked_node_features), stacked_data)

                contrastive_total_loss, stats = self.contrastive_loss(
                    gcn2, labels, adj_matrices, unlabeled_gcn2, proto_update=True,
                    n_pseudo=n_pseudo_in_batch,
                    clean_gcn2 = clean_gcn2
                )
                mt_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                if not mt_activate_cond:
                    mt_activate_cond = True
                    
                    if not hasattr(self, 'teacher_model') or self.teacher_model is None:
                        self.teacher_model = copy.deepcopy(self.model)
                        for p in self.teacher_model.parameters():
                            p.requires_grad = False
                        print("    Teacher model created (fresh)")
                    
                    self.teacher_momentum = 0.999

                    if self.byol_trainer.byol_head_teacher is None:
                        self.byol_trainer.byol_head_teacher = copy.deepcopy(self.byol_trainer.byol_head_student)
                        for p in self.byol_trainer.byol_head_teacher.parameters():
                            p.requires_grad = False
                        print("    BYOL teacher head created (fresh)")

                active_class_ids = torch.where(self.contrastive_loss.prototype_initialized)[0]
                mt_loss = self._compute_mean_teacher_loss(unlabeled_data_raw, self.aug, active_class_ids)

                if hasattr(self, 'teacher_model') and self.teacher_model is not None \
                        and self.byol_trainer.byol_head_teacher is not None:
                    with torch.no_grad():
                        teacher_weak_result = self.teacher_model.return_output_contextGCN(
                            unlabeled_weak, deterministic=False)
                        teacher_strong_result = self.teacher_model.return_output_contextGCN(
                            unlabeled_strong, deterministic=False)

                    s_weak_c = unlabeled_result['result'][:, self.center_idx, :]
                    s_strong_c = student_strong_result['result'][:, self.center_idx, :]
                    t_weak_c = teacher_weak_result['result'][:, self.center_idx, :]
                    t_strong_c = teacher_strong_result['result'][:, self.center_idx, :]

                    byol_loss_val, nn_byol_loss_val = self.byol_trainer.compute_byol_losses_from_feats(
                        s_strong_c, s_weak_c, t_weak_c, t_strong_c,
                        current_epoch=epoch, round_idx=round_idx)
                else:
                    byol_loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)
                    nn_byol_loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)

                if round_idx > 0:
                    current_nn_weight = self.nn_byol_weight
                else:
                    if epoch < self.byol_trainer.nn_byol_warmup_epochs:
                        current_nn_weight = 0.0
                    elif epoch < self.byol_trainer.nn_byol_warmup_epochs + 50:
                        progress = (epoch - self.byol_trainer.nn_byol_warmup_epochs) / 50.0
                        current_nn_weight = self.byol_trainer.nn_byol_weight * progress
                    else:
                        current_nn_weight = self.byol_trainer.nn_byol_weight

                logvar = torch.cat([logvar, clean_logvar, unlabeled_logvar, strong_unlabeled_logvar], dim=0)
                mu = torch.cat([mu, clean_mu, unlabeled_mu, strong_unlabeled_mu], dim=0)
                adj_matrices = torch.cat([adj_matrices, clean_adj_matrices, unlabeled_adj_matrices, strong_unlabeled_adj_matrices], dim=0)
                
                regularization_loss, kl_loss, density_loss, current_density = self.Regularization_Loss_criterion(
                    logvar, mu, adj_matrices
                )

                total_loss_batch = (
                    contrastive_total_loss 
                    + regularization_loss 
                    + self.mt_loss_weight * mt_loss
                    + self.byol_weight * byol_loss_val
                    + current_nn_weight * nn_byol_loss_val
                    + recon_loss
                )

                self.optimizer.zero_grad()
                total_loss_batch.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                self.optimizer.step()
                self._update_teacher()

                if epoch % 100 == 0:
                    print(f"{round_idx}R-Epoch {epoch}, Batch {batch_idx}: "
                        f"Total={total_loss_batch.item():.4f}, "
                        f"Contr={stats['loss_labeled_total']:.4f}(N/C={stats['loss_node']:.2f}/{stats['loss_center']:.2f}), "
                        f"KL={kl_loss.item():.4f}, "
                        f"density={density_loss.item():.4f}({current_density:.4f}), "
                        f"neg_loss={stats['neg_expansion_loss']:.2f}, "
                        f"mt_loss={mt_loss.item():.2f}, "
                        f"byol={byol_loss_val.item():.3f}, "
                        f"nn_byol={nn_byol_loss_val.item():.3f}(w={current_nn_weight:.2f}), "
                        f"queue={int(self.byol_trainer.nn_bank.num_stored.item())}, "
                        f"recon={recon_loss.item():.2f}")
            if epoch % val_interval == 0:
                val_acc, val_adj = self.compute_classification_metric(val_loader)
                
                if (val_acc > self.best_val_acc):
                    print(f"  $$$$ validation acc: {val_acc:.4f}, ND={val_adj:.4f} $$$$")
                    self.best_val_acc = val_acc

                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'contrastive_loss_state_dict': self.contrastive_loss.state_dict(),
                        'best_val_acc': self.best_val_acc,
                        'r': round_idx,
                        'epoch': epoch,

                        'teacher_model_state_dict': (
                            self.teacher_model.state_dict() 
                            if hasattr(self, 'teacher_model') and self.teacher_model is not None 
                            else None
                        ),
                    }
                    torch.save(checkpoint, os.path.join(self.DATASET, 'round_model.pth'))
            self.scheduler.step()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._load_checkpoint()

        print(f" Prototype learning completed! Best validation accuracy: {self.best_val_acc:.4f}")

        return self.best_val_acc

    def compute_classification_metric(self, val_loader):
        initialized_mask = self.contrastive_loss.prototype_initialized.clone()

        if initialized_mask.sum() < 2:
            return 0.0, 0.0
        
        active_class_ids = torch.where(initialized_mask)[0]
        label_to_active = torch.full((self.contrastive_loss.num_classes,), -1, 
                                    dtype=torch.long, device=self.device)
        label_to_active[active_class_ids] = torch.arange(len(active_class_ids), device=self.device)
        
        active_prototypes = self.contrastive_loss.prototypes[active_class_ids].clone()
        active_graph_prototypes = active_prototypes[:,self.center_idx,:]
        active_graph_prototypes = self.contrastive_loss.normalize_graph_features(active_graph_prototypes)
        
        total_correct = 0
        total_valid_samples = 0
        
        num = 0
        adj_mean = 0
        self.model.eval()
        self.contrastive_loss.eval()
        self.byol_trainer.byol_head_student.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                num += 1
                data, labels = data.to(self.device), labels.to(self.device)
                
                result = self.model.return_output_contextGCN(data, deterministic=True)
                context_z = result['result']
                adj_matrices = result['adj_matrices']
         
                adj_mean += torch.mean(adj_matrices)

                graph_features = context_z[:,self.center_idx,:]
                graph_features = self.contrastive_loss.normalize_graph_features(graph_features)
                
                similarities = torch.mm(graph_features, active_graph_prototypes.T)
                
                active_predictions = similarities.argmax(dim=1)
                class_predictions = active_class_ids[active_predictions]

                active_label_indices = label_to_active[labels]
                valid_mask = active_label_indices >= 0
                
                if valid_mask.any():
                    valid_predictions = class_predictions[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    correct = (valid_predictions == valid_labels).sum().item()
                    total_correct += correct
                    
                    total_valid_samples += len(valid_labels)
        adj_mean = adj_mean/ num

        if total_valid_samples == 0:
            return 0.0, adj_mean
        accuracy = total_correct / total_valid_samples
        return 1.0 * accuracy, adj_mean

    def compute_test_metrics(self, test_loader, only_test=False):
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                result = self.model(data, deterministic=True)
                logits = result['predictions']
                class_predictions = logits.argmax(dim=1)

                all_preds.append(class_predictions.cpu())
                all_labels.append(labels.cpu())

        if len(all_preds) == 0:
            return 0.0, 0.0, 0.0, 0.0

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        oa = float((all_preds == all_labels).mean())

        unique_labels = np.unique(all_labels)
        per_class_acc = np.zeros(len(unique_labels))
        for i, c in enumerate(unique_labels):
            mask = all_labels == c
            per_class_acc[i] = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0
        aa = float(per_class_acc.mean())

        kappa = float(cohen_kappa_score(all_labels, all_preds))
        if not only_test:
            print(f"\n{'='*60}")
            print(f"  Test Metrics (classifier forward)")
            print(f"{'='*60}")
            print(f"  OA    : {oa*100:.2f}%")
            print(f"  AA    : {aa*100:.2f}%")
            print(f"  Kappa : {kappa*100:.2f}")
            print(f"  Per-class accuracy:")
        per_class_acc_dict = {}
        for i, c in enumerate(unique_labels):
            n = int((all_labels == c).sum())
            if not only_test:
                print(f"    Class {int(c):2d}: {per_class_acc[i]*100:6.2f}%  (n={n})")
            per_class_acc_dict[int(c)] = float(per_class_acc[i])
        if not only_test:
            print(f"{'='*60}\n")

        return oa, aa, kappa, per_class_acc_dict

    def freeze_final_layers(self):
        self.model.classifier.eval()
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        print(" Final layers frozen")
    
    def unfreeze_final_layers(self):
        self.model.classifier.train()
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        print(" Final layers unfrozen")

    @torch.no_grad()
    def save_classification_map(self, all_loader, save_path, save_gt_path=None):
        ds = all_loader.dataset
        H = ds.label.shape[0] - 2 * ds.p
        W = ds.label.shape[1] - 2 * ds.p
        pred_map = np.zeros((H, W), dtype=np.int64)
        gt_map = np.zeros((H, W), dtype=np.int64)

        self.model.eval()
        for data, labels, coords in all_loader:
            data = data.to(self.device)
            logits = self.model(data, deterministic=True)['predictions']
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            xs = coords[:, 0].numpy()
            ys = coords[:, 1].numpy()
            pred_map[xs, ys] = preds
            gt_map[xs, ys] = labels.numpy() + 1

        palette = CLASSIFICATION_PALETTE
        max_id = int(max(pred_map.max(), gt_map.max()))
        if max_id >= palette.shape[0]:
            extra = plt.cm.tab20(np.linspace(0, 1, max_id + 1 - palette.shape[0]))[:, :3]
            palette = np.concatenate([palette, extra.astype(np.float32)], axis=0)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.imsave(save_path, palette[pred_map])
        print(f"  Classification map saved: {save_path}")
        if save_gt_path is not None:
            plt.imsave(save_gt_path, palette[gt_map])
            print(f"  GT map saved: {save_gt_path}")
