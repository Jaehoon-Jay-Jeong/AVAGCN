import torch
import numpy as np

import os
import collections
import json

import hydra
from hydra.core.hydra_config import HydraConfig

from dataset.dataset import get_HSIDataLoaders
from model.AVAGCN import AVAGCN
from training.trainer import SemiSupervisedTrainer

from util.utils import seed_generate, set_seed

import logging
log = logging.getLogger(__name__)
print = log.info

@hydra.main(config_path="configs", version_base=None)
def run(config):
    output_dir = HydraConfig.get().run.dir
    DATASET = config.DATASET
    DEVICE = config.DEVICE

    oa_list, aa_list, kappa_list, train_time_list, test_time_list = [], [], [], [], []
    test_only_oa_list, test_only_aa_list, test_only_kappa_list = [], [], []
    class_acc_dict = collections.defaultdict(list)
    test_only_class_acc_dict = collections.defaultdict(list)

    RANDOM_SEEDS =  seed_generate()
    print(f"{RANDOM_SEEDS=}")
    for RANDOM_SEED in RANDOM_SEEDS:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"NOW {RANDOM_SEED=}")

        CLASS_PER_TRAIN = config.CLASS_PER_TRAIN

        PATCH_SIZE = config.PATCH_SIZE
        CLASS_PER_BATCH = config.CLASS_PER_BATCH
        SAMPLES_PER_CLASS = config.SAMPLES_PER_CLASS
        BATCH_REPEAT = config.BATCH_REPEAT
        TEST_BATCH_SIZE = config.TEST_BATCH_SIZE
        BATCH_SIZE = CLASS_PER_BATCH * SAMPLES_PER_CLASS

        SAVE_PATH = os.path.join(output_dir, f"train_per_{CLASS_PER_TRAIN}", f"seed_{RANDOM_SEED}")
        
        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)

        set_seed(RANDOM_SEED)

        train_loader, val_loader, unlabeled_loader, test_loader, all_loader, test_only_loader, NUM_CLASS, BANDS = get_HSIDataLoaders(
            DATASET, CLASS_PER_TRAIN, dict(config.VALID_BY_CLASS), PATCH_SIZE, 
            CLASS_PER_BATCH, SAMPLES_PER_CLASS, BATCH_REPEAT, TEST_BATCH_SIZE)

        device = torch.device(f'cuda:{DEVICE}' if DEVICE>=0 and torch.cuda.is_available() else 'cpu')

        hidden_dim = config["hidden_dim"]
        latent_dim = config["latent_dim"]
        target_features = config["target_features"]
        dropout = config["dropout"]

        kl_weight = config.kl_weight
        density_target = config["density_target"]

        set_seed(RANDOM_SEED)

        model = AVAGCN(input_dim=BANDS, 
                        hidden_dim=hidden_dim,
                        latent_dim=latent_dim,
                        num_classes=NUM_CLASS, 
                        patch_size=PATCH_SIZE, 
                        target_features=target_features,
                        dropout=dropout,
                        )
        
        mt_loss_weight = config["mt_loss_weight"]
        byol_weight = config["byol_weight"]
        nn_byol_weight = config["nn_byol_weight"]
        
        set_seed(RANDOM_SEED)
        
        trainer = SemiSupervisedTrainer(model, 
                                        SAVE_PATH, 
                                        PATCH_SIZE=PATCH_SIZE, 
                                        device=device, 
                                        num_classes=NUM_CLASS, 
                                        target_feature=target_features, 
                                        raw_bands=BANDS, 
                                        BATCH_SIZE=BATCH_SIZE,
                                        mt_loss_weight=mt_loss_weight,
                                        byol_weight=byol_weight,
                                        nn_byol_weight=nn_byol_weight,
                                        )
        
        set_seed(RANDOM_SEED)
            
        history= trainer.R_Round_iterative_training(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            unlabeled_loader=unlabeled_loader,
            val_interval=1,
            num_rounds=config.num_round,
            round0_epochs=config.round0_epochs,
            round_epochs=config.round_epochs,
            conf_thresholds=config.conf_thresholds,
            per_class_caps=config.per_class_caps,
            kl_weight=kl_weight,
            density_target=density_target,
        )

        oa, aa, kappa, class_acc = trainer.round_dict["test_OA"], trainer.round_dict["test_AA"], trainer.round_dict["test_Kappa"], trainer.round_dict["test_per_class_accuracy"]
        best_val_acc, train_time, test_time = trainer.round_dict["val_acc"], trainer.round_dict["train_time_sec"], trainer.round_dict["test_time_sec"]
        print(f"Iterative Training best validation accuracy: {best_val_acc:.4f}")
        print(f"{RANDOM_SEED=}: "
            f"{oa=}, "
            f"{aa=}, "
            f"{kappa=}")
        
        with open(os.path.join(SAVE_PATH, "round_results.json"), "w") as f:
            json.dump({'round val acc': best_val_acc,
            'final_test_acc': oa,
            'train_time_sec': train_time,
            'test_time_sec': test_time,
            'history': history}, f, indent=2)

        map_path = os.path.join(SAVE_PATH, f"{DATASET}_map_seed{RANDOM_SEED}.png")
        trainer.save_classification_map(all_loader, map_path,
                                        save_gt_path=os.path.join(SAVE_PATH, f"{DATASET}_gt.png"))

        oa_list.append(oa)
        aa_list.append(aa)
        kappa_list.append(kappa)
        train_time_list.append(train_time)
        if test_time is not None:
            test_time_list.append(test_time)
        if class_acc:
            for key in class_acc.keys():
                class_acc_dict[key].append(class_acc[key])
                class_acc_dict[f"{key}_mean"] = np.mean(class_acc_dict[key])
                class_acc_dict[f"{key}_std"] = np.std(class_acc_dict[key])

        test_only_oa, test_only_aa, test_only_kappa, test_only_class_acc = trainer.compute_test_metrics(test_only_loader, only_test=True)
        test_only_oa_list.append(test_only_oa)
        test_only_aa_list.append(test_only_aa)
        test_only_kappa_list.append(test_only_kappa)
        if test_only_class_acc:
            for key in test_only_class_acc.keys():
                test_only_class_acc_dict[key].append(test_only_class_acc[key])
                test_only_class_acc_dict[f"{key}_mean"] = np.mean(test_only_class_acc_dict[key])
                test_only_class_acc_dict[f"{key}_std"] = np.std(test_only_class_acc_dict[key])
            
    print(f"Overall accuracy: mean:{np.mean(oa_list)}, std: std:{np.std(oa_list)}")
    print(f"Average accuracy: mean:{np.mean(aa_list)}, std: std:{np.std(aa_list)}")
    print(f"Kappa: mean:{np.mean(kappa_list)}, std: std:{np.std(kappa_list)}")

    doc = {
        "RANDOM SEEDS": RANDOM_SEEDS,
        "Overall accuracy": {"mean": np.mean(oa_list), "std": np.std(oa_list)},
        "Average accuracy": {"mean": np.mean(aa_list), "std": np.std(aa_list)},
        "Kappa accuracy": {"mean": np.mean(kappa_list), "std": np.std(kappa_list)},
        "Train time (s)": {"mean": np.mean(train_time_list), "std": np.std(train_time_list)},
        "Test time (s)": {"mean": np.mean(test_time_list), "std": np.std(test_time_list)} if test_time_list else None,
        "Class accuracy": class_acc_dict,
        "Overall accuracy List": oa_list,
        "Average accuracy List": aa_list,
        "Kappa accuracy List": kappa_list,
        "Train time List": train_time_list,
        "Test time List": test_time_list,
        }

    with open(os.path.join(output_dir, "full_seed_results.json"), "w") as f:
        json.dump(doc, f, indent=2)
    
    test_only_doc = {
        "RANDOM SEEDS": RANDOM_SEEDS,
        "Overall accuracy": {"mean": np.mean(test_only_oa_list), "std": np.std(test_only_oa_list)},
        "Average accuracy": {"mean": np.mean(test_only_aa_list), "std": np.std(test_only_aa_list)},
        "Kappa accuracy": {"mean": np.mean(test_only_kappa_list), "std": np.std(test_only_kappa_list)},
        "Class accuracy": test_only_class_acc_dict,
        "Overall accuracy List": test_only_oa_list,
        "Average accuracy List": test_only_aa_list,
        "Kappa accuracy List": test_only_kappa_list,
        }
            
    with open(os.path.join(output_dir, "full_seed_test_only_results.json"), "w") as f:
        json.dump(test_only_doc, f, indent=2)

if __name__ == '__main__':
    run()
    