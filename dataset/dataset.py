import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import spectral
from scipy.io import loadmat
import matplotlib.pyplot as plt

import random
import os

import logging
log = logging.getLogger(__name__)
print = log.info

CLASSIFICATION_PALETTE = np.array([
    [0,    0,    0   ],
    [0,    0,    1   ], 
    [0,    1,    0   ], 
    [0,    1,    1   ], 
    [1,    0,    0   ],
    [1,    0,    1   ], 
    [1,    1,    0   ], 
    [0.5,  0.5,  1   ], 
    [0.65, 0.35, 1   ],
    [0.75, 0.5,  0.75], 
    [0.75, 1,    0.5 ], 
    [0.5,  1,    0.65], 
    [0.65, 0.65, 0   ],
    [0.75, 1,    0.65], 
    [0,    0,    0.5 ], 
    [0,    1,    0.75], 
    [0.5,  0.75, 1   ],
], dtype=np.float32)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes_per_batch, n_samples_per_class, n_repeats=1):
        self.labels = labels
        self.n_classes = n_classes_per_batch
        self.n_samples = n_samples_per_class
        self.n_repeats = n_repeats
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label.item(), []).append(idx)
    
    def __iter__(self):
        classes = list(self.label_to_indices.keys())
        for _ in range(self.n_repeats):
            random.shuffle(classes)
            
            for i in range(0, len(classes), self.n_classes):
                batch_classes = classes[i:i+self.n_classes]

                batch = []
                for cls in batch_classes:
                    indices = self.label_to_indices[cls]
                    sampled = random.sample(indices, k=self.n_samples)
                    batch.extend(sampled)
                yield batch

def get_palette(max_class_id):
    palette = CLASSIFICATION_PALETTE
    if max_class_id >= palette.shape[0]:
        extra = plt.cm.tab20(np.linspace(0, 1, max_class_id + 1 - palette.shape[0]))[:, :3]
        palette = np.concatenate([palette, extra.astype(np.float32)], axis=0)
    return palette

def loadData(name):
    data_path = os.path.join(os.getcwd(),'datas')
    if name == 'IP':
        mat_path = os.path.join('IndianPines', 'Indian_pines_corrected.mat')
        labels_path = os.path.join('IndianPines', 'Indian_pines_gt.mat')
        data = loadmat(os.path.join(data_path, mat_path))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, labels_path))['indian_pines_gt']
        class_name = [ "Alfalfa", "Corn-notill", "Corn-mintill","Corn", "Grass-pasture", 
                       "Grass-trees","Grass-pasture-mowed", "Hay-windrowed", "Oats","Soybean-notill", "Soybean-mintill", "Soybean-clean",
                       "Wheat", "Woods", "Buildings-Grass-Trees-Drives","Stone-Steel-Towers"]
    elif name == 'PU':
        mat_path = os.path.join('PaviaU', 'PaviaU.mat')
        labels_path = os.path.join('PaviaU', 'PaviaU_gt.mat')
        data = loadmat(os.path.join(data_path, mat_path))['paviaU']
        labels = loadmat(os.path.join(data_path, labels_path))['paviaU_gt']
        class_name = ['Asphalt', 'Meadows', 'Gravel', 'Trees','Painted metal sheets', 'Bare Soil', 
                      'Bitumen','Self-Blocking Bricks', 'Shadows']
    elif name == 'SA':
            mat_path = os.path.join('Salinas', 'Salinas_corrected.mat')
            labels_path = os.path.join('Salinas', 'Salinas_gt.mat')
            data = loadmat(os.path.join(data_path, mat_path))['salinas_corrected']
            labels = loadmat(os.path.join(data_path, labels_path))['salinas_gt']
            class_name = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow',
                            'Fallow_rough_plow','Fallow_smooth','Stubble','Celery','Grapes_untrained',
                            'Soil_vinyard_develop','Corn_senesced_green','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk',
                            'Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical']
    # elif name == 'KSC':
    #     mat_path = os.path.join('KSC', 'KSC_corrected.mat')
    #     labels_path = os.path.join('KSC', 'KSC_gt.mat')
    #     data = loadmat(os.path.join(data_path, mat_path))['KSC']
    #     labels = loadmat(os.path.join(data_path, labels_path))['KSC_gt']
    #     class_name = ['Scrub', 'Willow_swamp', 'Cabbage_palm_hammock', 'Cabbage_palm/oak_hammock', 'Slash_pine', 
    #                   'Oak/broad_leaf_hammock', 'Hardwood_swamp', 'Graminoid_marsh', 'Spartina_marsh',
    #                   'Cattail_marsh', 'Salt_marsh', 'Mud_flats', 'Water']
    # elif name == "WHU-Hi-LongKou":
    #     mat_path = os.path.join('WHU-Hi-LongKou', 'WHU_Hi_LongKou.mat')
    #     labels_path = os.path.join('WHU-Hi-LongKou', 'WHU_Hi_LongKou_gt.mat')
    #     data = loadmat(os.path.join(data_path, mat_path))['WHU_Hi_LongKou']
    #     labels = loadmat(os.path.join(data_path, labels_path))['WHU_Hi_LongKou_gt']
    #     class_name = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean','Narrow-leaf soybean', 'Rice', 
    #                   'Water','Roads and houses', 'Mixed weed']
    # elif name == "WHU-Hi-HanChuan":
    #     mat_path = os.path.join('WHU-Hi-HanChuan', 'WHU_Hi_HanChuan.mat')
    #     labels_path = os.path.join('WHU-Hi-HanChuan', 'WHU_Hi_HanChuan_gt.mat')
    #     data = loadmat(os.path.join(data_path, mat_path))['WHU_Hi_HanChuan']
    #     labels = loadmat(os.path.join(data_path, labels_path))['WHU_Hi_HanChuan_gt']
    #     class_name = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum','Water spinach', 'Watermelon', 
    #                   'Greens','Trees', 'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water']
    # elif name == "WHU-Hi-HongHu":
    #     mat_path = os.path.join('WHU-Hi-HongHu', 'WHU_Hi_HongHu.mat')
    #     labels_path = os.path.join('WHU-Hi-HongHu', 'WHU_Hi_HongHu_gt.mat')
    #     data = loadmat(os.path.join(data_path, mat_path))['WHU_Hi_HongHu']
    #     labels = loadmat(os.path.join(data_path, labels_path))['WHU_Hi_HongHu_gt']
    #     class_name = ['Red roof', 'Road', 'Bare soil', 'Cotton','Cotton firewood', 'Rape', 'Chinese cabbage','Pakchoi', 'Cabbage', 
    #                   'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 
    #                   'Celtuce', 'Film covered lettuce', 'Romaine lettuce','Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree']
    return data, labels, class_name

def sampling_gt(gt, n_per_class=15):
    indices = np.where(gt != -1)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    unlabeled_gt = np.zeros_like(gt)
    
    is_dict = isinstance(n_per_class, dict)
    
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    unlabeled_indices = []
    
    for cls in unique_classes:
        if cls == 0:
            cls_indices = np.where(y == cls)[0]
            cls_coords = [X[i] for i in cls_indices]
            unlabeled_indices.extend(cls_coords)
            continue
        
        if is_dict:
            n_samples = n_per_class.get(int(cls), 0)
        else:
            n_samples = n_per_class
            
        cls_indices = np.where(y == cls)[0]
        cls_coords = [X[i] for i in cls_indices]
        
        if n_samples == 0:
            test_samples = cls_coords
            train_samples = []
        elif len(cls_coords) <= n_samples:
            train_samples = cls_coords
            test_samples = []
        else:
            sample_indices = np.random.choice(len(cls_coords), n_samples, replace=False)
            train_samples = [cls_coords[i] for i in sample_indices]
            
            train_set = set(tuple(coord) for coord in train_samples)
            test_samples = [coord for coord in cls_coords if tuple(coord) not in train_set]
        
        train_indices.extend(train_samples)
        test_indices.extend(test_samples)
    
    if train_indices:
        train_indices = list(zip(*train_indices))
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    
    if test_indices:
        test_indices = list(zip(*test_indices))
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    
    if unlabeled_indices:
        unlabeled_indices = list(zip(*unlabeled_indices))
        unlabeled_gt[tuple(unlabeled_indices)] = 100
    
    return train_gt, test_gt, unlabeled_gt

class PatchSet(Dataset):
    def __init__(self, data, gt, patch_size, is_unlabeled=False, is_pred=False, return_coords=False):
        super(PatchSet, self).__init__()
        self.is_unlabeled = is_unlabeled
        self.patch_size = patch_size
        self.return_coords = return_coords
        self.p = p = self.patch_size // 2
        self.data = np.pad(data,((p,p),(p,p),(0,0)),'constant',constant_values = 0)
        self.label = np.pad(gt,(p,p),'constant',constant_values = 0)
        x_pos, y_pos = np.nonzero(gt)
        x_pos, y_pos = x_pos + p, y_pos + p
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]
        data = np.asarray(data, dtype='float32')
        if not self.is_unlabeled:
            label = np.asarray(label-1, dtype='int64')
        else:
            label = np.asarray(-1, dtype='int64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        if self.return_coords:
            orig_x = x - self.p
            orig_y = y - self.p
            coord = torch.tensor([orig_x, orig_y], dtype=torch.long)
            return data, label, coord
        else:
            return data, label


def get_HSIDataLoaders(DATASET, CLASS_PER_TRAIN, VALID_BY_CLASS, PATCH_SIZE, CLASS_PER_BATCH, SAMPLES_PER_CLASS, BATCH_REPEAT, TEST_BATCH_SIZE):
    UNLABELED_BATCH_SIZE = CLASS_PER_BATCH * SAMPLES_PER_CLASS
        
    data, label, class_name = loadData(DATASET)
    data = (data - np.mean(data, axis=(0,1), keepdims=True)) / np.std(data, axis=(0,1), keepdims=True)
    NUM_CLASS = label.max()
    BANDS = data.shape[-1]

    palette = get_palette(int(label.max()))
        
    if not os.path.exists('./map'):
        os.mkdir('./map')
    
    if not os.path.exists(f'map/{DATASET}_RGB_origin.jpg'):
        spectral.save_rgb(f'map/{DATASET}_RGB_origin.jpg', data, (30, 20, 10))
    if not os.path.exists(f'map/{DATASET}_gt.png'):
        plt.imsave(f'map/{DATASET}_gt.png', palette[label], dpi=300)
    
    print(f" Creating data splits and loaders...")
        
    train_gt, test_gt, unlabeled_gt = sampling_gt(label, CLASS_PER_TRAIN)
    val_gt, test_only_gt, _ = sampling_gt(test_gt, VALID_BY_CLASS)
        
    combined_test_gt = np.zeros_like(label)
    val_indices = np.where(val_gt != 0)
    combined_test_gt[val_indices] = val_gt[val_indices]
    test_indices = np.where(test_only_gt != 0)
    combined_test_gt[test_indices] = test_only_gt[test_indices]
        
    train_data = PatchSet(data, train_gt, PATCH_SIZE)
    val_data = PatchSet(data, val_gt, PATCH_SIZE)
    unlabeled_data = PatchSet(data, unlabeled_gt, PATCH_SIZE, is_unlabeled=True)
    test_data = PatchSet(data, combined_test_gt, PATCH_SIZE, is_pred=True)
    test_only_data = PatchSet(data, test_only_gt, PATCH_SIZE, is_pred=True)
    all_data = PatchSet(data, label, PATCH_SIZE, is_pred=True, return_coords=True)

    train_labels = []
    for x, y in train_data.indices:
        l = train_data.label[x, y]
        train_labels.append(l - 1)
        
    balanced_sampler = BalancedBatchSampler(
        labels=torch.tensor(train_labels),
        n_classes_per_batch=CLASS_PER_BATCH, 
        n_samples_per_class=SAMPLES_PER_CLASS,
        n_repeats=BATCH_REPEAT
    )

    train_loader = DataLoader(
        train_data, 
        batch_sampler=balanced_sampler,
        num_workers=0, 
        pin_memory=True
    )

    print(f"Unique labels in train: {np.unique(train_labels)}")
    val_loader = DataLoader(val_data, TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_data, UNLABELED_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_only_loader = DataLoader(test_only_data, TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    all_loader = DataLoader(all_data, TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
    print(f" Loaders created")
    print(f" Dataset Split Statistics:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test only samples: {len(test_only_data)}")
    print(f"  Combined test samples (val+test): {len(test_data)}")
    print(f"  Unlabeled samples: {len(unlabeled_data)}")
    
    return train_loader, val_loader, unlabeled_loader, test_loader, all_loader, test_only_loader, NUM_CLASS, BANDS
