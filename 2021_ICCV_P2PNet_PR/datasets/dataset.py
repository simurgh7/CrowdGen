import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import h5py
import torchvision.transforms as standard_transforms
from scipy.ndimage import gaussian_filter # Added for visualization utility, not strictly for dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, purpose="train", aug_dict=None, args=None):
        self.dataset = dataset
        self.purpose = purpose
        self.transform = transform
        self.aug_dict = aug_dict
        self.args = args

        # Hardcoded paths - consider making these configurable via args if they change often
        dataset_base_path = args.base_dir
        npy_data_dir = args.npy_dir

        if self.dataset == 'shha':
            self.base_path = os.path.join(dataset_base_path, 'shanghaitechA')
        elif self.dataset == 'ucf':
            self.base_path = os.path.join(dataset_base_path, 'ucf-qnrf')
        elif self.dataset == 'nwpu':
            self.base_path = os.path.join(dataset_base_path, 'nwpu-crowd')
        elif self.dataset == 'jhu':
            self.base_path = os.path.join(dataset_base_path, 'jhu-crowd++')
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        if self.purpose == "train":
            npy_file_path = os.path.join(npy_data_dir, f'{self.dataset}_train.npy')
            data_subdir = 'train_data'
        elif self.purpose == "val":
            npy_file_path = os.path.join(npy_data_dir, f'{self.dataset}_val.npy')
            data_subdir = 'val_data'
        elif self.purpose == "test":
            npy_file_path = os.path.join(npy_data_dir, f'{self.dataset}_test.npy')
            data_subdir = 'test_data'
        else:
            raise ValueError(f"Invalid purpose: {purpose}. Must be 'train', 'val', or 'test'.")

        self.image_subdir = os.path.join(self.base_path, data_subdir, 'images_2048')
        self.gt_subdir = os.path.join(self.base_path, data_subdir, 'gt-h5_2048')

        try:
            self.img_list = np.load(npy_file_path, allow_pickle=True).tolist()
        except FileNotFoundError:
            print(f"Error: .npy file not found at {npy_file_path}. Please check your path.")
            self.img_list = []

        self.img_map = {}
        for img_full_path in self.img_list:
            img_filename = os.path.basename(img_full_path)
            gt_filename = img_filename.replace('.jpg', '.h5').replace('.png', '.h5')
            gt_full_path = os.path.join(self.gt_subdir, gt_filename)

            if os.path.exists(gt_full_path):
                self.img_map[img_full_path] = gt_full_path
            else:
                print(f"Warning: Corresponding GT not found for {img_full_path} at {gt_full_path}. Skipping.")

        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)
        self.train = (purpose == "train")

        if self.aug_dict is not None:
            self.patch = 'Crop' in self.aug_dict.AUGUMENTATION
            self.flip = 'Flip' in self.aug_dict.AUGUMENTATION
            self.upper_bound = self.aug_dict.UPPER_BOUNDER
            self.crop_size = self.aug_dict.CROP_SIZE
            self.crop_number = self.aug_dict.CROP_NUMBER
        else:
            self.patch = False
            self.flip = False
            self.upper_bound = -1
            self.crop_size = 128
            self.crop_number = 4

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]

        # <--- MODIFIED: Load image, density map, and the kpoint_map (binary 2D map)
        img, density, kpoint_map_2d = load_image_density_kpoints_and_map((img_path, gt_path))

        if self.transform is not None:
            img = self.transform(img)

        # Convert density map to tensor
        density = torch.from_numpy(density).unsqueeze(0).float() # Density becomes 1, H, W

        # <--- NEW: Convert kpoint_map_2d to Nx2 keypoint coordinates
        # Find (y, x) coordinates where the map is 1
        y_coords, x_coords = np.where(kpoint_map_2d == 1)
        # Combine into an Nx2 array, assuming (x, y) format is desired for kpoints
        # (original process_shha_h5 uses [0] for x, [1] for y, so (x, y) is consistent)
        kpoints = np.stack((x_coords, y_coords), axis=1).astype(np.float32)
        kpoints = torch.from_numpy(kpoints).float() # kpoints are now Nx2 tensor


        # Prepare image_id
        try:
            image_id = int(os.path.basename(img_path).split('.')[0].split('_')[-1])
        except ValueError:
            image_id = int(os.path.basename(img_path).split('.')[0])

        if self.train:
            scale_range = [0.7, 1.3]
            _, H, W = img.shape
            min_size = min(H, W)
            max_size = max(H, W)
            scale = random.uniform(*scale_range)

            # Complex scaling logic based on upper_bound
            if self.upper_bound != -1 and max_size > self.upper_bound:
                scale = random.uniform(self.upper_bound / max_size - 0.1, self.upper_bound / max_size)
            elif self.upper_bound != -1: 
                scale = random.uniform(*scale_range)
            else: 
                scale = random.uniform(0.7, 1.0)

            if scale * min_size > self.crop_size: # Only resize if the scaled image is larger than crop_size
                new_h = int(H * scale)
                new_w = int(W * scale)
                
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                density = torch.nn.functional.interpolate(density.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                density = density * (scale * scale) # Adjust density values after scaling

                # <--- MODIFIED: Scale keypoints (Nx2 array)
                if kpoints.numel() > 0: # Only scale if there are points
                    kpoints = kpoints * scale 

            if self.patch:
                # <--- MODIFIED: Call the new random_crop_transform
                img, density, kpoints = random_crop_transform(img, density, kpoints, num_patch=1, crop_size=self.crop_size)
                img = img.squeeze(0) # Remove the batch dimension added by random_crop_transform
                density = density.squeeze(0) # Remove the batch dimension
                # kpoints is already Nx2, no extra batch dim added by random_crop_transform

            # Flip after all other augmentations, if enabled
            if self.flip and random.random() > 0.5:
                img = torch.flip(img, dims=[2]) # Flip along width (C, H, W) -> dim 2
                density = torch.flip(density, dims=[2]) # Flip along width (C, H, W) -> dim 2

                # <--- MODIFIED: Flip keypoints along width (Nx2 array)
                if kpoints.numel() > 0: # Only flip if there are points
                    current_width = img.shape[2]
                    kpoints[:, 0] = current_width - kpoints[:, 0] # Assuming kpoints are (x, y)
                                                                  # If kpoints were (y, x), then kpoints[:, 1]
        else: # Validation/Test processing
            _, H, W = img.shape
            max_size = max(H, W)
            if self.upper_bound != -1 and max_size > self.upper_bound:
                scale = self.upper_bound / max_size
            elif max_size > 2560: 
                scale = 2560 / max_size
            else:
                scale = 1.0

            if scale != 1.0:
                new_h = int(H * scale)
                new_w = int(W * scale)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                density = torch.nn.functional.interpolate(density.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                density = density * (scale * scale)

                # <--- MODIFIED: Scale keypoints for validation/test (Nx2 array)
                if kpoints.numel() > 0: # Only scale if there are points
                    kpoints = kpoints * scale

        # The target dictionary holds the density map and the point coordinates.
        target = {
            'density': density,
            'points': kpoints, # <--- Pass kpoints as Nx2 tensor
            'image_id': torch.tensor([image_id], dtype=torch.long),
            'name': os.path.basename(img_path),
            'original_size': torch.tensor([H, W], dtype=torch.long)
        }

        return img, target

# <--- MODIFIED & RENAMED: Load image, density map, and the kpoint_map (binary 2D map)
def load_image_density_kpoints_and_map(img_gt_path):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: {img_path}")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img.mode == 'L':
        img = img.convert('RGB')

    try:
        with h5py.File(gt_path, 'r') as hf:
            density = np.array(hf['density_map']) # Expects 'density_map' key
            # <--- NEW: Load 'kpoint' which is a 2D binary map
            kpoint_map_2d = np.array(hf['kpoint']) 

    except FileNotFoundError:
        raise FileNotFoundError(f"GT .h5 file not found: {gt_path}")
    except KeyError as e:
        raise KeyError(f"Key not found in {gt_path}: {e}. Ensure 'density_map' and 'kpoint' keys exist.")
    except Exception as e:
        raise Exception(f"Error loading .h5 density map or kpoint_map from {gt_path}: {e}")

    return img, density, kpoint_map_2d # Returns PIL Image, NumPy density, NumPy kpoint_map_2d

# <--- MODIFIED & RENAMED: Random crop for image, density, and keypoints (Nx2 array)
def random_crop_transform(img, density, kpoints, num_patch=4, crop_size=128):
    """
    Crops multiple patches from a single image, density map, and filters/adjusts Nx2 keypoints.
    Modified to return a single patch if num_patch=1.
    """
    _, H, W = img.shape # Image is (C, H, W)
    
    # Pre-allocate tensors for crops
    crops_img = torch.zeros((num_patch, img.shape[0], crop_size, crop_size), dtype=img.dtype, device=img.device)
    crops_den = torch.zeros((num_patch, density.shape[0], crop_size, crop_size), dtype=density.dtype, device=density.device)
    
    # Store kpoints for each patch in a list, as number of points can vary
    crops_kpoints_list = [None] * num_patch 

    for i in range(num_patch):
        start_h = random.randint(0, H - crop_size)
        start_w = random.randint(0, W - crop_size)
        end_h = start_h + crop_size
        end_w = start_w + crop_size

        crops_img[i] = img[:, start_h:end_h, start_w:end_w]
        crops_den[i] = density[:, start_h:end_h, start_w:end_w]
        
        # Filter kpoints that fall within the current crop and adjust their coordinates
        if kpoints.numel() > 0: # Ensure there are points to process
            # Assuming kpoints are (x, y)
            x_coords = kpoints[:, 0]
            y_coords = kpoints[:, 1]

            # Find points within the crop boundaries
            mask_x = (x_coords >= start_w) & (x_coords < end_w)
            mask_y = (y_coords >= start_h) & (y_coords < end_h)
            mask = mask_x & mask_y

            cropped_kpoints = kpoints[mask].clone() # .clone() to ensure a new tensor
            
            # Adjust coordinates relative to the new crop's top-left corner
            if cropped_kpoints.numel() > 0: # Check if there are points after cropping
                cropped_kpoints[:, 0] -= start_w
                cropped_kpoints[:, 1] -= start_h
        else:
            cropped_kpoints = torch.empty((0, 2), dtype=torch.float32, device=kpoints.device) # Empty tensor if no points

        crops_kpoints_list[i] = cropped_kpoints
    
    if num_patch == 1:
        # Return single image, single density map, and single kpoints tensor (no batch dim for kpoints)
        return crops_img.squeeze(0), crops_den.squeeze(0), crops_kpoints_list[0] 
    else:
        # If returning multiple patches, you will need a custom collate_fn for the DataLoader
        # to handle the list of kpoints tensors of varying lengths.
        # For simplicity, returning a list of kpoints tensors here.
        return crops_img, crops_den, crops_kpoints_list

# This 'build' function will be imported by __init__.py
def build(purpose, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name = args.dataset
    aug_dict = getattr(args, 'aug_dict', None) 

    dataset = ImageDataset(dataset=dataset_name,
                           transform=transform,
                           purpose=purpose,
                           aug_dict=aug_dict,
                           args=args)
    return dataset