import argparse

# Initialize the argument parser directly at the module level.
# This parser defines all the command-line arguments and their default values
# for configuring the P2PNet model, training process, and dataset paths.
parser = argparse.ArgumentParser('Set parameters for training or testing P2PNet', add_help=False)
# --- Dataset Parameters ---
parser.add_argument('--dataset', default='shha', type=str,
                    help='Name of the dataset to be used (e.g., "shha", "ucf", "nwpu").')
parser.add_argument('--base_dir', default=r'D:\crowd\data\localization',
                    help='Root path to the dataset directory.')
parser.add_argument('--data_root', default=r'D:\crowd\data\localization\shanghaitechA',
                    help='Root path to the specific dataset directory.')
parser.add_argument('--npy_dir', default=r'D:\crowd\code\2021_ICCV_P2PNet_PR\datasets\npydata',
                    help='Root path to the dataset directory.')
# --- Training Parameters ---
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Initial learning rate for the optimizer.')
parser.add_argument('--lr_backbone', default=1e-5, type=float,
                    help='Learning rate specifically for the backbone network parameters.')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Number of samples per batch for training.')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay (L2 penalty) for optimizer to prevent overfitting.')
parser.add_argument('--epochs', default=12, type=int,
                    help='Total number of training epochs.')
parser.add_argument('--lr_drop', default=3500, type=int,
                    help='Epoch at which the learning rate will be dropped (e.g., by a factor of 10).')
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='Maximum norm for gradient clipping, used to prevent exploding gradients during training.')

# --- Model Parameters ---
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to a pretrained model checkpoint. If set during training, "
                         "only specific parts (e.g., mask head) might be trained with frozen backbone.")
parser.add_argument('--backbone', default='vgg16_bn', type=str,
                    help="Name of the convolutional backbone architecture to use (e.g., 'vgg16_bn', 'resnet50').")

# * Matcher Parameters (for assigning predictions to ground truth during training)
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Coefficient for the classification cost in the matching algorithm.")
parser.add_argument('--set_cost_point', default=0.05, type=float,
                    help="Coefficient for the L1 point regression cost in the matching algorithm.")

# * Loss Coefficients
parser.add_argument('--point_loss_coef', default=0.0002, type=float,
                    help='Coefficient for the point regression loss in the total loss function.')
parser.add_argument('--eos_coef', default=0.5, type=float,
                    help="Relative classification weight of the 'no-object' class (End-Of-Sequence coefficient). "
                         "A higher value penalizes false positives more strongly.")
parser.add_argument('--row', default=2, type=int,
                    help="Number of rows for the anchor points grid used in the model's point head.")
parser.add_argument('--line', default=2, type=int,
                    help="Number of columns (lines) for the anchor points grid used in the model's point head.")


# --- Output and Logging Directories ---
parser.add_argument('--output_dir', default=r'D:\crowd\code\2021_ICCV_P2PNet_PR\outputs',
                    help='Path where training logs and primary outputs (like TensorBoard events) will be saved.')
parser.add_argument('--weight_dir', default=r'D:\crowd\code\2021_ICCV_P2PNet_PR\weights',
                    help='Path where model checkpoints (weights) will be saved during training.')
parser.add_argument('--weight', default=r'D:\crowd\code\sota-weights\shha_p2pnet.pth',
                    help='Path to the model weights (.pth file) to be loaded for evaluation (used in test.py).')
parser.add_argument('--checkpoint', default=None,
                    help='Path to a checkpoint file to resume training from (used in train.py).')
parser.add_argument('--visualization_dir', default=r'D:\crowd\output\visualize\2021_ICCV_P2PNet', type=str,
                    help='Directory to save visualized predictions during evaluation. Set to None to skip.')
parser.add_argument('--explanation_dir', default=r'C:\Users\crowd\explanation\l2t', type=str,
                        help='Directory to save explainability maps.')
parser.add_argument('--advex_dir', default=r'C:\Users\crowd\ablation\p2pnet_cam', type=str,
                    help='Directory to save adv examples.')
# -- Adv parameters -- #
parser.add_argument('--epsilon', default=8/255, type=float,
                    help="controls the perturbation of the U-Net")
parser.add_argument('--alpha', default=0.01, type=float,
                    help="controls the logit loss for perturbation of the U-Net")                    
parser.add_argument('--beta', default=0.01, type=float,
                    help="controls the l2 norm error for perturbation of the U-Net")
parser.add_argument('--gamma', default=0.05, type=float,
                    help="controls the TV loss for perturbation of the U-Net")
parser.add_argument('--zeta', default=0.01, type=float,
                    help="controls the freq loss for perturbation of the U-Net")
parser.add_argument('--kappa', default=0.5, type=float,
                    help="controls the cam loss balancing term")
parser.add_argument('--cam_weight', default=0.5, type=float,
                    help="controls how strongly to weight CAM regions")
parser.add_argument('--diffusion_steps', default = 1000, type=int,
                    help="controls how many timesteps to take")
parser.add_argument('--beta_schedule', default = "linear", type=str,  # or "cosine"
                    help="controls diffusion")
parser.add_argument('--diffusion_weight', default = 0.5, type=float,  # Weight for diffusion MSE loss
                    help="controls how strongly to weight diffusion")

# --- Training Control and Miscellaneous Parameters ---
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed for reproducibility across multiple runs.')
parser.add_argument('--resume', default='',
                    help='Path to a checkpoint file to resume training from.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='Starting epoch number when resuming training.')
parser.add_argument('--eval', action='store_true',
                    help='If set, the script will run in evaluation-only mode (primarily for train.py).')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of data loading workers for PyTorch DataLoaders.')
parser.add_argument('--eval_freq', default=5, type=int,
                    help='Frequency (in epochs) at which to perform evaluation during training.')
parser.add_argument('--log_interval', default=100, type=int,
                    help='Frequency (in epochs) at which to perform logging during training.')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='The ID of the GPU to use for training or testing (e.g., 0, 1, etc.).')


# --- Distributed Training Parameters (NEW / RE-ADDED) ---
parser.add_argument('--distributed', action='store_true',
                    help='Enable distributed training/evaluation using PyTorch DDP.')
parser.add_argument('--world_size', default=1, type=int,
                    help='Number of distributed processes/GPUs.')
parser.add_argument('--dist_url', default='env://',
                    help='URL used to set up distributed training. "env://" uses environment variables.')
parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current process, should be automatically set in DDP setup.')
parser.add_argument('--gpu', default=None, type=int, # Internal GPU ID for DDP process
                    help='GPU ID of the current process, set by init_distributed_mode.')
parser.add_argument('--device', default='cuda', type=str, # General device argument
                    help='Device to use for training / testing (e.g., "cuda" or "cpu").')

# --- Adversarial Example Evaluation --- #
parser.add_argument('--clean_npy_path', default=r'D:\crowd\code\2021_ICCV_P2PNet_PR\datasets\npydata\shha_val.npy', type=str,
                    help='Path to .npy file containing list of clean image paths.')
parser.add_argument('--adv_npy_path', default=r'D:\crowd\code\2021_ICCV_P2PNet_PR\datasets\npydata\shha_bench_l2t.npy', type=str,
                    help='Path to .npy file containing list of adversarial image paths.')

# Parse the arguments. 'args' will contain all the parsed values.
args = parser.parse_args()

# 'return_args' is often used in some frameworks to explicitly return the parsed arguments
# from a function, but here, it's identical to 'args' and might be redundant
# unless there's a specific downstream use case.
return_args = parser.parse_args()