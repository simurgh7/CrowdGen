import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser('Set parameters for training SASNet model', add_help=False)

# --- Dataset Configuration ---
parser.add_argument('--dataset', default='shha', type=str,
                    help='Name of the dataset to be used (e.g., "shha", "jhu", "nwpu", "ucf")')
parser.add_argument('--base_dir', default=r'D:\crowd\data\localization',
                    help='Root path to the dataset directory.')
parser.add_argument('--data_root', default=r'D:\crowd\data\localization\shanghaitechA',
                    help='Root path to the specific dataset directory.')
parser.add_argument('--npy_dir', default=r'D:\crowd\code\2021_AAAI_SASNet_DM\datasets\npydata',
                    help='Root path to the dataset directory.')

# --- Training Parameters ---
parser.add_argument('--batch_size', default=8, type=int,
                    help='Number of samples per batch for training.')
parser.add_argument('--epochs', default=12, type=int,
                    help='Total number of training epochs.')
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training logs.')
parser.add_argument('--eval_freq', default=5, type=int,
                    help='Frequency (in epochs) at which to perform evaluation during training.')
parser.add_argument('--log_interval', default=100, type=int,
                    help='Frequency (in epochs) at which to perform logging during training.')

# --- Model and Data Processing Parameters ---
parser.add_argument('--mean_std', default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), type=tuple,
                    help='Mean and standard deviation for image normalization')
parser.add_argument('--block_size', type=int, default=32, 
                    help='patch size for feature level selection')
parser.add_argument('--log_para', type=int, default=1000, 
                    help='magnify the target density map')
parser.add_argument('--label_factor', default=8, type=int,
                    help='Label factor for downsampling')
parser.add_argument('--train_min_size', default=(512, 512), type=tuple,
                    help='Minimum size for training images')
parser.add_argument('--train_max_size', default=(2048, 2048), type=tuple,
                    help='Maximum size for training images')
parser.add_argument('--augment', default=1, type=int,
                    help='Whether to use data augmentation (0 or 1)')

# --- Model and Experiment Configuration ---
parser.add_argument('--output_dir', default=r'D:\crowd\code\2021_AAAI_SASNet_DM\outputs',
                    help='Path where training logs and primary outputs (like TensorBoard events) will be saved.')
parser.add_argument('--weight_dir', default=r'D:\crowd\code\2021_AAAI_SASNet_DM\weights',
                    help='Path where model checkpoints (weights) will be saved during training.')
parser.add_argument('--weight', default=r'D:\crowd\code\sota-weights\shha_sasnet.pth', type=str,
                    help='Path to pretrained model for resuming training')
parser.add_argument('--checkpoint', default=None,
                    help='Path to a checkpoint file to resume training from (used in train.py).')
parser.add_argument('--visualization_dir', default=None, type=str,
                    help='Directory to save visualized predictions during evaluation. Set to None to skip.')
parser.add_argument('--explanation_dir', default=r"C:\Users\crowd\explanation\sasnet_unet", type=str,
                        help='Directory to save explainability maps.')
parser.add_argument('--advex_dir', default=r'C:\Users\crowd\ablation\sasnet_unet', type=str,
                    help='Directory to save adv examples.')
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed for reproducibility')

# --- Hardware Configuration ---
parser.add_argument('--gpu', default=0, type=int,
                    help='The ID of the GPU to use for training or testing (e.g., 0, 1, etc.).')
parser.add_argument('--device', default='cuda', type=str, # General device argument
                    help='Device to use for training / testing (e.g., "cuda" or "cpu").')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of data loading workers for PyTorch DataLoaders.')

# --- Adv Training Specs ---
parser.add_argument('--epsilon', default=10/255, type=float,
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
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')

# --- Adversarial Example Evaluation --- #
parser.add_argument('--clean_npy_path', default=r'D:\crowd\code\2021_AAAI_SASNet_DM\datasets\npydata\shha_val.npy', type=str,
                    help='Path to .npy file containing list of clean image paths.')
parser.add_argument('--adv_npy_path', default=r'D:\crowd\code\2021_AAAI_SASNet_DM\datasets\npydata\shha_sasnet_unet.npy', type=str,
                    help='Path to .npy file containing list of adversarial image paths.')
# Parse the arguments
args = parser.parse_args()

# # Set dataset-specific defaults based on the dataset argument
if args.dataset == 'jhu':
    args.data_root = r'D:\crowd\data\localization\jhu-crowd++'
    args.advex_dir = r'C:\Users\crowd\inference\jhu-cross\sasnet_unet'
    args.mean_std = ([0.447, 0.407, 0.385], [0.228, 0.221, 0.216])
    args.log_para = 1000.
    args.train_batch_size = 8

elif args.dataset == 'nwpu':
    args.data_root = r'D:\crowd\data\localization\nwpu-crowd'
    #args.advex_dir = r'C:\Users\crowd\inference\nwpu-cross\sasnet_unet'
    args.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    args.log_para = 100.
    args.train_batch_size = 4

elif args.dataset == 'shha':
    args.data_root = r'D:\crowd\data\localization\shanghaitechA'
    #args.advex_dir = r'C:\Users\crowd\inference\shha\sasnet_unet'
    args.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    args.log_para = 1000. #100
    args.train_batch_size = 8

elif args.dataset == 'ucf':
    args.data_root = r'D:\crowd\data\localization\ucf-qnrf'
    #args.advex_dir = r'C:\Users\crowd\inference\ucf\sasnet_unet'
    args.mean_std = ([0.430, 0.430, 0.430], [0.180, 0.180, 0.180])
    args.log_para = 1000.
    args.label_factor = 1
    args.train_batch_size = 4

else:
    raise ValueError("Unsupported dataset")

# Return the parsed arguments
return_args = args