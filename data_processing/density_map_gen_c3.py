import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import h5py

from scipy.signal import convolve2d

def generate_density_map(image, points, k_size=15, sigma=4.0):
    """
    Generates a density map from point annotations using a Gaussian kernel,
    closely replicating the MATLAB implementation.

    Args:
        image (numpy.ndarray): The input image (used for shape).
        points (numpy.ndarray): An N x 2 array of head center coordinates (x, y).
        k_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: The generated density map (single channel).
    """
    h, w, _ = image.shape[:3]
    density_map = np.zeros((h, w), dtype=np.float32)

    if points is None or len(points) == 0:
        return density_map

    for j in range(points.shape[0]):
        f_sz = k_size
        H_full = _generate_gaussian_kernel_matlab(f_sz, sigma)

        x_center = int(np.floor(points[j, 0]))
        y_center = int(np.floor(points[j, 1]))

        x1 = x_center - f_sz // 2
        y1 = y_center - f_sz // 2
        x2 = x_center + f_sz // 2
        y2 = y_center + f_sz // 2

        h_kernel, w_kernel = H_full.shape

        x_start = max(0, x1)
        y_start = max(0, y1)
        x_end = min(w, x2 + 1)
        y_end = min(h, y2 + 1)

        h_overlap = y_end - y_start
        w_overlap = x_end - x_start

        if h_overlap > 0 and w_overlap > 0:
            h_start_kernel = h_overlap - (y_end - (y_center + f_sz // 2)) if y2 >= h else 0
            w_start_kernel = w_overlap - (x_end - (x_center + f_sz // 2)) if x2 >= w else 0
            h_end_kernel = h_start_kernel + h_overlap
            w_end_kernel = w_start_kernel + w_overlap

            h_start_kernel = max(0, -y1)
            w_start_kernel = max(0, -x1)
            h_end_kernel = min(h_kernel, h_start_kernel + h_overlap)
            w_end_kernel = min(w_kernel, w_start_kernel + w_overlap)

            kernel_to_add = H_full[h_start_kernel:h_end_kernel, w_start_kernel:w_end_kernel]
            density_map[y_start:y_end, x_start:x_end] += kernel_to_add

    return density_map

def _generate_gaussian_kernel_matlab(size_y, sigma, size_x=None):
    """
    Generates a 2D Gaussian kernel similar to MATLAB's fspecial('Gaussian').

    Args:
        size_y (int): Height of the kernel.
        sigma (float): Standard deviation of the Gaussian.
        size_x (int, optional): Width of the kernel. If None, size_x = size_y.

    Returns:
        numpy.ndarray: The 2D Gaussian kernel.
    """
    if size_x is None:
        size_x = size_y
    yy, xx = np.indices((size_y, size_x), dtype=np.float32)
    center_y = size_y // 2
    center_x = size_x // 2
    exponent = -((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2)
    gaussian = np.exp(exponent)
    return gaussian / np.sum(gaussian)

def process_ucfqnrf_h5(data_path, output_path, max_size=(2048, 2048)):
    """
    Processes the UCF-QNRF dataset to resize images and generate density maps,
    saving the density maps in HDF5 format.

    Args:
        data_path (str): Path to the UCF-QNRF dataset (containing 'Train' and 'Test' folders).
        output_path (str): Path to save the processed images and density maps.
        max_size (tuple): Maximum height and width for resizing (e.g., (1024, 1024)).
        density_sigma (int): Sigma for the Gaussian kernel in density map generation.
        density_kernel_size (int, optional): Kernel size for density map generation.
    """
    for subset in ['train_data','val_data', 'test_data']:
        img_path = os.path.join(data_path, subset, 'images')
        gt_path = os.path.join(data_path, subset, 'point-mats')
        output_img_path = os.path.join(output_path, subset, 'images_2048')
        output_den_path = os.path.join(output_path, subset, 'gt-h5_2048')
        print(output_img_path, output_den_path)
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_den_path, exist_ok=True)

        image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])

        for i, img_file in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f'{subset} Set: Processing {i + 1}/{len(image_files)} files')

            img_name = os.path.splitext(img_file)[0]
            mat_file = os.path.join(gt_path, f'{img_name}.mat')
            img_full_path = os.path.join(img_path, img_file)

            try:
                ann_data = loadmat(mat_file)
                points = ann_data['annPoints']
                if points is not None:
                    points = points.astype(np.float32)
                else:
                    points = np.empty((0, 2), dtype=np.float32) # Handle cases with no annotations

                im = cv2.imread(img_full_path)
                if im is None:
                    print(f"Error reading image: {img_full_path}")
                    continue

                h, w = im.shape[:2]
                #original_points = points.copy()
                if h > max_size[0] or w > max_size[1]:
                    # Resize the image
                    rate = max_size[0] / h
                    rate_w_temp = w * rate
                    if rate_w_temp > max_size[1]:
                        rate = max_size[1] / w
                    rate_h = int(round(h * rate / 16)) * 16 / h
                    rate_w = int(round(w * rate / 16)) * 16 / w
                    resized_im = cv2.resize(im, (int(w * rate_w), int(h * rate_h)))

                    # Adjust point annotations
                    resized_points = points * np.array([rate_w, rate_h])
                else:
                    resized_im = im.copy() # Use a copy to avoid modifying the original
                    resized_points = points.copy() # No change to points
                # Generate density map
                density_map = generate_density_map(resized_im, resized_points, k_size=15, sigma=4.0)

                # Generate kpoint map (binary 2D map with 1s at annotation points)
                kpoint_map = np.zeros((resized_im.shape[0], resized_im.shape[1]), dtype=np.uint8)
                for p in resized_points:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < resized_im.shape[1] and 0 <= y < resized_im.shape[0]:
                        kpoint_map[y, x] = 1

                # Save the processed image
                cv2.imwrite(os.path.join(output_img_path, img_name+'.jpg'), resized_im)

                # Save the density map in HDF5 format
                h5_filename = os.path.join(output_den_path, img_name+'.h5')
                with h5py.File(h5_filename, 'w') as hf:
                    hf['density_map'] = density_map
                    hf['kpoint'] = kpoint_map

            except FileNotFoundError:
                print(f"Annotation file not found: {mat_file}")
            except Exception as e:
                print(f"An error occurred while processing {img_file}: {e}")

def process_ucf():
    # --- Usage ---
    data_root = r'D:\crowd\data\localization\ucf-qnrf'  # Replace with the actual path to your UCF-QNRF dataset
    output_root = r'D:\crowd\data\localization\ucf-qnrf' # Replace with your desired output path
    target_size = (2048, 2048) # Resize to 2048x2048

    process_ucfqnrf_h5(data_root, output_root, max_size=target_size)

def process_shha_h5(data_path, output_path, max_size=(2048, 2048)):
    """
    Processes the SHHA dataset to resize images and generate density maps,
    saving the density maps in HDF5 format.

    Args:
        data_path (str): Path to the SHHA dataset (containing 'Train' and 'Test' folders).
        output_path (str): Path to save the processed images and density maps.
        max_size (tuple): Maximum height and width for resizing (e.g., (1024, 1024)).
        density_sigma (int): Sigma for the Gaussian kernel in density map generation.
        density_kernel_size (int, optional): Kernel size for density map generation.
    """
    for subset in ['train_data','val_data', 'test_data']:
        img_path = os.path.join(data_path, subset, 'images')
        gt_path = os.path.join(data_path, subset, 'point-mats')
        output_img_path = os.path.join(output_path, subset, 'images_2048')
        output_den_path = os.path.join(output_path, subset, 'gt-h5_2048')
        print(output_img_path, output_den_path)
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_den_path, exist_ok=True)

        image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])

        for i, img_file in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f'{subset} Set: Processing {i + 1}/{len(image_files)} files')

            img_name = os.path.splitext(img_file)[0]
            mat_file = os.path.join(gt_path, f'{img_name}.mat')
            img_full_path = os.path.join(img_path, img_file)

            try:
                ann_data = loadmat(mat_file)
                points = ann_data['image_info'][0][0][0][0][0]
                if points is not None:
                    points = points.astype(np.float32)
                else:
                    points = np.empty((0, 2), dtype=np.float32) # Handle cases with no annotations

                im = cv2.imread(img_full_path)
                if im is None:
                    print(f"Error reading image: {img_full_path}")
                    continue

                h, w = im.shape[:2]
                #original_points = points.copy()
                if h > max_size[0] or w > max_size[1]:
                    # Resize the image
                    rate = max_size[0] / h
                    rate_w_temp = w * rate
                    if rate_w_temp > max_size[1]:
                        rate = max_size[1] / w
                    rate_h = int(round(h * rate / 16)) * 16 / h
                    rate_w = int(round(w * rate / 16)) * 16 / w
                    resized_im = cv2.resize(im, (int(w * rate_w), int(h * rate_h)))

                    # Adjust point annotations
                    resized_points = points * np.array([rate_w, rate_h])
                else:
                    resized_im = im.copy() # Use a copy to avoid modifying the original
                    resized_points = points.copy() # No change to points

                # Generate density map
                density_map = generate_density_map(resized_im, resized_points, k_size=15, sigma=4.0)

                # Generate kpoint map (binary 2D map with 1s at annotation points)
                kpoint_map = np.zeros((resized_im.shape[0], resized_im.shape[1]), dtype=np.uint8)
                for p in resized_points:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < resized_im.shape[1] and 0 <= y < resized_im.shape[0]:
                        kpoint_map[y, x] = 1

                # Save the processed image
                cv2.imwrite(os.path.join(output_img_path, img_name+'.jpg'), resized_im)

                # Save the density map in HDF5 format
                h5_filename = os.path.join(output_den_path, img_name+'.h5')
                with h5py.File(h5_filename, 'w') as hf:
                    hf['density_map'] = density_map
                    hf['kpoint'] = kpoint_map

            except FileNotFoundError:
                print(f"Annotation file not found: {mat_file}")
            except Exception as e:
                print(f"An error occurred while processing {img_file}: {e}")

def process_shha():
    # --- Usage ---
    data_root = r'D:\crowd\data\localization\shanghaitechA'  # Replace with the actual path to your UCF-QNRF dataset
    output_root = r'D:\crowd\data\localization\shanghaitechA' # Replace with your desired output path
    target_size = (2048, 2048) # Resize to 2048x2048
    #print("here")
    process_shha_h5(data_root, output_root, max_size=target_size)

def process_jhu_h5(data_path, output_path, max_size=(2048, 2048)):
    for subset in ['train_data','val_data', 'test_data']:
        img_path = os.path.join(data_path, subset, 'images')
        gt_path = os.path.join(data_path, subset, 'point-txts')
        output_img_path = os.path.join(output_path, subset, 'images_2048')
        output_den_path = os.path.join(output_path, subset, 'gt-h5_2048')
        print(output_img_path, output_den_path)
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_den_path, exist_ok=True)

        image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])

        for i, img_file in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f'{subset} Set: Processing {i + 1}/{len(image_files)} files')

            img_name = os.path.splitext(img_file)[0]
            txt_file = os.path.join(gt_path, f'{img_name}.txt')
            img_full_path = os.path.join(img_path, img_file)

            try:
                ## ONLY FOR JHU BEGINS ##
                mat_data = open(txt_file)
                mat = [list(map(int, line.rstrip().split())) for line in mat_data]
                points = np.array([[mat[i][0],mat[i][1]] for i in range(len(mat))])
                ## ONLY FOR JHU ENDS ##
                if points is not None:
                    points = points.astype(np.float32)
                else:
                    points = np.empty((0, 2), dtype=np.float32) # Handle cases with no annotations

                im = cv2.imread(img_full_path)
                if im is None:
                    print(f"Error reading image: {img_full_path}")
                    continue

                h, w = im.shape[:2]
                #original_points = points.copy()
                if h > max_size[0] or w > max_size[1]:
                    # Resize the image
                    rate = max_size[0] / h
                    rate_w_temp = w * rate
                    if rate_w_temp > max_size[1]:
                        rate = max_size[1] / w
                    rate_h = int(round(h * rate / 16)) * 16 / h
                    rate_w = int(round(w * rate / 16)) * 16 / w
                    resized_im = cv2.resize(im, (int(w * rate_w), int(h * rate_h)))

                    # Adjust point annotations
                    resized_points = points * np.array([rate_w, rate_h])
                else:
                    resized_im = im.copy() # Use a copy to avoid modifying the original
                    resized_points = points.copy() # No change to points

                # Generate density map
                density_map = generate_density_map(resized_im, resized_points, k_size=15, sigma=4.0)

                # Generate kpoint map (binary 2D map with 1s at annotation points)
                kpoint_map = np.zeros((resized_im.shape[0], resized_im.shape[1]), dtype=np.uint8)
                for p in resized_points:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < resized_im.shape[1] and 0 <= y < resized_im.shape[0]:
                        kpoint_map[y, x] = 1

                # Save the processed image
                cv2.imwrite(os.path.join(output_img_path, img_name+'.jpg'), resized_im)

                # Save the density map in HDF5 format
                h5_filename = os.path.join(output_den_path, img_name+'.h5')
                with h5py.File(h5_filename, 'w') as hf:
                    hf['density_map'] = density_map
                    hf['kpoint'] = kpoint_map

            except FileNotFoundError:
                print(f"Annotation file not found: {txt_file}")
            except Exception as e:
                print(f"An error occurred while processing {img_file}: {e}")

def process_jhu():
    # --- Usage ---
    data_root = r'D:\crowd\data\localization\jhu-crowd++'  # Replace with the actual path to your UCF-QNRF dataset
    output_root = r'D:\crowd\data\localization\jhu-crowd++' # Replace with your desired output path
    target_size = (2048, 2048) # Resize to 2048x2048
    #print("here")
    process_jhu_h5(data_root, output_root, max_size=target_size)

def process_nwpu_h5(data_path, output_path, max_size=(2048, 2048)):
    """
    Processes the NWPU dataset to resize images and generate density maps,
    saving the density maps in HDF5 format.

    Args:
        data_path (str): Path to the SHHA dataset (containing 'Train' and 'Test' folders).
        output_path (str): Path to save the processed images and density maps.
        max_size (tuple): Maximum height and width for resizing (e.g., (1024, 1024)).
        density_sigma (int): Sigma for the Gaussian kernel in density map generation.
        density_kernel_size (int, optional): Kernel size for density map generation.
    """
    for subset in ['train_data','val_data']:
        img_path = os.path.join(data_path, subset, 'images')
        gt_path = os.path.join(data_path, subset, 'point-mats')
        output_img_path = os.path.join(output_path, subset, 'images_2048')
        output_den_path = os.path.join(output_path, subset, 'gt-h5_2048')
        print(output_img_path, output_den_path)
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_den_path, exist_ok=True)

        image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])

        for i, img_file in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f'{subset} Set: Processing {i + 1}/{len(image_files)} files')

            img_name = os.path.splitext(img_file)[0]
            mat_file = os.path.join(gt_path, f'{img_name}.mat')
            img_full_path = os.path.join(img_path, img_file)

            try:
                ann_data = loadmat(mat_file)
                points = ann_data['annPoints']
                
                if points is None or points.size == 0:
                    #print(f"{img_file} annPoints shape: {ann_data['annPoints'].shape}")
                    points = np.empty((0, 2), dtype=np.float32)
                    #print(f"{img_file} processed points shape: {points.shape}")
                else:
                    points = points.astype(np.float32)
                    if points.ndim != 2 or points.shape[1] != 2:
                        print(f"Invalid point shape in {mat_file}: {points.shape}")
                        points = np.empty((0, 2), dtype=np.float32)
                im = cv2.imread(img_full_path)
                if im is None:
                    print(f"Error reading image: {img_full_path}")
                    continue

                h, w = im.shape[:2]
                #original_points = points.copy()
                if h > max_size[0] or w > max_size[1]:
                    # Resize the image
                    rate = max_size[0] / h
                    rate_w_temp = w * rate
                    if rate_w_temp > max_size[1]:
                        rate = max_size[1] / w
                    rate_h = int(round(h * rate / 16)) * 16 / h
                    rate_w = int(round(w * rate / 16)) * 16 / w
                    resized_im = cv2.resize(im, (int(w * rate_w), int(h * rate_h)))

                    # Adjust point annotations
                    resized_points = points * np.array([rate_w, rate_h])
                else:
                    resized_im = im.copy() # Use a copy to avoid modifying the original
                    resized_points = points.copy() # No change to points

                # Generate density map
                density_map = generate_density_map(resized_im, resized_points, k_size=15, sigma=4.0)

                # Generate kpoint map (binary 2D map with 1s at annotation points)
                kpoint_map = np.zeros((resized_im.shape[0], resized_im.shape[1]), dtype=np.uint8)
                for p in resized_points:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < resized_im.shape[1] and 0 <= y < resized_im.shape[0]:
                        kpoint_map[y, x] = 1

                # Save the processed image
                cv2.imwrite(os.path.join(output_img_path, img_name+'.jpg'), resized_im)

                # Save the density map in HDF5 format
                h5_filename = os.path.join(output_den_path, img_name+'.h5')
                with h5py.File(h5_filename, 'w') as hf:
                    hf['density_map'] = density_map
                    hf['kpoint'] = kpoint_map

            except FileNotFoundError:
                print(f"Annotation file not found: {mat_file}")
            except Exception as e:
                print(f"An error occurred while processing {img_file}: {e}")

def process_nwpu():
    # --- Usage ---
    data_root = r'D:\crowd\data\localization\nwpu-crowd'  # Replace with the actual path to your UCF-QNRF dataset
    output_root = r'D:\crowd\data\localization\nwpu-crowd' # Replace with your desired output path
    target_size = (2048, 2048) # Resize to 2048x2048

    process_nwpu_h5(data_root, output_root, max_size=target_size)

if __name__ == '__main__':
    #print("here")
    # process_shha()
    # process_ucf()
    # process_jhu()
    process_nwpu()
    
