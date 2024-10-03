import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
import torch
import geopandas as gpd

ortho_means = [31.63289965, 34.25222465, 31.6679167]  # Replace with actual calculated means for each band
ortho_stds = [45.82212139, 46.20130014, 42.59491153]  # Replace with actual calculated stds for each band
lidar_mean = 483.59665233887523  # Replace with actual calculated mean
lidar_std = 480.7912533243046   # Replace with actual calculated std


def random_flip_and_rotate(image, mask):
    # Flip horizontal
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    # Flip vertical
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    # Rotate 90 degrees
    if np.random.rand() > 0.5:
        image = np.rot90(image)
        mask = np.rot90(mask)
    return image, mask

class RAMDataset(Dataset):
    def __init__(self, ortho_images, lidar_images, masks, image_names, use_lidar=True, transform=None):
        self.ortho_images = ortho_images
        self.lidar_images = lidar_images
        self.masks = masks
        self.image_names = image_names
        self.use_lidar = use_lidar  # New parameter
        self.transform = transform

    def __len__(self):
        return len(self.ortho_images)

    def __getitem__(self, idx):
        ortho_image = self.ortho_images[idx]
        mask = self.masks[idx]
        image_name = self.image_names[idx]

        if self.use_lidar:
            lidar_image = self.lidar_images[idx]
            image = np.concatenate((ortho_image, lidar_image), axis=2)
        else:
            image = ortho_image  # Use only ortho images

        if self.transform:
            image, mask = self.transform(image, mask)

        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # Convert to CHW
        mask = np.ascontiguousarray(mask)  # Ensure positive strides
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        return image, mask, image_name


def preload_images(ortho_paths, lidar_paths, mask_paths, use_lidar=True):
    ortho_images = []
    lidar_images = []
    masks = []
    image_names = []

    for ortho_path, lidar_path, mask_path in zip(ortho_paths, lidar_paths, mask_paths):
        with rasterio.open(ortho_path) as src:
            ortho_image = src.read().astype(np.float32)
            ortho_image = np.transpose(ortho_image, (1, 2, 0))

            # Apply normalization to ortho images
            for i in range(3):  # 3 bands for ortho
                ortho_image[:, :, i] = (ortho_image[:, :, i] - ortho_means[i]) / ortho_stds[i]

        if use_lidar:
            with rasterio.open(lidar_path) as src:
                lidar_image = src.read(1).astype(np.float32) 
                lidar_image = (lidar_image - lidar_mean) / lidar_std  
                lidar_image = np.expand_dims(lidar_image, axis=2) 
            lidar_images.append(lidar_image)
        else:
            lidar_images.append(None) 

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.clip(mask, 0, 1) 

        ortho_images.append(ortho_image)
        masks.append(mask)
        image_names.append(os.path.basename(ortho_path)) 

    return ortho_images, lidar_images, masks, image_names


def get_data_loaders(ortho_dir, lidar_dir, mask_dir, gpkg_path, batch_size=16, num_workers=4, use_lidar=True):
    gdf = gpd.read_file(gpkg_path)

    train_departements = gdf[gdf['group'] == 'train']['code'].tolist()
    val_departements = gdf[gdf['group'] == 'val']['code'].tolist()
    test_departements = gdf[gdf['group'] == 'test']['code'].tolist()

    def filter_paths_by_department(paths, departments):
        return [path for path in paths if os.path.basename(path)[:2] in departments]

    ortho_paths = [os.path.join(ortho_dir, f) for f in os.listdir(ortho_dir) if f.endswith('.tif')]
    lidar_paths = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith('.tif')]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')]

    ortho_paths_train = filter_paths_by_department(ortho_paths, train_departements)
    ortho_paths_val = filter_paths_by_department(ortho_paths, val_departements)
    ortho_paths_test = filter_paths_by_department(ortho_paths, test_departements)
    print('Number of train: ' + str(len(ortho_paths_train)))
    print('Number of val: ' + str(len(ortho_paths_val)))
    print('Number of test: ' + str(len(ortho_paths_test)))

    lidar_paths_train = filter_paths_by_department(lidar_paths, train_departements)
    lidar_paths_val = filter_paths_by_department(lidar_paths, val_departements)
    lidar_paths_test = filter_paths_by_department(lidar_paths, test_departements)

    mask_paths_train = filter_paths_by_department(mask_paths, train_departements)
    mask_paths_val = filter_paths_by_department(mask_paths, val_departements)
    mask_paths_test = filter_paths_by_department(mask_paths, test_departements)

    ortho_images_train, lidar_images_train, masks_train, image_names_train = preload_images(
        ortho_paths_train, lidar_paths_train, mask_paths_train, use_lidar=use_lidar
    )
    ortho_images_val, lidar_images_val, masks_val, image_names_val = preload_images(
        ortho_paths_val, lidar_paths_val, mask_paths_val, use_lidar=use_lidar
    )
    ortho_images_test, lidar_images_test, masks_test, image_names_test = preload_images(
        ortho_paths_test, lidar_paths_test, mask_paths_test, use_lidar=use_lidar
    )

    train_transform = random_flip_and_rotate
    valid_transform = None

    train_dataset = RAMDataset(ortho_images_train, lidar_images_train, masks_train, image_names_train, use_lidar=use_lidar, transform=train_transform)
    valid_dataset = RAMDataset(ortho_images_val, lidar_images_val, masks_val, image_names_val, use_lidar=use_lidar, transform=valid_transform)
    test_dataset = RAMDataset(ortho_images_test, lidar_images_test, masks_test, image_names_test, use_lidar=use_lidar, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
