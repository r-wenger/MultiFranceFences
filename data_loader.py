import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
import torch
import geopandas as gpd

ortho_means = [31.63289965, 34.25222465, 31.6679167]
ortho_stds = [45.82212139, 46.20130014, 42.59491153]
lidar_mean = 483.59665233887523
lidar_std = 480.7912533243046

class RAMDataset(Dataset):
    def __init__(self, ortho_images, lidar_images, masks, image_names, transform=None, output_mode="both"):
        """
        Dataset pour charger les images, les DSM et les masques.

        Args:
            ortho_images (list): Liste des images orthophotos.
            lidar_images (list): Liste des images DSM.
            masks (list): Liste des masques.
            image_names (list): Liste des noms des images.
            transform (callable, optional): Transformation à appliquer.
            output_mode (str, optional): Mode de sortie. 
                Options: "both" (par défaut), "image", "dsm", "concat".
        """
        self.ortho_images = ortho_images
        self.lidar_images = lidar_images
        self.masks = masks
        self.image_names = image_names
        self.transform = transform
        self.output_mode = output_mode

    def __len__(self):
        return len(self.ortho_images)

    def __getitem__(self, idx):
        ortho_image = self.ortho_images[idx]
        lidar_image = self.lidar_images[idx]
        mask = self.masks[idx]
        image_name = self.image_names[idx]

        if self.transform:
            ortho_image, lidar_image, mask = self.transform(ortho_image, lidar_image, mask)

        # Préparer les sorties selon le mode
        if self.output_mode == "image":
            output = np.ascontiguousarray(np.transpose(ortho_image, (2, 0, 1)))
        elif self.output_mode == "dsm":
            output = np.ascontiguousarray(np.transpose(lidar_image, (2, 0, 1)))
        elif self.output_mode == "concat":
            # Concaténation des bandes de l'image et du DSM (4 canaux)
            concat_image = np.concatenate((ortho_image, lidar_image), axis=-1)
            output = np.ascontiguousarray(np.transpose(concat_image, (2, 0, 1)))
        elif self.output_mode == "both":
            # Deux sorties séparées (image et DSM)
            ortho_image = np.ascontiguousarray(np.transpose(ortho_image, (2, 0, 1)))
            lidar_image = np.ascontiguousarray(np.transpose(lidar_image, (2, 0, 1)))
            output = (ortho_image, lidar_image)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        mask = np.ascontiguousarray(mask)
        mask = np.expand_dims(mask, axis=0)

        return output, mask, image_name


def random_flip_and_rotate(ortho_image, lidar_image, mask):
    """
    Applique des augmentations aléatoires à une image avec une probabilité globale de 50%.
    Si l'image est augmentée, applique un ou plusieurs des transformations suivantes :
    - Flip horizontal (gauche-droite)
    - Flip vertical (haut-bas)
    - Rotation aléatoire de 90°, 180°, ou 270°

    Args:
        ortho_image (ndarray): Image orthophoto (H, W, C).
        lidar_image (ndarray): Image lidar (H, W, C).
        mask (ndarray): Masque (H, W).

    Returns:
        tuple: Images augmentées ou non (ortho_image, lidar_image, mask).
    """
    # Probabilité globale de 50% pour l'augmentation
    if np.random.rand() > 0.5:
        # Flip gauche-droite
        if np.random.rand() > 0.5:
            ortho_image = np.fliplr(ortho_image)
            lidar_image = np.fliplr(lidar_image)
            mask = np.fliplr(mask)

        # Flip haut-bas
        if np.random.rand() > 0.5:
            ortho_image = np.flipud(ortho_image)
            lidar_image = np.flipud(lidar_image)
            mask = np.flipud(mask)

        # Rotation 90, 180 ou 270 degrés
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])  # 1=90°, 2=180°, 3=270°
            ortho_image = np.rot90(ortho_image, k=k)
            lidar_image = np.rot90(lidar_image, k=k)
            mask = np.rot90(mask, k=k)

    return ortho_image, lidar_image, mask


def preload_images(ortho_paths, lidar_paths, mask_paths):
    ortho_images, lidar_images, masks, image_names = [], [], [], []
    for ortho_path, lidar_path, mask_path in zip(ortho_paths, lidar_paths, mask_paths):
        with rasterio.open(ortho_path) as src:
            ortho_image = src.read().astype(np.float32)
            ortho_image = np.transpose(ortho_image, (1, 2, 0))
            for i in range(3):
                ortho_image[:, :, i] = (ortho_image[:, :, i] - ortho_means[i]) / ortho_stds[i]

        with rasterio.open(lidar_path) as src:
            lidar_image = src.read(1).astype(np.float32)
            lidar_image = (lidar_image - lidar_mean) / lidar_std
            lidar_image = np.expand_dims(lidar_image, axis=2)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.clip(mask, 0, 1)

        ortho_images.append(ortho_image)
        lidar_images.append(lidar_image)
        masks.append(mask)
        image_names.append(os.path.basename(ortho_path))

    return ortho_images, lidar_images, masks, image_names


def get_data_loaders(ortho_dir, lidar_dir, mask_dir, gpkg_path, batch_size=16, num_workers=4, stratification_method='geographic', random_seed=42, output_mode="both"):
    """
    Charge les DataLoaders avec un mode de sortie configurable.

    Args:
        ortho_dir (str): Répertoire des images orthophotos.
        lidar_dir (str): Répertoire des images LiDAR.
        mask_dir (str): Répertoire des masques.
        gpkg_path (str): Chemin vers le fichier GeoPackage pour la stratification géographique.
        batch_size (int): Taille des lots.
        num_workers (int): Nombre de processus pour charger les données.
        stratification_method (str): Méthode de stratification ('geographic' ou 'random').
        random_seed (int): Graine pour l'échantillonnage aléatoire.
        output_mode (str): Mode de sortie. 
            Options: "both", "image", "dsm", "concat".

    Returns:
        tuple: DataLoaders pour l'entraînement, la validation et les tests.
    """
    ortho_paths = [os.path.join(ortho_dir, f) for f in os.listdir(ortho_dir) if f.endswith('.tif')]
    lidar_paths = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith('.tif')]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')]

    # Stratification comme dans votre code actuel
    if stratification_method == 'geographic':
        # Stratification géographique
        gdf = gpd.read_file(gpkg_path)
        train_departments = gdf[gdf['group'] == 'train']['code'].tolist()
        val_departments = gdf[gdf['group'] == 'val']['code'].tolist()
        test_departments = gdf[gdf['group'] == 'test']['code'].tolist()

        def filter_paths_by_department(paths, departments):
            return [path for path in paths if os.path.basename(path)[:2] in departments]

        ortho_paths_train = filter_paths_by_department(ortho_paths, train_departments)
        ortho_paths_val = filter_paths_by_department(ortho_paths, val_departments)
        ortho_paths_test = filter_paths_by_department(ortho_paths, test_departments)

        lidar_paths_train = filter_paths_by_department(lidar_paths, train_departments)
        lidar_paths_val = filter_paths_by_department(lidar_paths, val_departments)
        lidar_paths_test = filter_paths_by_department(lidar_paths, test_departments)

        mask_paths_train = filter_paths_by_department(mask_paths, train_departments)
        mask_paths_val = filter_paths_by_department(mask_paths, val_departments)
        mask_paths_test = filter_paths_by_department(mask_paths, test_departments)

    elif stratification_method == 'random':
        # Échantillonnage aléatoire
        np.random.seed(random_seed)

        indices = np.arange(len(ortho_paths))
        train_idx, test_val_idx = train_test_split(indices, test_size=0.4, random_state=random_seed)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=random_seed)

        def filter_paths_by_indices(paths, indices):
            return [paths[i] for i in indices]

        ortho_paths_train = filter_paths_by_indices(ortho_paths, train_idx)
        ortho_paths_val = filter_paths_by_indices(ortho_paths, val_idx)
        ortho_paths_test = filter_paths_by_indices(ortho_paths, test_idx)

        lidar_paths_train = filter_paths_by_indices(lidar_paths, train_idx)
        lidar_paths_val = filter_paths_by_indices(lidar_paths, val_idx)
        lidar_paths_test = filter_paths_by_indices(lidar_paths, test_idx)

        mask_paths_train = filter_paths_by_indices(mask_paths, train_idx)
        mask_paths_val = filter_paths_by_indices(mask_paths, val_idx)
        mask_paths_test = filter_paths_by_indices(mask_paths, test_idx)

    else:
        raise ValueError(f"Unknown stratification method: {stratification_method}")

    # Chargement des données
    ortho_images_train, lidar_images_train, masks_train, image_names_train = preload_images(ortho_paths_train, lidar_paths_train, mask_paths_train)
    ortho_images_val, lidar_images_val, masks_val, image_names_val = preload_images(ortho_paths_val, lidar_paths_val, mask_paths_val)
    ortho_images_test, lidar_images_test, masks_test, image_names_test = preload_images(ortho_paths_test, lidar_paths_test, mask_paths_test)

    # Création des datasets
    train_dataset = RAMDataset(ortho_images_train, lidar_images_train, masks_train, image_names_train, transform=random_flip_and_rotate, output_mode=output_mode)
    valid_dataset = RAMDataset(ortho_images_val, lidar_images_val, masks_val, image_names_val, output_mode=output_mode)
    test_dataset = RAMDataset(ortho_images_test, lidar_images_test, masks_test, image_names_test, output_mode=output_mode)

    # Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
