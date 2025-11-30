"""Dataset module for loading preprocessed ear images with albumentations."""

import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EarDataset(Dataset):
    """Dataset for loading preprocessed ear images from .npy files."""

    def __init__(self, npy_path: str, transform=None, is_training: bool = True):
        """
        Args:
            npy_path: Path to .npy file containing preprocessed images
            transform: Optional albumentations transform
            is_training: Whether this is training data (for augmentation)
        """
        self.data = np.load(npy_path)
        self.is_training = is_training

        # Normalize to [0, 1] if not already
        if self.data.max() > 1.0:
            self.data = self.data.astype(np.float32) / 255.0
        else:
            self.data = self.data.astype(np.float32)

        # Use provided transform or create default
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_default_transform(is_training)

    @staticmethod
    def get_default_transform(is_training: bool = True):
        """
        Get default albumentations transform.

        Args:
            is_training: Whether to apply training augmentations

        Returns:
            Albumentations compose transform
        """
        if is_training:
            transform = A.Compose([
                # Geometric transformations: rotation, scale, translation
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),

                # Rotation with probability
                A.Rotate(limit=30, border_mode=0, p=0.5),

                # Scale and translation
                A.ShiftScaleRotate(
                    shift_limit=0.15,      # Translation (15% of image size)
                    scale_limit=0.2,       # Scale (±20%)
                    rotate_limit=0,        # Rotation handled separately
                    border_mode=0,
                    p=0.6
                ),

                # Random crop and resize
                A.OneOf([
                    A.RandomResizedCrop(
                        height=256,
                        width=256,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        p=1.0
                    ),
                    A.CenterCrop(height=224, width=224, p=1.0),
                ], p=0.3),

                # Perspective and distortion
                A.Perspective(scale=(0.05, 0.15), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.2),

                # Photometric jitter (color, brightness, contrast, saturation, hue)
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.3,    # ±30% brightness
                        contrast=0.3,      # ±30% contrast
                        saturation=0.3,    # ±30% saturation
                        hue=0.15,          # ±15% hue shift
                        p=1.0
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.7),

                # Additional photometric augmentations
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                A.ChannelShuffle(p=0.1),

                # Random blur (Gaussian, motion, median, advanced)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.AdvancedBlur(blur_limit=(3, 7), p=1.0),
                    A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1.0),
                ], p=0.4),

                # Noise augmentations
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.3),

                # Synthetic occlusions (dropout, cutout, grid dropout)
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=1.0
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        unit_size_min=8,
                        unit_size_max=16,
                        holes_number_x=4,
                        holes_number_y=4,
                        p=1.0
                    ),
                    A.Cutout(
                        num_holes=8,
                        max_h_size=16,
                        max_w_size=16,
                        fill_value=0,
                        p=1.0
                    ),
                ], p=0.25),

                # Additional occlusions with different patterns
                A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.15),

                # Environmental effects
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.2
                ),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.1),
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3,
                    brightness_coefficient=0.7,
                    rain_type=None,
                    p=0.05
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    src_radius=100,
                    p=0.05
                ),

                # Quality degradation
                A.OneOf([
                    A.Downscale(scale_min=0.5, scale_max=0.75, p=1.0),
                    A.ImageCompression(quality_lower=60, quality_upper=90, p=1.0),
                ], p=0.15),

                # Ensure values are in [0, 1]
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
                ToTensorV2(),
            ])

        return transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Ensure correct shape for albumentations (H, W, C)
        if image.ndim == 2:
            # Grayscale image - add channel dimension
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 3:
            # Check if channels are first (C, H, W) and permute if needed
            if image.shape[0] in [1, 3] and image.shape[0] < image.shape[1]:
                image = np.transpose(image, (1, 2, 0))

        # Apply transform
        transformed = self.transform(image=image)
        image_tensor = transformed['image']

        return image_tensor


def get_train_transform(image_size: int = 256):
    """
    Get comprehensive training augmentation transform with rotation, crop, scale,
    translation, photometric jitter, blur, and synthetic occlusions.

    Args:
        image_size: Target image size

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Resize if needed
        A.Resize(image_size, image_size),

        # === GEOMETRIC TRANSFORMATIONS ===

        # Flip augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),

        # Rotation (random rotation up to ±30 degrees)
        A.Rotate(limit=30, border_mode=0, p=0.5),

        # Scale and Translation (shift_limit controls translation, scale_limit controls zoom)
        A.ShiftScaleRotate(
            shift_limit=0.15,      # Translation: ±15% of image size
            scale_limit=0.2,       # Scale: ±20%
            rotate_limit=0,        # Rotation handled separately
            border_mode=0,
            p=0.6
        ),

        # Random crop and resize (simulates different distances/viewpoints)
        A.OneOf([
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),      # Crop to 80-100% of original
                ratio=(0.9, 1.1),      # Aspect ratio variation
                p=1.0
            ),
            A.CenterCrop(height=int(image_size * 0.875), width=int(image_size * 0.875), p=1.0),
        ], p=0.3),

        # Advanced geometric transformations
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.2),

        # === PHOTOMETRIC JITTER ===

        # Color jitter (brightness, contrast, saturation, hue)
        A.OneOf([
            A.ColorJitter(
                brightness=0.3,        # ±30% brightness
                contrast=0.3,          # ±30% contrast
                saturation=0.3,        # ±30% saturation
                hue=0.15,              # ±15% hue shift
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.7),

        # Additional photometric augmentations
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.ChannelShuffle(p=0.1),

        # === RANDOM BLUR ===

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.AdvancedBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.4),

        # === NOISE ===

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        # === SYNTHETIC OCCLUSIONS ===

        A.OneOf([
            # Coarse dropout (rectangular holes)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=1.0
            ),
            # Grid dropout (regular grid pattern)
            A.GridDropout(
                ratio=0.3,
                unit_size_min=8,
                unit_size_max=16,
                holes_number_x=4,
                holes_number_y=4,
                p=1.0
            ),
            # Cutout (random rectangular regions)
            A.Cutout(
                num_holes=8,
                max_h_size=16,
                max_w_size=16,
                fill_value=0,
                p=1.0
            ),
        ], p=0.25),

        # Mask dropout (random irregular shapes)
        A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.15),

        # === ENVIRONMENTAL EFFECTS ===

        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.1),
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.7,
            rain_type=None,
            p=0.05
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=2,
            src_radius=100,
            p=0.05
        ),

        # === QUALITY DEGRADATION ===

        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.75, p=1.0),
            A.ImageCompression(quality_lower=60, quality_upper=90, p=1.0),
        ], p=0.15),

        # Normalize and convert to tensor
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 256):
    """
    Get validation transform (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ToTensorV2(),
    ])
