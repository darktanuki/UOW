#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 # %%
import os 
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, v2

import numpy as np 
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from src.utils.engraving_utils import create_I_from_noise_and_DataMatrix

def unnormalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    # Compute unnormalize parameters.
    return v2.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                        std=[1/s for s in std])

def normalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    return v2.Normalize(mean=mean, std=std)

def embed_in_black_background(x: torch.Tensor, target_size: int) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape

    min_size = max(H, W)
    base = 56
    computed_size = ((min_size + base - 1) // base) * base
    final_size = max(target_size, computed_size)

    device = x.device
    dtype = x.dtype
    background = torch.zeros((B, C, final_size, final_size), dtype=dtype, device=device)

    top = (final_size - H) // 2
    left = (final_size - W) // 2
    background[:, :, top:top+H, left:left+W] = x

    return background

@dataclass
class Parameters:
    size: tuple[int, int] = (8, 8)
    img_size: tuple[int, int] = (64, 64)

class RandomAffineWithPhotometricNoise(nn.Module):
    """
    Apply RandomAffine followed by photometric and noise augmentations:
    - Brightness
    - Contrast
    - Uniform noise
    - Random erasing

    Expected input: Tensor of shape (B, C, H, W) or (C, H, W)
    All operations are done on CPU.
    """

    def __init__(
        self,
        img_size=(64, 64),
        degrees=(-30, 30),
        translate=(0.1, 0.1),
        scale=(1.0, 1.0),
        brightness_max=0.3,
        contrast_max=0.3,
        uniform_noise_scale=0.05,
        erasing_scale=(0.02, 0.1),
        erasing_ratio=(0.5, 2.0),
        blur_kernel_size = (7,7),
        blur_sigma=(0.01, 0.5),
        p_affine=1.0,
        p_photometric=1.0,
        normalize_mean=0.5,
        normalize_std=0.5,
        use_erasing=True,
    ):
        super().__init__()
        self.last_params = {}
        self.use_erasing = use_erasing
        self.p_affine = p_affine
        # ---- Geometric augmentation ----
        self.affine = v2.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            interpolation=InterpolationMode.BICUBIC,
            fill=0,
        )
        self.to_pil = v2.ToPILImage()
        self.pil_to_tensor = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        # ---- Photometric augmentations ----
        self.brightness = v2.RandomApply(
            [v2.ColorJitter(brightness=(1 - brightness_max, 1 + brightness_max))],
            p=p_photometric,
        )

        self.contrast = v2.RandomApply(
            [v2.ColorJitter(contrast=(1 - contrast_max, 1 + contrast_max))],
            p=p_photometric,
        )
        # ---- Blur ----
        self.blur = v2.RandomApply(
            [v2.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)],
            p=p_photometric,
        )

        # ---- Noise ----
        self.uniform_noise_scale = uniform_noise_scale

        # ---- Erasing ----
        self.erasing = v2.RandomErasing(
            scale=erasing_scale,
            ratio=erasing_ratio,
            value=0.0,
            p=1.0,
        )

        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.normalize = normalize_function(
            mean=self.normalize_mean,
            std=self.normalize_std,
        )

        # inverse transform (for visualization/debug)
        self.unnormalize = unnormalize_function(
            mean=self.normalize_mean,
            std=self.normalize_std,
        )
        self.img_size = img_size
        # ---- Final resize (ensure fixed output resolution) ----
        self.final_resize = v2.Resize(size=self.img_size)

    def add_uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add uniform noise in [-scale, +scale].
        """
        if self.uniform_noise_scale <= 0:
            return x
        noise = torch.empty_like(x).uniform_(
            -self.uniform_noise_scale,
            self.uniform_noise_scale
        )
        return x + noise

    def apply_per_image_transforms(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for xi in x:
            if torch.rand(()) < self.p_affine:
                xi_dtype = xi.dtype
                xi = self.to_pil(xi)
                xi = self.affine(xi)
                xi = self.pil_to_tensor(xi).to(dtype=xi_dtype)

            xi = self.brightness(xi)
            xi = self.contrast(xi)
            xi = self.blur(xi)

            out.append(xi)

        return torch.stack(out)

    def forward(self, x: torch.Tensor):
        """
        Apply the full augmentation pipeline.
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        # Keep augmentations on CPU for this pipeline.
        device = x.device
        x = x.cpu()
        # 0) upscale by a factor of 2 with bicubic interpolation (before all augmentations)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # 1) embed into black background
        # x = embed_in_black_background(x, target_size=2*112)
        x = embed_in_black_background(x, target_size=104)

        # ---- Geometric + Photometric ----
        x = self.apply_per_image_transforms(x)

        # ---- Noise + Occlusion ----
        x = self.add_uniform_noise(x)
        if self.use_erasing:
            x = torch.stack([self.erasing(xi) for xi in x])

        x = torch.clamp(x, 0.0, 1.0)

        # ---- Normalize to [-1, 1] using configured mean/std ----
        x = self.normalize(x)

        # Back to original device
        x = x.to(device)

        if single:
            x = x.squeeze(0)
        return x

class SyntheticEngravingDataset(Dataset):
    def __init__(self, params, nb_signatures_per_code=2, batch_size=256, split="train", seed=0):
        self.split = split
        self.seed = seed
        self.rng_train_code      = np.random.default_rng(seed  )
        self.rng_train_signature = np.random.default_rng(seed+1) 
        self.rng_val_code        = np.random.default_rng(seed+2)
        self.rng_val_signature   = np.random.default_rng(seed+3)

        self.nb_signatures_per_code = nb_signatures_per_code
        self.params = params

        if split == "train":
            self.nb_batch_per_epoch = 100
            self.batch_size = batch_size

            self.nb_codes_per_batch = self.batch_size // self.nb_signatures_per_code
            if self.nb_codes_per_batch * self.nb_signatures_per_code != self.batch_size:
                raise ValueError(f"Batch size={self.batch_size} is not divisible by the number of signatures per code={self.nb_signatures_per_code}")

        elif split == "val":
            self.nb_batch_per_epoch = 20
            self.batch_size = 100
            self.nb_codes_per_batch = 1
            self.nb_signatures_per_code = self.batch_size

        elif split == "test":
            self.nb_batch_per_epoch = 20
            self.batch_size = 100
        else:
            raise ValueError("Only train, val and test supported")

        self.transform_test   = RandomAffineWithPhotometricNoise(img_size=params.img_size,use_erasing=False)
        self.transform = RandomAffineWithPhotometricNoise(img_size=params.img_size,use_erasing=True)

    def __len__(self):
            return self.nb_batch_per_epoch
            
    def __getitem__(self, idx):
        H, W = self.params.size[0] * 8, self.params.size[1] * 8

        if self.split in ("train", "val"):
            images = []
            biases = []
            rng_code = self.rng_train_code if self.split == "train" else self.rng_val_code
            rng_signature = self.rng_train_signature if self.split == "train" else self.rng_val_signature

            for i in range(self.nb_codes_per_batch):
                DataMatrix = rng_code.integers(0, 2, size=self.params.size) * 255
                bits = (DataMatrix > 0).reshape(-1).astype(np.int64)
                bias = torch.from_numpy(bits)  # (64,)

                for j in range(self.nb_signatures_per_code):
                    Noise = rng_signature.standard_normal((H, W, 2))

                    img_generated, _ = create_I_from_noise_and_DataMatrix(DataMatrix, Noise)
                    img_generated = cv2.resize(img_generated, self.params.img_size, interpolation=cv2.INTER_CUBIC)

                    img = Image.fromarray(img_generated).convert("L")
                    img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

                    images.append(img)
                    biases.append(bias)

            images = torch.stack(images, dim=0)  # (B, 1, H, W)
            biases = torch.stack(biases, dim=0)  # (B, 64)

            # apply transform twice to create two views
            x1 = self.transform(images)
            x2 = self.transform(images)
            return [x1, x2], biases

        elif self.split == "test":
            images0 = []
            images1 = []

            DataMatrix0 = self.rng_val_code.integers(0, 2, size=self.params.size) * 255
            DataMatrix1 = self.rng_val_code.integers(0, 2, size=self.params.size) * 255

            for j in range(self.batch_size):
                Noise = self.rng_val_signature.standard_normal((H, W, 2))

                img_generated0, _ = create_I_from_noise_and_DataMatrix(DataMatrix0, Noise)
                img_generated1, _ = create_I_from_noise_and_DataMatrix(DataMatrix1, Noise)
                img_generated0 = cv2.resize(img_generated0, self.params.img_size, interpolation=cv2.INTER_CUBIC) #problème peut etre lié à linterpolation
                img_generated1 = cv2.resize(img_generated1, self.params.img_size, interpolation=cv2.INTER_CUBIC)

                img0 = Image.fromarray(img_generated0).convert("L")
                img1 = Image.fromarray(img_generated1).convert("L")
                img0 = torch.from_numpy(np.array(img0)).unsqueeze(0).float() / 255.0
                img1 = torch.from_numpy(np.array(img1)).unsqueeze(0).float() / 255.0

                images0.append(img0)
                images1.append(img1)

            images0 = torch.stack(images0, dim=0)  # (B, 1, H, W)
            images1 = torch.stack(images1, dim=0)  # (B, 1, H, W)

            # apply transform twice to create two views
            x1 = self.transform_test(images0)
            x2 = self.transform_test(images1)
            return x1, x2

def get_synthetic_engraving_loaders(batch_size=256, nb_signatures_per_code=2, num_workers=0):
    from torch.utils.data import DataLoader
    params= Parameters()
    train_ds = SyntheticEngravingDataset(params, batch_size=batch_size, nb_signatures_per_code=nb_signatures_per_code, split="train")
    val_ds   = SyntheticEngravingDataset(params, batch_size=batch_size, nb_signatures_per_code=nb_signatures_per_code, split="val"  )
    test_ds  = SyntheticEngravingDataset(params, batch_size=batch_size, nb_signatures_per_code=nb_signatures_per_code, split="test" )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x[0])
    return train_loader, val_loader, test_loader

def visualize_masked_comparison(img_a, img_b, c_a, c_b, params, label_a="s0c0", label_b="s0c1"):
    a = img_a / 255.0
    b = img_b / 255.0

    # masks
    mask_a = cv2.resize(c_a, params.img_size, interpolation=cv2.INTER_NEAREST) > 0
    mask_b = cv2.resize(c_b, params.img_size, interpolation=cv2.INTER_NEAREST) > 0
    mask = mask_a * mask_b

    diff = np.abs(a - b)
    mse = np.mean((a - b) ** 2)
    masked_diff = np.where(mask, diff, np.nan)
    masked_mse =  np.mean((a - b) ** 2 * mask)

    cmap = plt.cm.inferno.copy() # type: ignore 
    cmap.set_bad(color="black")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    axes[0,0].imshow(a, cmap="gray")
    axes[0,0].set_title(label_a)
    axes[0,1].imshow(b, cmap="gray")
    axes[0,1].set_title(label_b)
    im_top = axes[0,2].imshow(diff, cmap=cmap, vmin=0, vmax=1)
    axes[0,2].set_title(f"Diff mse={mse:.4f}")

    axes[1,0].imshow(a * mask, cmap="gray")
    axes[1,0].set_title(f"{label_a} (masked)")
    axes[1,1].imshow(b * mask, cmap="gray")
    axes[1,1].set_title(f"{label_b} (masked)")
    im_bottom = axes[1,2].imshow(masked_diff, cmap=cmap, vmin=0, vmax=1)
    axes[1,2].set_title(f"Masked diff mse={masked_mse:.4f}")

    # Add colorbars for heatmaps
    plt.colorbar(im_top, ax=axes[0,2], fraction=0.046, pad=0.04)
    plt.colorbar(im_bottom, ax=axes[1,2], fraction=0.046, pad=0.04)

    for ax in axes.flatten(): ax.axis("off")
    plt.tight_layout()
    plt.show()

def test_visualization():
    print("Running structured visualization test...")

    params = Parameters()
    # ================= STEP 1: RAW VISUALIZATION (NO AUGMENTATION) =================
    print("STEP 1: Raw images + mask + diff (no transform)")

    # Build raw images without augmentation
    def get_raw(code_idx, signature_idx):
        H, W = params.size[0] * 8, params.size[1] * 8
        DataMatrix = np.random.RandomState(code_idx).randint(0, 2, size=params.size) * 255
        Noise = np.random.RandomState(signature_idx).randn(H, W, 2)

        img, _ = create_I_from_noise_and_DataMatrix(DataMatrix, Noise)
        img = cv2.resize(img, params.img_size, interpolation=cv2.INTER_CUBIC)

        return img, DataMatrix

    img_a, c_a = get_raw(0, 0)
    img_b, c_b = get_raw(1, 0)
    visualize_masked_comparison(img_a, img_b, c_a, c_b, params, label_a="s0c0", label_b="s0c1")


    # ---- Additional comparison: s0c0 vs s1c0 (same code, different signatures) ----
    img_a2, c_a2 = get_raw(0, 0)
    img_b2, c_b2 = get_raw(0, 1)
    visualize_masked_comparison(img_a2, img_b2, c_a2, c_b2, params, label_a="s0c0", label_b="s1c0")

    # ================= STEP 2: BATCH STRUCTURE =================
    print("STEP 2: Batch structure (train, val & test)")

    train_loader, val_loader, test_loader = get_synthetic_engraving_loaders(
        batch_size=256,
        nb_signatures_per_code=2,
        num_workers=0,
    )

    def print_loader_info(name, loader):
        dataset = loader.dataset
        print(
            f"{name}: "
            f"split={dataset.split}, "
            f"len={len(dataset)}, "
            f"batch_size={dataset.batch_size}, "
            f"nb_codes_per_batch={getattr(dataset, 'nb_codes_per_batch', None)}, "
            f"nb_signatures_per_code={dataset.nb_signatures_per_code}"
        )

    print_loader_info("TRAIN loader", train_loader)
    print_loader_info("VAL loader", val_loader)
    print_loader_info("TEST loader", test_loader)

    def show_train_like_batches(name, loader):
        print(f"{name} batch")

        batch_idx = 0
        for batch in loader:
            if batch_idx == 2:
                break

            xs, bias = batch
            nb_images = min(4, xs[0].size(0))
            fig, axes = plt.subplots(2, nb_images, figsize=(3 * nb_images, 6))
            if nb_images == 1:
                axes = axes.reshape(2, 1)
            fig.suptitle(f"{name} batch {batch_idx}: two augmented views (x1 / x2)", fontsize=14)
            for i in range(nb_images):
                axes[0, i].imshow(xs[0][i].squeeze().numpy(), cmap="gray")
                bits = bias[i].cpu().numpy().astype(int)
                val = int("".join(bits.astype(str)), 2)
                axes[0, i].set_title(f"{val}")
                axes[0, i].axis("off")

                axes[1, i].imshow(xs[1][i].squeeze().numpy(), cmap="gray")
                bits = bias[i].cpu().numpy().astype(int)
                val = int("".join(bits.astype(str)), 2)
                axes[1, i].set_title(f"{val}")
                axes[1, i].axis("off")

            plt.tight_layout()
            plt.show()
            batch_idx += 1

    # ---- TRAIN ----
    show_train_like_batches("TRAIN", train_loader)

    # ---- VAL ----
    show_train_like_batches("VAL", val_loader)

    # ---- TEST ----
    print("TEST batch")

    batch_idx = 0
    for batch in test_loader:
        if batch_idx == 2:
            break

        sic0, sic1 = batch
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"TEST batch {batch_idx}: sic0 / sic1", fontsize=14)
        for i in range(4):
            axes[0, i].imshow(sic0[i].squeeze().numpy(), cmap="gray")
            axes[1, i].imshow(sic1[i].squeeze().numpy(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()
        batch_idx += 1

if __name__ == "__main__":
    test_visualization()


# %%
