# https://github.com/reiinakano/invariant-risk-minimization/blob/master/colored_mnist.py (identical to https://github.com/reiinakano/invariant-risk-minimization/blob/master/colored_mnist.py) 
#%% 
import os
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import kornia.augmentation as K
from torchvision.transforms import Normalize

import torch.nn as nn

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr

def unnormalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    # Compute unnormalize parameters.
    return Normalize(mean=[-m/s for m, s in zip(mean, std)],
                     std=[1/s for s in std])

def normalize_function(mean, std):
    # If mean and std are scalars, wrap them in a list.
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]
    return Normalize(mean=mean, std=std)

class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(
          os.path.join(self.root, 'ColoredMNIST', env) + '.pt',
          weights_only=False,
      )
    elif env == 'all_train':
      self.data_label_tuples = torch.load(
          os.path.join(self.root, 'ColoredMNIST', 'train1.pt'),
          weights_only=False,
      ) + torch.load(
          os.path.join(self.root, 'ColoredMNIST', 'train2.pt'),
          weights_only=False,
      )
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target, bias_label, digit_label = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, bias_label, digit_label

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    train1_path = os.path.join(colored_mnist_dir, 'train1.pt')
    train2_path = os.path.join(colored_mnist_dir, 'train2.pt')
    test_path = os.path.join(colored_mnist_dir, 'test.pt')

    if os.path.exists(train1_path) and os.path.exists(train2_path) and os.path.exists(test_path):
      try:
        sample = torch.load(train1_path, weights_only=False)[0]
        if isinstance(sample, tuple) and len(sample) == 4:
          print('Colored MNIST dataset already exists')
          return
        print('Colored MNIST cache found but in old format, regenerating')
      except Exception:
        print('Colored MNIST cache found but unreadable, regenerating')

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):  # type: ignore
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      bias_label = 0 if color_red else 1
      digit_label = int(label)
      sample = (Image.fromarray(colored_arr), int(binary_label), int(bias_label), digit_label)

      if idx < 20000:
        train1_set.append(sample)
      elif idx < 40000:
        train2_set.append(sample)
      else:
        test_set.append(sample)

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train1_set, train1_path)
    torch.save(train2_set, train2_path)
    torch.save(test_set, test_path)

class RandomAffineWithPhotometricNoise(nn.Module):
    """
    Apply RandomAffine followed by photometric and noise augmentations:
    - Brightness
    - Contrast
    - Uniform noise
    - Random erasing

    Expected input: Tensor of shape (B, C, H, W) or (C, H, W)
    All operations are done on CPU (Kornia-safe).
    """

    def __init__(
        self,
        img_size=(32, 32),
        padded_size=42,
        degrees=(-20, 20),
        translate=(0.2, 0.2),
        scale=(1.4, 1.4),
        shear=(-10, 10),
        brightness_max=0.4,
        contrast_max=0.4,
        p_affine=1.0,
        p_photometric=1.0,
        normalize_mean=0.5,
        normalize_std=0.5,
    ):
        super().__init__()

        # ---- Geometric augmentation ----
        self.affine = K.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            p=p_affine,
            keepdim=True,
            padding_mode="zeros",
        )

        # ---- Photometric augmentations ----
        self.brightness = K.RandomBrightness(
            brightness=(1 - brightness_max, 1 + brightness_max),
            p=p_photometric,
        )

        self.contrast = K.RandomContrast(
            contrast=(1 - contrast_max, 1 + contrast_max),
            p=p_photometric,
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
        self.padded_size = padded_size
        # ---- Final resize (ensure fixed output resolution) ----
        # Kornia expects size as (H, W)
        self.final_resize = K.Resize(size=self.img_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the full augmentation pipeline.
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        device = x.device
        x = x.cpu()

        # ---- Embed into black square before augmentation ----
        B, C, H, W = x.shape
        padded_size = self.padded_size
        padded_batch = torch.zeros((B, C, padded_size, padded_size), dtype=x.dtype)

        # center placement
        start_h = (padded_size - H) // 2
        start_w = (padded_size - W) // 2
        padded_batch[:, :, start_h:start_h + H, start_w:start_w + W] = x

        x = padded_batch

        # ---- Geometric ----
        x = self.affine(x)

        # ---- Photometric ----
        x = self.brightness(x)
        x = self.contrast(x)


        # ---- Ensure fixed output resolution ----
        # (B, 1, H, W) -> (B, 1, img_size[0], img_size[1])
        x = self.final_resize(x)

        # Clamp before normalization
        x = torch.clamp(x, 0.0, 1.0)

        # ---- Normalize to [-1, 1] using configured mean/std ----
        x = self.normalize(x)

        # Back to original device
        x = x.to(device)

        if single:
            x = x.squeeze(0)

        return x

class DeterministicPadResizeNormalize(nn.Module):
    """
    Deterministic preprocessing that mirrors the non-random structural steps of
    RandomAffineWithPhotometricNoise:
    - embed the image into a black square of size `padded_size`
    - resize to `img_size`
    - clamp to [0, 1]
    - normalize with the configured mean/std

    Expected input: Tensor of shape (B, C, H, W) or (C, H, W)
    """

    def __init__(
        self,
        img_size=(28, 28),
        padded_size=42,
        normalize_mean=0.5,
        normalize_std=0.5,
    ):
        super().__init__()
        self.img_size = img_size
        self.padded_size = padded_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.normalize = normalize_function(mean=normalize_mean, std=normalize_std)
        self.unnormalize = unnormalize_function(mean=normalize_mean, std=normalize_std)
        self.final_resize = K.Resize(size=self.img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        device = x.device
        x = x.cpu()

        B, C, H, W = x.shape
        padded_size = self.padded_size
        padded_batch = torch.zeros((B, C, padded_size, padded_size), dtype=x.dtype)

        start_h = (padded_size - H) // 2
        start_w = (padded_size - W) // 2
        padded_batch[:, :, start_h:start_h + H, start_w:start_w + W] = x
        x = padded_batch

        x = self.final_resize(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.normalize(x)
        x = x.to(device)

        if single:
            x = x.squeeze(0)

        return x

def preview_colored_mnist(root='./data', env='train1', num_samples=8):
  import matplotlib.pyplot as plt

  ds = ColoredMNIST(root=root, env=env)
  n = min(num_samples, len(ds))

  fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
  if n == 1:
    axes = [axes]

  for i in range(n):
    img, bias_label, digit_label = ds[i]
    print(f"Sample {i}: bias={bias_label}, digit={digit_label}, img shape={img.size}, img mode={img.mode}")
    axes[i].imshow(img)
    axes[i].set_title(f'b={bias_label}, d={digit_label}')
    axes[i].axis('off')

  fig.suptitle(f'ColoredMNIST preview ({env})')
  plt.tight_layout()
  plt.show()

def preview_affine_noise(root='./data', env='train1', num_samples=4):
  import matplotlib.pyplot as plt
  import torchvision.transforms as T

  ds = ColoredMNIST(root=root, env=env, transform=T.ToTensor())
  aug = RandomAffineWithPhotometricNoise()

  n = min(num_samples, len(ds))

  fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
  if n == 1:
    axes = np.array([axes])

  for i in range(n):
    img, bias_label, digit_label = ds[i]

    aug_img = aug(img)

    # unnormalize for visualization
    aug_img = aug.unnormalize(aug_img)

    img_np = img.permute(1, 2, 0).numpy()
    aug_np = aug_img.permute(1, 2, 0).detach().numpy()

    img_np = np.clip(img_np, 0, 1)
    aug_np = np.clip(aug_np, 0, 1)

    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title(f'orig, b={bias_label}, d={digit_label}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(aug_np.squeeze(), cmap='gray')
    axes[i, 1].set_title('augmented')
    axes[i, 1].axis('off')

  fig.suptitle(f'RandomAffineWithPhotometricNoise ({env})')
  plt.tight_layout()
  plt.show()

class TensorTransformWrapper:
    """Convert PIL image to tensor before applying a tensor transform."""

    def __init__(self, tensor_transform):
        self.to_tensor = transforms.ToTensor()
        self.tensor_transform = tensor_transform

    def __call__(self, img):
        x = self.to_tensor(img)
        return self.tensor_transform(x)

class MultiViewWrapper:
    """Return a list of independently augmented views of the same image."""

    def __init__(self, base_transform, num_views=2):
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, img):
        return [self.base_transform(img) for _ in range(self.num_views)]

def get_colored_mnist_loaders(
    root='./data',
    batch_size=256,
    num_workers=0,
    eval_mode=False):
    train_aug = RandomAffineWithPhotometricNoise(img_size=(32, 32))
    eval_aug = DeterministicPadResizeNormalize(img_size=(32, 32))

    eval_transform = TensorTransformWrapper(eval_aug)

    if eval_mode:
        train_transform = eval_transform
    else:
        train_transform = MultiViewWrapper(
            TensorTransformWrapper(train_aug),
            num_views=2,
        )

    train_ds = ColoredMNIST(root=root, env='all_train', transform=train_transform)
    test_ds = ColoredMNIST(root=root, env='test', transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

def get_mnist_loaders(
    root='./data',
    batch_size=256,
    num_workers=0,
    eval_mode=False,
):
    train_aug = RandomAffineWithPhotometricNoise(img_size=(32, 32))
    eval_aug = DeterministicPadResizeNormalize(img_size=(32, 32))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        eval_aug,
    ])

    if eval_mode:
        train_transform = eval_transform
    else:
        train_transform = MultiViewWrapper(
            transforms.Compose([
                transforms.ToTensor(),
                train_aug,
            ]),
            num_views=2,
        )

    base_train_ds = datasets.MNIST(root=root, train=True, download=True, transform=train_transform)
    base_test_ds = datasets.MNIST(root=root, train=False, download=True, transform=eval_transform)

    class MNISTWithBias(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            x, y = self.ds[idx]
            bias = torch.zeros_like(torch.tensor(y))
            return x, bias, y

    train_ds = MNISTWithBias(base_train_ds)
    test_ds = MNISTWithBias(base_test_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

def summarize_bias_correlations(root='./data'):
    """Print empirical proportions P(b | y) for train, val, and test splits."""

    split_to_env = {
        'train': 'all_train',
        'test': 'test',
    }

    for split_name, env in split_to_env.items():
        ds = ColoredMNIST(root=root, env=env, transform=None)

        y_list = []
        b_list = []
        d_list = []
        for _, b, d in ds:
            b_list.append(int(b))
            d_list.append(int(d))

        y = np.array(y_list)
        b = np.array(b_list)
        d = np.array(d_list)

        print(f'\n{split_name.upper()} bias summary:')
        print(f'  n = {len(ds)}')

        for y_val in [0, 1]:
            mask = (y == y_val)
            n_y = int(mask.sum())
            if n_y == 0:
                print(f'  P(b | y={y_val}): no samples')
                continue

            p_b0 = float((b[mask] == 0).mean())
            p_b1 = float((b[mask] == 1).mean())
            print(
                f'  P(b=0 | y={y_val}) = {p_b0:.4f} | '
                f'P(b=1 | y={y_val}) = {p_b1:.4f} | '
                f'count = {n_y}'
            )

        digit_counts = np.bincount(d, minlength=10)
        print('  digit counts =', digit_counts.tolist())

def main_colored_mnist():

  train_loader, val_loader, _ = get_colored_mnist_loaders(batch_size=4)
  summarize_bias_correlations()

  train_views, train_b, train_d = next(iter(train_loader))
  print('train:')
  print('  num views =', len(train_views))
  print('  view[0] shape =', train_views[0].shape)
  print('  bias labels =', train_b)
  print('  digit labels =', train_d)

  val_x, val_b, val_d = next(iter(val_loader))
  print('val:')
  print('  x shape =', val_x.shape)
  print('  bias labels =', val_b)
  print('  digit labels =', val_d)

  import matplotlib.pyplot as plt

  # ---- Visualize TRAIN (multiple views) ----
  views = train_views  # list of tensors [num_views][B, C, H, W]
  num_views = len(views)
  B = views[0].shape[0]

  fig, axes = plt.subplots(num_views, B, figsize=(2 * B, 2 * num_views), squeeze=False)

  for v in range(num_views):
    for i in range(B):
      img = views[v][i]
      img = img.permute(1, 2, 0).cpu().numpy()
      img = (img * 0.5) + 0.5  # unnormalize
      img = np.clip(img, 0, 1)
      axes[v][i].imshow(img)
      axes[v][i].set_title(f"v{v} b={train_b[i]} d={train_d[i]}")
      axes[v][i].axis('off')

  fig.suptitle("Train augmented views")
  plt.tight_layout()
  plt.show()

  # ---- Visualize VAL ----
  fig, axes = plt.subplots(1, B, figsize=(2 * B, 2), squeeze=False)
  axes = axes[0]

  for i in range(B):
    img = val_x[i]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    axes[i].imshow(img)
    axes[i].set_title(f"b={val_b[i]} d={val_d[i]}")
    axes[i].axis('off')

  fig.suptitle("Validation samples")
  plt.tight_layout()
  plt.show()

def main_mnist():
  train_loader, val_loader, _ = get_mnist_loaders(batch_size=4)

  train_views, train_b, train_y = next(iter(train_loader))
  print('mnist train:')
  print('  num views =', len(train_views))
  print('  view[0] shape =', train_views[0].shape)
  print('  bias =', train_b)
  print('  labels =', train_y)

  val_x, val_b, val_y = next(iter(val_loader))
  print('mnist val:')
  print('  x shape =', val_x.shape)
  print('  bias =', val_b)
  print('  labels =', val_y)

  import matplotlib.pyplot as plt

  views = train_views
  num_views = len(views)
  B = views[0].shape[0]

  fig, axes = plt.subplots(num_views, B, figsize=(2 * B, 2 * num_views), squeeze=False)

  for v in range(num_views):
    for i in range(B):
      img = views[v][i]
      img = img.permute(1, 2, 0).cpu().numpy()
      img = (img * 0.5) + 0.5
      img = np.clip(img, 0, 1)
      axes[v][i].imshow(img.squeeze(), cmap='gray')
      axes[v][i].set_title(f"v{v} b={train_b[i]} y={train_y[i]}")
      axes[v][i].axis('off')

  fig.suptitle("MNIST train augmented views")
  plt.tight_layout()
  plt.show()

  fig, axes = plt.subplots(1, B, figsize=(2 * B, 2), squeeze=False)
  axes = axes[0]

  for i in range(B):
    img = val_x[i]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"b={val_b[i]} y={val_y[i]}")
    axes[i].axis('off')

  fig.suptitle("MNIST validation samples")
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main_colored_mnist()
#   main_mnist()


# %%
