#%%
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None  # type: ignore


ATTRIBUTE_SHORTCUTS = {
    "a": "Attractive",
    "b": "Big_Nose",
    "e": "Bags_Under_Eyes",
    "o": "Mouth_Slightly_Open",
    "m": "Male",
    "y": "Young",
}


def _resolve_attribute_name(name: str) -> str:
    key = name.strip()
    return ATTRIBUTE_SHORTCUTS.get(key, key)


def _resolve_attribute_spec(spec: str) -> list[str]:
    return [_resolve_attribute_name(part) for part in spec.split("&")]


def celeba_num_classes_from_spec(spec: str) -> int:
    return 2 ** len(_resolve_attribute_spec(spec))


def encode_celeba_attribute_combination(row, attrs: list[str]) -> int:
    value = 0
    for attr_name in attrs:
        value = (value << 1) | int(int(row[attr_name]) == 1)
    return value


def parse_celeba_experiment(experiment: str) -> tuple[str, str]:
    cleaned = experiment.replace(" ", "")
    parts = cleaned.split("/")
    if len(parts) != 2 or not parts[0].startswith("T=") or not parts[1].startswith("S="):
        raise ValueError(
            "experiment must look like 'T=a/S=m', 'T=a&o/S=m', or 'T=a/S=m&y'"
        )
    return parts[0][2:], parts[1][2:]


class SimCLRTwoCropTransform:
    def __init__(self):
        color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = T.Compose([
            T.RandomResizedCrop(128, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]


class CelebAAttributes(Dataset):
    """
    CelebA wrapper around Hugging Face datasets.

    Dataset source:
      load_dataset("flwrlabs/celeba")

    Default setup:
      - target label: Attractive
      - sensitive / bias attribute: Male
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_attr="Attractive",
        bias_attr="Male",
        experiment=None,
    ):
        del root  # kept for API compatibility with other loaders
        self.transform = transform
        if experiment is not None:
            target_attr, bias_attr = parse_celeba_experiment(experiment)
        self.target_attr = target_attr
        self.bias_attr = bias_attr
        self.target_attrs = _resolve_attribute_spec(target_attr)
        self.bias_attrs = _resolve_attribute_spec(bias_attr)

        if load_dataset is None:
            raise RuntimeError(
                "The Hugging Face `datasets` package is required for CelebA. "
                "Install it in your current environment with `pip install datasets`."
            )

        split_aliases = {
            "train": ["train"],
            "val": ["valid", "validation", "val"],
            "test": ["test"],
        }
        if split not in split_aliases:
            raise ValueError("split must be one of {'train', 'val', 'test'}")

        dataset_dict = load_dataset("flwrlabs/celeba")
        ds = None
        for split_name in split_aliases[split]:
            if split_name in dataset_dict:
                ds = dataset_dict[split_name]
                break

        if ds is None:
            available_splits = list(dataset_dict.keys())
            if split == "val" and "test" in dataset_dict:
                ds = dataset_dict["test"]
            elif split == "test" and "val" in dataset_dict:
                ds = dataset_dict["val"]
            elif "train" in dataset_dict:
                raise RuntimeError(
                    f"Could not find a suitable split for {split!r}. "
                    f"Available splits: {available_splits}"
                )
            else:
                raise RuntimeError(f"No usable splits found in flwrlabs/celeba: {available_splits}")

        column_names = set(ds.column_names)
        for attr_name in self.target_attrs:
            if attr_name not in column_names:
                raise ValueError(
                    f"Unknown target attribute {attr_name!r} from spec {self.target_attr!r}. "
                    f"Available columns include: {sorted(column_names)[:10]}..."
                )
        for attr_name in self.bias_attrs:
            if attr_name not in column_names:
                raise ValueError(
                    f"Unknown bias attribute {attr_name!r} from spec {self.bias_attr!r}. "
                    f"Available columns include: {sorted(column_names)[:10]}..."
                )
        if "image" not in column_names:
            raise ValueError("Expected an `image` column in flwrlabs/celeba.")

        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            # datasets Image feature usually returns PIL already, but keep this robust
            img = Image.fromarray(img)

        y = encode_celeba_attribute_combination(row, self.target_attrs)
        bias = encode_celeba_attribute_combination(row, self.bias_attrs)

        if self.transform is not None:
            img = self.transform(img)

        return img, bias, y


def show_simclr_augmentations(dataset, idx=0):
    row = dataset.ds[idx]
    img = row["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    transform = SimCLRTwoCropTransform()
    view1, view2 = transform(img)

    original_resized = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
    ])(img)

    images = [original_resized, view1, view2]
    titles = ["Original", "SimCLR View 1", "SimCLR View 2"]

    plt.figure(figsize=(12, 4))
    for i, (img_tensor, title) in enumerate(zip(images, titles), start=1):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        if title != "Original":
            img_np = img_np * std + mean
        img_np = img_np.clip(0, 1)
        plt.subplot(1, 3, i)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


show_byol_augmentations = show_simclr_augmentations


def get_celeba_loaders(
    root,
    batch_size=32,
    num_workers=4,
    eval_mode=False,
    target_attr="Attractive",
    bias_attr="Male",
    experiment=None,
):
    val_transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if eval_mode:
        train_transform = val_transform
    else:
        train_transform = SimCLRTwoCropTransform()

    train_ds = CelebAAttributes(
        root=root,
        split="train",
        transform=train_transform,
        target_attr=target_attr,
        bias_attr=bias_attr,
        experiment=experiment,
    )
    val_ds = CelebAAttributes(
        root=root,
        split="val",
        transform=val_transform,
        target_attr=target_attr,
        bias_attr=bias_attr,
        experiment=experiment,
    )
    test_ds = CelebAAttributes(
        root=root,
        split="test",
        transform=val_transform,
        target_attr=target_attr,
        bias_attr=bias_attr,
        experiment=experiment,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root = "./data"

    train_loader, val_loader, test_loader = get_celeba_loaders(root, batch_size=4)

    x, b, y = next(iter(train_loader))
    print("Train batch:")
    print("  x1 shape:", x[0].shape)
    print("  x2 shape:", x[1].shape)
    print("  bias:", b[:5])
    print("  target:", y[:5])

    x_val, b_val, y_val = next(iter(val_loader))
    print("Val batch:")
    print("  x shape:", x_val.shape)
    print("  bias:", b_val[:5])
    print("  target:", y_val[:5])

    x_test, b_test, y_test = next(iter(test_loader))
    print("Test batch:")
    print("  x shape:", x_test.shape)
    print("  bias:", b_test[:5])
    print("  target:", y_test[:5])

# %%
if __name__ == '__main__':
    result = parse_celeba_experiment('T=a/S=m')
    print(result)
# %%
