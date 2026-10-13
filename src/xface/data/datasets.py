from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            for image_path in sorted((self.root / class_name).glob("*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
