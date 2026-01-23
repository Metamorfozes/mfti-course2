from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset


class ImagePathsDataset(Dataset):
    def __init__(self, root_dir: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")):
        self.root_dir = Path(root_dir)
        self.exts = tuple(ext.lower() for ext in exts)
        self.paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.exts
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> str:
        return str(self.paths[idx])
