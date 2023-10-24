import multiprocessing
import os
from pathlib import Path


from torch.utils.data import DataLoader, Dataset, random_split

from fabriclab.pipeline.dataset import LabDataset

filepath = Path(__file__)
PROJECTPATH = os.getcwd()
NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class LabDataModule:
    def __init__(
        self,
        dataset: Dataset = LabDataset,
        data_dir: str = "data",
        split: bool = True,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
    ):
        super().__init__()
        self.data_dir = os.path.join(PROJECTPATH, data_dir, "cache")
        self.dataset = dataset
        self.split = split
        self.train_size = train_size
        self.num_workers = num_workers

    def prepare_data(self):
        # setup
        full_dataset = self.dataset(self.data_dir)
        train_size = int(len(full_dataset) * self.train_size)
        test_size = len(full_dataset) - train_size
        self.train_data, self.val_data = random_split(full_dataset, lengths=[train_size, test_size])
        self.test_data = self.dataset(self.data_dir)

        # create dataloaders
        self.train_dataload = DataLoader(self.train_data, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_data, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.val_data, num_workers=self.num_workers)
