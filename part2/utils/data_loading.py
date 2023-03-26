import torch
import h5py
from torch.utils.data import Dataset


class CombustionSystemDataset(Dataset):
    """" combustion system Dataset
    this class is for reading the data from file and covert it into class of Dataset,so we could use the data
    loader to train our models
    the return format is torch.tensor
    """
    def __init__(self, mat_file_path, data_key, label_key=None, transform=None):
        with h5py.File(mat_file_path, 'r') as f:
            self.data = torch.tensor(f[data_key][:], dtype=torch.float32).t()
            self.labels = torch.tensor(f[label_key][:], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].reshape(1, 250, 100)
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
