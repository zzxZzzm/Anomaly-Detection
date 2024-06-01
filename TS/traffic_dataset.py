import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, pred_len=1):
        self.data = pd.read_csv(csv_file).values
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y

def get_dataloader(csv_file, batch_size=32, seq_len=5, pred_len=1, shuffle=True):
    dataset = TrafficDataset(csv_file, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
