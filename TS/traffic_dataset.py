import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def clean_traffic_data(data):
    """清洗交通数据，处理异常值和字符串数据"""
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.split('.').str[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(method='forward').fillna(method='backward').fillna(df.mean())
    
    for col in df.columns:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df.values.astype(np.float32)

class TrafficDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, pred_len=1):
        try:
            if isinstance(csv_file, str):
                if csv_file.endswith('.xlsx'):
                    data = pd.read_excel(csv_file, header=None).values
                elif csv_file.endswith('.csv'):
                    data = pd.read_csv(csv_file, header=None).values
                else:
                    raise ValueError(f"Unsupported file format: {csv_file}")
            else:
                data = csv_file  
            
            self.data = clean_traffic_data(data)
            print(f"Loaded data shape: {self.data.shape}")
            
        except Exception as e:
            print(f"Error loading data from {csv_file}: {e}")
            raise
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        if len(self.data) < seq_len + pred_len:
            raise ValueError(f"Dataset too small: {len(self.data)} < {seq_len + pred_len}")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y

def get_dataloader(csv_file, batch_size=32, seq_len=5, pred_len=1, shuffle=True):
    """创建数据加载器"""
    try:
        dataset = TrafficDataset(csv_file, seq_len, pred_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"Created dataloader with {len(dataset)} samples")
        return dataloader
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        raise
