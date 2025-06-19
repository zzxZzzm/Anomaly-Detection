import torch
import numpy as np
import pandas as pd


def clean_data(data):
    """清洗数据，处理异常值"""
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(method='forward').fillna(method='backward').fillna(0)
    
    for col in df.columns:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=Q1, upper=Q3)
    
    return df.values.astype(np.float32)

def load_matrix(file_path):
    """加载矩阵文件，支持CSV和Excel"""
    try:
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, header=None).values
        else:
            data = pd.read_csv(file_path, header=None).values
        return clean_data(data)
    except Exception as e:
        print(f"Error loading matrix from {file_path}: {e}")
        raise

def load_data(file_path, len_train, len_val):
    """加载时序数据，支持CSV和Excel"""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None).values
        else:
            df = pd.read_csv(file_path, header=None).values
        
        df = clean_data(df)
        
        train = df[:len_train]
        val = df[len_train:len_train + len_val]
        test = df[len_train + len_val:]
        
        return train, val, test
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def data_transform(data, n_his, n_pred, day_slot, device):
    """数据转换函数"""
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    y = np.zeros([n_day * n_slot, n_route])
    
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
            y[t] = data[e + n_pred - 1]
    
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
