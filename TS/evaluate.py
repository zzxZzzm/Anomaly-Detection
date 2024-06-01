import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from carn import StudentNetwork
from traffic_dataset import TrafficDataset
from distillation_loss import reconstruction_loss, distillation_loss, estimator_loss

def calculate_metrics(true, pred):
    mse = np.mean((true - pred) ** 2)
    mae = np.mean(np.abs(true - pred))
    return mse, mae

def evaluate(model, dataloader):
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.unsqueeze(1).float()
            y = y.unsqueeze(1).float()
            output, _, _, _ = model(x, None)
            all_true.append(y.numpy())
            all_pred.append(output.numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    mse, mae = calculate_metrics(all_true, all_pred)
    return mse, mae, all_true, all_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to the traffic test data CSV file')
    args = parser.parse_args()
    
   
    dataset = TrafficDataset(args.data, seq_len=5, pred_len=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    

    model = StudentNetwork()
    model.load_state_dict(torch.load('student_model.pth'))
    
    mse, mae, true, pred = evaluate(model, dataloader)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(true.flatten(), label='True')
    plt.plot(pred.flatten(), label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Traffic Speed')
    plt.legend()
    plt.savefig('evaluation_results.png')
