import torch
import numpy as np
import argparse
import pandas as pd
from carn import StudentNetwork
from traffic_dataset import TrafficDataset
from torch.utils.data import DataLoader

def load_model(model_path):
    model = StudentNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, data):
    data = torch.tensor(data).unsqueeze(1).float() 
    with torch.no_grad():
        prediction, _, _, _ = model(data, None)
    return prediction.squeeze(1).numpy()

def main(args):
    model = load_model(args.model)
    data = pd.read_csv(args.data).values

    if args.batch:
        dataset = TrafficDataset(args.data, seq_len=args.seq_len, pred_len=args.pred_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_predictions = []
        for x, _ in dataloader:
            predictions = predict(model, x)
            all_predictions.extend(predictions)

        pd.DataFrame(all_predictions).to_csv(args.output, index=False)
    else:
        window = data[:args.seq_len] 
        predictions = []

        for i in range(args.seq_len, len(data)):
            pred = predict(model, window)
            predictions.append(pred[-1])
            window = np.append(window[1:], data[i]).reshape(-1, 1)

        pd.DataFrame(predictions).to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the trained student model')
    parser.add_argument('--data', type=str, required=True, help='Path to the traffic data CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length of the input data')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--batch', action='store_true', help='Enable batch prediction mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch prediction')
    args = parser.parse_args()

    main(args)
