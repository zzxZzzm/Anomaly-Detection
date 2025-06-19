import torch
import torch.optim as optim
from carn import TeacherNetwork
from traffic_dataset import get_dataloader
from distillation_loss import imitation_loss, reconstruction_loss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the traffic data CSV file')
args = parser.parse_args()

dataloader = get_dataloader(args.data)

model = TeacherNetwork()
criterion_recon = reconstruction_loss
criterion_imit = imitation_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(args.epochs):
    for x, y in dataloader:
        x = x.unsqueeze(1).float()
        y = y.unsqueeze(1).float()
        
        output, features = model(x)
        loss_recon = criterion_recon(output, y)
        loss_imit = criterion_imit(features[-1], x) 
        loss = loss_recon + loss_imit
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'teacher_model.pth')
