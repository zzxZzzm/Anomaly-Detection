import torch
import torch.optim as optim
from carn import StudentNetwork, TeacherNetwork
from traffic_dataset import get_dataloader
from distillation_loss import distillation_loss, reconstruction_loss, estimator_loss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the traffic data CSV file')
args = parser.parse_args()

dataloader = get_dataloader(args.data)

teacher_model = TeacherNetwork()
teacher_model.load_state_dict(torch.load('teacher_model.pth'))
teacher_model.eval()

student_model = StudentNetwork()
criterion_recon = reconstruction_loss
criterion_distill = distillation_loss
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

for epoch in range(args.epochs):
    for x, y in dataloader:
        x = x.unsqueeze(1).float()  
        y = y.unsqueeze(1).float()
        
        with torch.no_grad():
            teacher_output, teacher_features = teacher_model(x)
        
        student_output, student_features, mu, b = student_model(x, teacher_features)
        
        loss_recon = criterion_recon(student_output, y)
        loss_distill = criterion_distill((teacher_output, teacher_features), (student_output, student_features))
        loss_estimator = estimator_loss(mu, b, teacher_features, student_features)
        loss = loss_recon + loss_distill + loss_estimator
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(student_model.state_dict(), 'student_model.pth')
