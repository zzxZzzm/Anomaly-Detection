import torch

def imitation_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

def reconstruction_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)

def distillation_loss(teacher_outputs, student_outputs):
    teacher_features = teacher_outputs[1]
    student_features = student_outputs[1]
    
    loss = 0
    for t, s in zip(teacher_features, student_features):
        loss += torch.mean((t - s) ** 2)
    
    return loss

def estimator_loss(mu, b, teacher_features, student_features):
    loss = 0
    for t, s in zip(teacher_features, student_features):
        f_t = t.detach()
        mu_k = mu.unsqueeze(-1)
        b_k = b.unsqueeze(-1)
        loss += torch.sum(torch.log(b_k) + (f_t - mu_k) ** 2 / (2 * b_k ** 2))
    return loss
