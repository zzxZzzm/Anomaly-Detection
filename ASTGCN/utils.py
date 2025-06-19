import numpy as np
import torch

def scaled_laplacian(W):
    n = W.shape[0]
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.power(d + 1e-8, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt.dot(W).dot(D_inv_sqrt)
    return L

def cheb_poly(L, K):
    N = L.shape[0]
    laplacian = 2 * L - np.eye(N)
    
    if K == 0:
        return np.eye(N).reshape(N, N, 1)
    elif K == 1:
        return np.stack([np.eye(N), laplacian], axis=-1)
    else:
        Lk = np.zeros((N, N, K))
        Lk[:, :, 0] = np.eye(N)
        Lk[:, :, 1] = laplacian
        for i in range(2, K):
            Lk[:, :, i] = 2 * np.dot(laplacian, Lk[:, :, i-1]) - Lk[:, :, i-2]
        return Lk

def evaluate_model(model, loss_fn, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss_fn(y_pred, y, 0.5)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    model.train()
    return l_sum / n
