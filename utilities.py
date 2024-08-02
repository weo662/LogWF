import numpy as np
import torch
from src.configuration import device

def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y - torch.mean(data)
    if not cov:
        cov = torch.cov(data.T)
    inv_covmat = torch.linalg.inv(cov)
    left = torch.dot(y_mu, inv_covmat)
    mahal = torch.dot(left, y_mu.T)
    return mahal.diagonal()

def init_center_c(input_dim, train_loader, model, eps=0.1):
    n_samples = 0
    c = torch.zeros(input_dim, device=device)

    model.eval()
    with torch.no_grad():
        for data in train_loader:

            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(torch.sigmoid(outputs.squeeze()), dim=0)

    c /= n_samples


    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f"Initialize c successfully")

    return c

def get_radius(dist, nu):

    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

