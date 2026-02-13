import torch
import pandas as pd
from model import SiameseResNet50

def load_model(path, device):
    model = SiameseResNet50().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
