# ==================== 1. Imports ====================
import os, gc, random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image

# ==================== 2. Config ====================
CONFIG = {
    "train_pkl": "train.pkl",
    "epochs": 50,
    "batch_size": 64,
    "resize": (96, 96),
    "random_seed": 42,
    "val_split_ratio": 0.2,
    "num_workers": 2,
    "pin_memory": True,
    "lr_backbone": 1e-4,
    "lr_classifier": 2e-4,
    "weight_decay": 1e-4,
    "scheduler_T_0": 10,
    "scheduler_T_mult": 2,
    "scheduler_eta_min": 1e-6,
    "dropout_1": 0.3,
    "dropout_2": 0.15,
}

# ==================== 3. Seed ====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 4. Dataset ====================
class PairedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = Image.open(row["img1"]).convert("L")
        img2 = Image.open(row["img2"]).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return img1, img2, label

# ==================== 5. Model ====================
class SiameseResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_1"]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_2"]),
            nn.Linear(256, 1)
        )

    def forward_once(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        return self.classifier(torch.cat([f1, f2], dim=1))

# ==================== 6. Training ====================
def main():
    set_seed(CONFIG["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_pickle(CONFIG["train_pkl"])
    df = df.sample(frac=1).reset_index(drop=True)

    split_idx = int(len(df) * (1 - CONFIG["val_split_ratio"]))
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    transform = transforms.Compose([
        transforms.Resize(CONFIG["resize"]),
        transforms.ToTensor(),
    ])

    train_ds = PairedDataset(train_df, transform)
    val_ds = PairedDataset(val_df, transform)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = SiameseResNet50().to(device)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": CONFIG["lr_backbone"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["lr_classifier"]}
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["scheduler_T_0"],
        T_mult=CONFIG["scheduler_T_mult"],
        eta_min=CONFIG["scheduler_eta_min"]
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                output = model(img1, img2).squeeze()
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        print(f"Epoch {epoch+1} completed")

    torch.save(model.state_dict(), "siamese_resnet50.pth")

if __name__ == "__main__":
    main()
