# train_model_v5_strong_resume.py

import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

# ========================================================================
# PATHS (COLAB FILE STRUCTURE)
# ========================================================================

BASE_DIR = "/content/colab_files"          # ← TÜM DOSYALARIN BURADA

SEQ_DIR = f"{BASE_DIR}/sequences_v5"       # veri klasörü
MODEL_DIR = f"{BASE_DIR}/models_v5"        # model kayıt klasörü
os.makedirs(MODEL_DIR, exist_ok=True)

X_PATH = os.path.join(SEQ_DIR, "X.npy")
Y_PATH = os.path.join(SEQ_DIR, "y.npy")

STATE_FILE = os.path.join(MODEL_DIR, "training_state.json")
CHECKPOINT_LAST = os.path.join(MODEL_DIR, "checkpoint_last.pth")
CHECKPOINT_BEST = os.path.join(MODEL_DIR, "best.pth")


# ========================================================================
# HYPERPARAMS
# ========================================================================

SEQ_LEN = 64
BATCH_SIZE = 128
EPOCHS = 18
LR = 3e-4
WARMUP_EPOCHS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 128
DROPOUT = 0.1

# ========================================================================
# DATASET
# ========================================================================

class SeqDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(int(self.y[idx]), dtype=torch.float32)

# ========================================================================
# MODEL
# ========================================================================

class TransformerStrong(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dim_feedforward=DIM_FF, dropout=DROPOUT,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len]
        x = self.norm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ========================================================================
# LR SCHEDULER
# ========================================================================

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ========================================================================
# TRAIN + RESUME
# ========================================================================

def train():

    # ----------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------
    X_sample = np.load(X_PATH, mmap_mode="r")
    INPUT_DIM = X_sample.shape[2]
    print("📥 INPUT DIM:", INPUT_DIM)

    ds = SeqDataset(X_PATH, Y_PATH)
    n = len(ds)
    print("📊 TOTAL SEQUENCES:", n)

    # ----------------------------------------------------
    # SPLIT FIX – ALWAYS SAME ORDER
    # ----------------------------------------------------
    if os.path.exists(STATE_FILE):
        print("🔁 Önceki eğitim bulundu → Shuffle tekrar yükleniyor.")
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        indices = np.array(state["indices"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
    else:
        print("🔥 Yeni eğitim başlatılıyor.")
        indices = np.arange(n)
        np.random.shuffle(indices)
        start_epoch = 1
        best_f1 = 0.0

    idx_train = int(n * 0.80)
    idx_val = int(n * 0.90)

    train_loader = DataLoader(Subset(ds, indices[:idx_train]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, indices[idx_train:idx_val]), batch_size=256, num_workers=0)
    test_loader = DataLoader(Subset(ds, indices[idx_val:]), batch_size=256, num_workers=0)

    # ----------------------------------------------------
    # MODEL + OPTIMIZER
    # ----------------------------------------------------
    model = TransformerStrong(INPUT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ----------------------------------------------------
    # RESUME CHECKPOINT
    # ----------------------------------------------------
    if os.path.exists(CHECKPOINT_LAST):
        print("🔄 Checkpoint bulundu → model yükleniyor.")
        ckpt = torch.load(CHECKPOINT_LAST, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # ----------------------------------------------------
    # CLASS IMBALANCE
    # ----------------------------------------------------
    y_all = np.load(Y_PATH)
    pos = (y_all == 1).sum()
    neg = (y_all == 0).sum()
    pos_weight = torch.tensor([neg / (pos + 1e-9)], dtype=torch.float32).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ====================================================================
    # TRAIN LOOP (AUTO-RESUME)
    # ====================================================================

    for epoch in range(start_epoch, EPOCHS + 1):

        t0 = time.time()
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb).cpu().numpy()
                p = (1 / (1 + np.exp(-logits))) > 0.5
                preds.extend(p.astype(int))
                labels.extend(yb.numpy().astype(int))

        val_acc = accuracy_score(labels, preds)
        val_f1 = f1_score(labels, preds)

        print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f} | time={time.time() - t0:.1f}s")

        # ----------------------------------------------------
        # SAVE LAST CHECKPOINT
        # ----------------------------------------------------
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, CHECKPOINT_LAST)

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_f1": val_f1
            }, CHECKPOINT_BEST)
            print("✔ En iyi model güncellendi.")

        # ----------------------------------------------------
        # SAVE STATE (for resume)
        # ----------------------------------------------------
        with open(STATE_FILE, "w") as f:
            json.dump({
                "epoch": epoch,
                "best_f1": best_f1,
                "indices": indices.tolist()
            }, f)

    # ====================================================================
    # FINAL TEST
    # ====================================================================

    print("\n🔥 Eğitim tamamlandı → En iyi model yükleniyor.")
    best = torch.load(CHECKPOINT_BEST, map_location=DEVICE)
    model.load_state_dict(best["model"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb).cpu().numpy()
            p = (1 / (1 + np.exp(-logits))) > 0.5
            preds.extend(p.astype(int))
            labels.extend(yb.numpy().astype(int))

    print("\n📌 FINAL TEST RESULTS")
    print("ACC:", accuracy_score(labels, preds))
    print("F1 :", f1_score(labels, preds))


if __name__ == "__main__":
    train()
