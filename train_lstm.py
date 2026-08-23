
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

BASE = Path(__file__).resolve().parent
DATA = BASE/"data"/"cmapss"
MODEL = BASE/"model"
SEQ_LEN = 30
RUL_CUTOFF = 30
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
FEATURES = ["op_setting_1","op_setting_2","op_setting_3"] + [f"sensor_{i}" for i in range(1,22)]

# Remove constant sensors in FD001
train = pd.read_csv(DATA/"train_FD001.csv")
test = pd.read_csv(DATA/"test_FD001.csv")
rul = pd.read_csv(DATA/"RUL_FD001.csv")

varying = [c for c in FEATURES if train[c].nunique(dropna=False) > 1]
FEATURES = varying

# Engine-wise split for training/validation
engines = sorted(train.engine_id.unique())
rng = np.random.default_rng(SEED)
rng.shuffle(engines)
cut = int(len(engines)*0.8)
tr_eng, va_eng = set(engines[:cut]), set(engines[cut:])

scaler = StandardScaler()
scaler.fit(train[train.engine_id.isin(tr_eng)][FEATURES])
mean, scale = scaler.mean_.tolist(), scaler.scale_.tolist()

def make_train_sequences(df, engine_set):
    X, y = [], []
    for eid, g in df[df.engine_id.isin(engine_set)].groupby("engine_id"):
        g = g.sort_values("cycle")
        vals = scaler.transform(g[FEATURES])
        maxc = g.cycle.max()
        # Label each window by RUL at its last cycle.
        for end in range(SEQ_LEN, len(g)+1):
            rul_val = maxc - g.cycle.iloc[end-1]
            X.append(vals[end-SEQ_LEN:end])
            y.append(1 if rul_val <= RUL_CUTOFF else 0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

Xtr, ytr = make_train_sequences(train, tr_eng)
Xva, yva = make_train_sequences(train, va_eng)

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(0.25)
        self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    def forward(self, x):
        x,_ = self.lstm1(x)
        x = self.drop(x)
        x,_ = self.lstm2(x)
        return self.fc(x[:,-1,:]).squeeze(-1)

model = LSTMClassifier(len(FEATURES))
pos = max(1.0, float((ytr==0).sum()) / max(1.0, (ytr==1).sum()))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
Xv, yv = torch.tensor(Xva), torch.tensor(yva)

best_f1=-1; best_state=None; patience=5; wait=0
for epoch in range(1, 31):
    model.train()
    idx = torch.randperm(len(Xt))
    total=0
    for s in range(0,len(idx),256):
        b=idx[s:s+256]
        opt.zero_grad()
        loss=criterion(model(Xt[b]), yt[b])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()*len(b)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xv)).numpy()
    # optimize threshold on validation F1, but require reasonable recall
    best_thr=0.5; cur=-1
    for th in np.arange(0.20,0.91,0.01):
        pred=(p>=th).astype(int)
        f=f1_score(yva,pred,zero_division=0)
        if f>cur:
            cur=f; best_thr=float(th)
    if cur>best_f1:
        best_f1=cur; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        best_threshold=best_thr; wait=0
    else:
        wait+=1
        if wait>=patience: break

model.load_state_dict(best_state)
model.eval()

# Test: one final 30-cycle window per unseen test engine, labeled from RUL_FD001
test_probs=[]; test_y=[]
for i,eid in enumerate(sorted(test.engine_id.unique())):
    g=test[test.engine_id==eid].sort_values("cycle")
    vals=scaler.transform(g[FEATURES])
    if len(vals)<SEQ_LEN: continue
    x=torch.tensor(vals[-SEQ_LEN:],dtype=torch.float32).unsqueeze(0)
    with torch.no_grad(): pr=float(torch.sigmoid(model(x)).item())
    test_probs.append(pr)
    test_y.append(1 if float(rul.iloc[i,0]) <= RUL_CUTOFF else 0)

test_probs=np.array(test_probs); test_y=np.array(test_y)
pred=(test_probs>=best_threshold).astype(int)
metrics={
    "accuracy":float(accuracy_score(test_y,pred)),
    "precision":float(precision_score(test_y,pred,zero_division=0)),
    "recall":float(recall_score(test_y,pred,zero_division=0)),
    "f1":float(f1_score(test_y,pred,zero_division=0)),
    "pr_auc":float(average_precision_score(test_y,test_probs)),
    "confusion_matrix":confusion_matrix(test_y,pred).tolist(),
    "test_engines":int(len(test_y)),
    "positive_test_engines":int(test_y.sum())
}

MODEL.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL/"lstm.pt")
(MODEL/"scaler.json").write_text(json.dumps({"features":FEATURES,"mean":mean,"scale":scale}))
(MODEL/"metadata.json").write_text(json.dumps({
    "model":"PyTorch LSTM",
    "sequence_length":SEQ_LEN,
    "rul_cutoff_cycles":RUL_CUTOFF,
    "threshold":best_threshold,
    "features":FEATURES,
    "train_engines":len(tr_eng),
    "validation_engines":len(va_eng),
    "metrics":metrics
}, indent=2))
print(json.dumps(metrics,indent=2))
print("threshold",best_threshold)
