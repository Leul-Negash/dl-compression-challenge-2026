# %%
import os, gc, glob, random
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import prune
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_cosine_schedule_with_warmup

TEACHER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
STUDENT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_LEN = 256
FOLDS = 5
T_EPOCHS = 4
S_EPOCHS = 20
RECOVER_EPOCHS = 5
PRUNE_AMOUNT = 0.25
KD_T = 3.0
KD_ALPHA = 0.7
SEED = 42
DATA = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
AUG = glob.glob("/kaggle/input/**/augment.csv", recursive=True)
OUT = "/kaggle/working"
LETTERS = list("ABCDE")

device = torch.device("cuda")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

train_df = pd.read_csv(f"{DATA}/train.csv")
test_df = pd.read_csv(f"{DATA}/test.csv")
train_df["label"] = train_df["answer"].map({l: i for i, l in enumerate(LETTERS)})
y_true = train_df["label"].values
N = len(train_df)

if AUG:
    aug_df = pd.read_csv(AUG[0])
    aug_df["label"] = aug_df["answer"].map({l: i for i, l in enumerate(LETTERS)})
    pool_df = pd.concat([train_df, aug_df], ignore_index=True)
else:
    pool_df = train_df.copy()

extra = np.arange(N, len(pool_df))
print(DATA)
print(f"{N} provided + {len(extra)} written = {len(pool_df)} training rows, {len(test_df)} test")

# %%
class MCQData(Dataset):
    def __init__(self, df, tok, labelled=True):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.labelled = labelled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok([str(r["prompt"])] * 5, [str(r[l]) for l in LETTERS],
                       truncation=True, max_length=MAX_LEN, padding="max_length",
                       return_tensors="pt")
        item = dict(enc)
        if self.labelled:
            item["labels"] = torch.tensor(int(r["label"]))
        return item


def map3(logits, labels):
    top = np.argsort(-logits, axis=1)[:, :3]
    total = 0.0
    for row, y in zip(top, labels):
        for rank, p in enumerate(row):
            if p == y:
                total += 1.0 / (rank + 1)
                break
    return total / len(labels)


@torch.no_grad()
def predict(model, loader, with_labels=False):
    model.eval()
    out, lab = [], []
    for batch in loader:
        y = batch.pop("labels", None)
        batch.pop("idx", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda'):
            out.append(model(**batch).logits.float().cpu().numpy())
        if y is not None:
            lab.append(y.numpy())
    logits = np.concatenate(out)
    return (logits, np.concatenate(lab)) if with_labels else logits


def fit(model, loader, epochs, lr, soft=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    for ep in range(epochs):
        running = 0.0
        for batch in loader:
            y = batch.pop("labels").to(device)
            idx = batch.pop("idx", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(**batch).logits
                loss = F.cross_entropy(logits, y)
                if soft is not None:
                    kd = F.kl_div(F.log_softmax(logits / KD_T, -1),
                                  F.softmax(soft[idx].to(device) / KD_T, -1),
                                  reduction="batchmean") * KD_T ** 2
                    loss = KD_ALPHA * kd + (1 - KD_ALPHA) * loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
        print(f"  ep {ep+1}/{epochs}  loss {running/len(loader):.4f}")
    return model

# %%
t_tok = AutoTokenizer.from_pretrained(TEACHER)
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(train_df, y_true))

oof = np.zeros((N, 5), dtype=np.float32)
fold_full = np.zeros((FOLDS, len(pool_df), 5), dtype=np.float32)
all_loader = DataLoader(MCQData(pool_df, t_tok), batch_size=4)

for f, (tr, va) in enumerate(folds):
    print(f"fold {f+1}")
    m = AutoModelForMultipleChoice.from_pretrained(TEACHER, ignore_mismatched_sizes=True).float().to(device)
    m.gradient_checkpointing_enable()
    sub = np.concatenate([tr, extra])
    fit(m, DataLoader(MCQData(pool_df.iloc[sub], t_tok), batch_size=2, shuffle=True), T_EPOCHS, 6e-6)
    full = predict(m, all_loader)
    fold_full[f] = full
    oof[va] = full[va]
    del m; gc.collect(); torch.cuda.empty_cache()

soft_logits = fold_full.mean(0)

print(f"teacher OOF MAP@3 = {map3(oof, y_true):.4f}")
np.save(f"{OUT}/oof.npy", oof)
np.save(f"{OUT}/soft_logits.npy", soft_logits)

# %%
s_tok = AutoTokenizer.from_pretrained(STUDENT)

# the 128k embedding table is over half the model; keep only ids this dataset can produce
keep = set(s_tok.all_special_ids)
for df in (pool_df, test_df):
    for _, r in df.iterrows():
        for l in LETTERS:
            keep.update(s_tok(str(r["prompt"]), str(r[l]), truncation=True, max_length=MAX_LEN)["input_ids"])
keep = sorted(keep)

remap = np.zeros(s_tok.vocab_size + 10, dtype=np.int64)
for new, old in enumerate(keep):
    remap[old] = new
print(f"vocab {s_tok.vocab_size} -> {len(keep)}")


def build_student():
    m = AutoModelForMultipleChoice.from_pretrained(STUDENT, ignore_mismatched_sizes=True).float()
    emb = m.deberta.embeddings.word_embeddings
    trimmed = nn.Embedding(len(keep), emb.embedding_dim)
    trimmed.weight.data = emb.weight.data[keep].clone()
    m.deberta.embeddings.word_embeddings = trimmed
    m.config.vocab_size = len(keep)
    return m.to(device)


class RemappedData(MCQData):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["input_ids"] = torch.from_numpy(remap[item["input_ids"].numpy()])
        item["idx"] = torch.tensor(i)
        return item


print(f"student {sum(p.numel() for p in build_student().parameters())/1e6:.1f}M params")

# %%
# fold f's targets come from teacher f, which never trained on va_f
s_oof = np.zeros((N, 5), dtype=np.float32)

for f, (tr, va) in enumerate(folds):
    print(f"student fold {f+1}")
    m = build_student()
    keep_idx = np.concatenate([tr, extra])
    sub = pool_df.iloc[keep_idx].reset_index(drop=True)
    idx_map = torch.tensor(keep_idx)

    class FoldData(RemappedData):
        def __getitem__(self, i):
            item = super().__getitem__(i)
            item["idx"] = idx_map[i]
            return item

    fit(m, DataLoader(FoldData(sub, s_tok), batch_size=4, shuffle=True), S_EPOCHS, 2e-5,
        soft=torch.tensor(fold_full[f]))
    s_oof[va] = predict(m, DataLoader(RemappedData(pool_df.iloc[va], s_tok), batch_size=8))
    del m; gc.collect(); torch.cuda.empty_cache()

print(f"student OOF MAP@3 = {map3(s_oof, y_true):.4f}")

# %%
soft = torch.tensor(soft_logits)
train_loader = DataLoader(RemappedData(pool_df, s_tok), batch_size=4, shuffle=True)
eval_loader = DataLoader(RemappedData(train_df, s_tok), batch_size=8)

student = build_student()
fit(student, train_loader, S_EPOCHS, 2e-5, soft=soft)
print(f"student fit MAP@3 = {map3(*predict(student, eval_loader, with_labels=True)):.4f}")

# %%
linears = [(m, "weight") for m in student.modules() if isinstance(m, nn.Linear)]
prune.global_unstructured(linears, pruning_method=prune.L1Unstructured, amount=PRUNE_AMOUNT)

fit(student, train_loader, RECOVER_EPOCHS, 8e-6, soft=soft)
for m, name in linears:
    prune.remove(m, name)

zeros = sum(int((p == 0).sum()) for p in student.parameters())
total = sum(p.numel() for p in student.parameters())
lin_zeros = sum(int((m.weight == 0).sum()) for m, _ in linears)
lin_total = sum(m.weight.numel() for m, _ in linears)
print(f"sparsity: linear {100*lin_zeros/lin_total:.1f}%  whole model {100*zeros/total:.1f}%")
print(f"student MAP@3 after pruning = {map3(*predict(student, eval_loader, with_labels=True)):.4f}")

# %%
student = student.cpu().eval()
packed, scales = {}, {}
for k, v in student.state_dict().items():
    if v.dtype.is_floating_point and v.dim() == 2 and min(v.shape) > 1:
        s = v.abs().amax(dim=1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        packed[k] = torch.round(v / s).clamp(-127, 127).to(torch.int8)
        scales[k] = s.to(torch.float16)
    else:
        packed[k] = v.to(torch.float16)

q_zeros = sum(int((v == 0).sum()) for v in packed.values())
q_total = sum(v.numel() for v in packed.values())
print(f"sparsity after quantization = {100*q_zeros/q_total:.1f}%")

torch.save({"q": packed, "scales": scales, "remap": remap}, f"{OUT}/student_int8.pt")
s_tok.save_pretrained(f"{OUT}/tokenizer")
student.config.save_pretrained(f"{OUT}/model_config")
print(f"artifact {os.path.getsize(f'{OUT}/student_int8.pt')/1e6:.2f} MB")
