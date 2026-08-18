# %%
import os, glob, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig

ART = os.path.dirname(glob.glob("/kaggle/input/**/student_int8.pt", recursive=True)[0])
DATA = os.path.dirname(glob.glob("/kaggle/input/**/test.csv", recursive=True)[0])
MAX_LEN = 256
BATCH = 32
LETTERS = list("ABCDE")
device = torch.device("cuda")

MODEL_PATH = f"{ART}/student_int8.pt"
print(ART, DATA)

# %%
blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
remap = blob["remap"]

cfg = AutoConfig.from_pretrained(f"{ART}/model_config")
model = AutoModelForMultipleChoice.from_config(cfg)
emb = model.deberta.embeddings.word_embeddings
model.deberta.embeddings.word_embeddings = nn.Embedding(cfg.vocab_size, emb.embedding_dim)

state = {}
for k, v in blob["q"].items():
    state[k] = (v.float() * blob["scales"][k].float()).half() if v.dtype == torch.int8 else v.half()

missing, unexpected = model.load_state_dict(state, strict=False)
assert not unexpected, unexpected[:5]
assert not [k for k in missing if "position_ids" not in k], missing[:5]

model = model.half().to(device).eval()
tok = AutoTokenizer.from_pretrained(f"{ART}/tokenizer")

# %%
test_df = pd.read_csv(f"{DATA}/test.csv")

encoded = [tok([str(r["prompt"])] * 5, [str(r[l]) for l in LETTERS],
               truncation=True, max_length=MAX_LEN)["input_ids"]
           for _, r in test_df.iterrows()]

pad_to = max(len(s) for q in encoded for s in q)
pad_id = tok.pad_token_id

ids, mask = [], []
for q in encoded:
    ids.append(torch.from_numpy(remap[np.array([s + [pad_id] * (pad_to - len(s)) for s in q])]))
    mask.append(torch.tensor([[1] * len(s) + [0] * (pad_to - len(s)) for s in q]))
ids, mask = torch.stack(ids), torch.stack(mask)
print(f"{len(test_df)} questions, padded to {pad_to} tokens")

# %%
with torch.inference_mode():
    for _ in range(3):
        model(input_ids=ids[:2].to(device), attention_mask=mask[:2].to(device))
    torch.cuda.synchronize()

    start = time.time()
    logits = []
    for i in range(0, len(ids), BATCH):
        logits.append(model(input_ids=ids[i:i + BATCH].to(device),
                            attention_mask=mask[i:i + BATCH].to(device)).logits.float().cpu())
    torch.cuda.synchronize()
    elapsed = time.time() - start

logits = torch.cat(logits).numpy()
top3 = np.argsort(-logits, axis=1)[:, :3]
pd.DataFrame({"id": test_df["id"],
              "prediction": [" ".join(LETTERS[j] for j in row) for row in top3]}
             ).to_csv("submission.csv", index=False)

size_mb = os.path.getsize(MODEL_PATH) / 1e6
latency = elapsed / len(test_df)
print(f"Model Size: {size_mb:.2f}")
print(f"Average Latency: {latency:.6f} seconds per sample")
print(f"size x{min(1.0, 150/size_mb):.3f}  latency x{min(1.0, 0.030/latency):.3f}")
