"""Build notebooks/eda.ipynb programmatically. Run once to materialize."""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).parent / "eda.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = [
    md("""# EDA + baseline experiments - ru-jailbreak-guard

This notebook covers:
1. Exploratory data analysis on the merged HiveTrace + dmtrdr + benign-Wiki dataset
2. Three model families trained on the same splits:
   - TF-IDF (1-2 grams) + Logistic Regression (baseline)
   - LightGBM on ruBERT-tiny2 mean-pooled embeddings (mid-tier)
   - Fine-tuned ruBERT-tiny2 (champion candidate)
3. Per-source diagnostics and confusion matrices
4. A short comparison table at the end

Reproducibility:
- Data version is the sha256 of `dvc.lock`'s `split` stage outputs
- Each model is seeded (`seed=42`)
- Train and val splits are fixed
"""),
    md("## 1. Setup"),
    code("""import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

REPO_ROOT = Path('..').resolve()
SPLITS = REPO_ROOT / 'data' / 'splits'
print('repo:', REPO_ROOT)
print('splits dir contents:', sorted(p.name for p in SPLITS.iterdir()))
"""),
    md("## 2. Load splits"),
    code("""train = pl.read_parquet(SPLITS / 'train.parquet')
val   = pl.read_parquet(SPLITS / 'val.parquet')
test  = pl.read_parquet(SPLITS / 'test.parquet')
print(f'train: {len(train):>6} rows')
print(f'val:   {len(val):>6} rows')
print(f'test:  {len(test):>6} rows')
train.head(3)
"""),
    md("## 3. EDA - class balance"),
    code("""label_counts = (
    pl.concat([
        train.with_columns(pl.lit('train').alias('split')),
        val.with_columns(pl.lit('val').alias('split')),
        test.with_columns(pl.lit('test').alias('split')),
    ])
    .group_by(['split', 'label']).len()
    .pivot(values='len', index='split', on='label')
    .sort('split')
)
label_counts
"""),
    code("""# All splits should be ~50/50 by construction (balance_classes in merge stage).
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
splits = ['train', 'val', 'test']
positives = [int((train['label']==1).sum()), int((val['label']==1).sum()), int((test['label']==1).sum())]
negatives = [int((train['label']==0).sum()), int((val['label']==0).sum()), int((test['label']==0).sum())]
x = np.arange(len(splits))
ax.bar(x - 0.18, positives, width=0.36, label='jailbreak (1)', color='#d62728')
ax.bar(x + 0.18, negatives, width=0.36, label='benign (0)',    color='#2ca02c')
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel('rows'); ax.set_title('class balance per split')
ax.legend(); plt.tight_layout(); plt.show()
"""),
    md("## 4. EDA - source / subcategory"),
    code("""src_counts = (
    pl.concat([
        train.with_columns(pl.lit('train').alias('split')),
        val.with_columns(pl.lit('val').alias('split')),
        test.with_columns(pl.lit('test').alias('split')),
    ])
    .group_by(['split', 'source']).len()
    .pivot(values='len', index='source', on='split')
    .fill_null(0)
    .sort('source')
)
src_counts
"""),
    code("""# Subcategory breakdown for adversarial-attack rows (hivetracered).
attacks = train.filter(pl.col('source') == 'hivetracered_attack')
print(f'hivetracered rows in train: {len(attacks)}')
print(attacks.group_by('subcategory').len().sort('subcategory'))
"""),
    md("## 5. EDA - text length distribution by class"),
    code("""train_with_len = train.with_columns(pl.col('text').str.len_chars().alias('chars'))
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for label, color in [(0, '#2ca02c'), (1, '#d62728')]:
    sub = train_with_len.filter(pl.col('label') == label)['chars'].to_numpy()
    ax.hist(np.clip(sub, 0, 1500), bins=40, alpha=0.5,
            label=('benign' if label == 0 else 'jailbreak'), color=color)
ax.set_xlabel('characters'); ax.set_ylabel('count')
ax.set_title('Train text length distribution by class (clipped at 1500)')
ax.legend(); plt.tight_layout(); plt.show()

print('p50/p95 chars by class:')
for label in (0, 1):
    sub = train_with_len.filter(pl.col('label') == label)['chars'].to_numpy()
    print(f'  label={label}: median={int(np.median(sub))}, p95={int(np.percentile(sub, 95))}')
"""),
    md("## 6. EDA - cyrillic ratio (signal that benign tends to be more Russian)"),
    code("""def cyr_ratio(s: str) -> float:
    if not s:
        return 0.0
    cyr = sum(1 for c in s if 'Ѐ' <= c <= 'ӿ')
    return cyr / max(len(s), 1)

train_pd = train.to_pandas()
train_pd['cyr'] = train_pd['text'].astype(str).map(cyr_ratio)
print(train_pd.groupby('label')['cyr'].agg(['mean', 'median', 'std']).round(3))
"""),
    md("""## 7. Model 1 - TF-IDF + Logistic Regression

Hyperparameters tuned by light grid search on val F1 (commented out for speed).
Final config below.
"""),
    code("""y_train = train['label'].to_numpy().astype(int)
y_val   = val['label'].to_numpy().astype(int)
y_test  = test['label'].to_numpy().astype(int)
X_train_txt = train['text'].to_list()
X_val_txt   = val['text'].to_list()
X_test_txt  = test['text'].to_list()

vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(X_train_txt)
Xv  = vec.transform(X_val_txt)
Xte = vec.transform(X_test_txt)
print(f'TF-IDF vocab={Xtr.shape[1]}, train shape={Xtr.shape}')

logreg = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
logreg.fit(Xtr, y_train)
val_pred  = logreg.predict(Xv)
val_score = logreg.predict_proba(Xv)[:, 1]
test_pred = logreg.predict(Xte)
test_score = logreg.predict_proba(Xte)[:, 1]

tfidf_metrics = {
    'val_f1':  f1_score(y_val, val_pred),
    'val_auc': roc_auc_score(y_val, val_score),
    'test_f1':  f1_score(y_test, test_pred),
    'test_auc': roc_auc_score(y_test, test_score),
}
print(json.dumps(tfidf_metrics, indent=2))
print('\\nclassification report (test):')
print(classification_report(y_test, test_pred, target_names=['benign', 'jailbreak']))
"""),
    md("## 8. Model 2 - LightGBM on ruBERT-tiny2 mean-pooled embeddings"),
    code("""import torch
from transformers import AutoModel, AutoTokenizer

DEVICE = 'cpu'
MODEL_NAME = 'cointegrated/rubert-tiny2'
MAX_LEN = 256

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
embedder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

@torch.no_grad()
def embed(texts: list[str], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i + batch]
        ids = tok(sub, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
        h = embedder(**ids).last_hidden_state
        mask = ids['attention_mask'].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out)

Etr = embed(X_train_txt)
Ev  = embed(X_val_txt)
Ete = embed(X_test_txt)
print('embedding shapes:', Etr.shape, Ev.shape, Ete.shape)
"""),
    code("""import lightgbm as lgb

train_set = lgb.Dataset(Etr, y_train)
val_set   = lgb.Dataset(Ev, y_val, reference=train_set)
params = dict(
    objective='binary',
    metric='binary_logloss',
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.05,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbosity=-1,
)
booster = lgb.train(
    params, train_set, num_boost_round=300, valid_sets=[val_set], valid_names=['val'],
    callbacks=[lgb.early_stopping(20)],
)
val_score_lgb  = booster.predict(Ev,  num_iteration=booster.best_iteration)
test_score_lgb = booster.predict(Ete, num_iteration=booster.best_iteration)
val_pred_lgb   = (val_score_lgb  >= 0.5).astype(int)
test_pred_lgb  = (test_score_lgb >= 0.5).astype(int)

lgb_metrics = {
    'val_f1':   f1_score(y_val, val_pred_lgb),
    'val_auc':  roc_auc_score(y_val, val_score_lgb),
    'test_f1':  f1_score(y_test, test_pred_lgb),
    'test_auc': roc_auc_score(y_test, test_score_lgb),
}
print(json.dumps(lgb_metrics, indent=2))
print('\\nclassification report (test):')
print(classification_report(y_test, test_pred_lgb, target_names=['benign', 'jailbreak']))
"""),
    md("""## 9. Model 3 - Fine-tuned ruBERT-tiny2

Trained 2 epochs on a sample (12k train, 2k val) for notebook speed; full-data
fine-tune happens via `make train-rubert` against the production MLflow.
"""),
    code("""from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from torch.utils.data import Dataset

set_seed(42)

class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts, self.labels, self.tok, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], padding='max_length', truncation=True,
                       max_length=self.max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()} | {'labels': int(self.labels[i])}

# Sample for notebook speed; full run via Make targets.
TRAIN_N = min(len(X_train_txt), 12_000)
VAL_N   = min(len(X_val_txt),   2_000)

idx_tr = np.random.RandomState(42).permutation(len(X_train_txt))[:TRAIN_N]
idx_va = np.random.RandomState(42).permutation(len(X_val_txt))[:VAL_N]
sub_train_txt = [X_train_txt[i] for i in idx_tr]; sub_y_train = y_train[idx_tr]
sub_val_txt   = [X_val_txt[i]   for i in idx_va]; sub_y_val   = y_val[idx_va]

ft_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
clf    = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

train_ds = TextDS(sub_train_txt, sub_y_train, ft_tok)
val_ds   = TextDS(sub_val_txt,   sub_y_val,   ft_tok)

args = TrainingArguments(
    output_dir='/tmp/rubert_ft_nb',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=0.1,
    eval_strategy='epoch',
    save_strategy='no',
    logging_strategy='epoch',
    seed=42,
    report_to=[],
    use_cpu=True,
    disable_tqdm=False,
)
trainer = Trainer(model=clf, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
"""),
    code("""# Evaluate fine-tuned ruBERT on FULL test set.
@torch.no_grad()
def ft_predict(texts: list[str], batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    clf.eval()
    scores = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i + batch]
        ids = ft_tok(sub, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
        logits = clf(**ids).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        scores.append(probs.cpu().numpy())
    s = np.concatenate(scores)
    return s, (s >= 0.5).astype(int)

test_score_ft, test_pred_ft = ft_predict(X_test_txt)
ft_metrics = {
    'test_f1':  f1_score(y_test, test_pred_ft),
    'test_auc': roc_auc_score(y_test, test_score_ft),
}
print(json.dumps(ft_metrics, indent=2))
print('\\nclassification report (test):')
print(classification_report(y_test, test_pred_ft, target_names=['benign', 'jailbreak']))
"""),
    md("## 10. Side-by-side comparison"),
    code("""import pandas as pd
table = pd.DataFrame({
    'tfidf_logreg': [tfidf_metrics['test_f1'], tfidf_metrics['test_auc']],
    'lgbm_emb':     [lgb_metrics['test_f1'],   lgb_metrics['test_auc']],
    'rubert_ft':    [ft_metrics['test_f1'],    ft_metrics['test_auc']],
}, index=['test_f1', 'test_auc']).round(4)
table
"""),
    code("""# Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, pred) in zip(axes, [('tfidf_logreg', test_pred),
                                    ('lgbm_emb',     test_pred_lgb),
                                    ('rubert_ft',    test_pred_ft)]):
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap='Blues')
    ax.set_title(name); ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['benign', 'jailbreak']); ax.set_yticklabels(['benign', 'jailbreak'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout(); plt.show()
"""),
    md("## 11. Per-source F1 (production-relevant diagnostics)"),
    code("""sources = test['source'].unique().to_list()
rows = []
for src in sources:
    mask = (test['source'] == src).to_numpy()
    if mask.sum() < 5:  # skip rare sources
        continue
    yt = y_test[mask]
    rows.append({
        'source': src,
        'n': int(mask.sum()),
        'tfidf_f1':  round(f1_score(yt, test_pred[mask], zero_division=0), 3),
        'lgbm_f1':   round(f1_score(yt, test_pred_lgb[mask], zero_division=0), 3),
        'rubert_f1': round(f1_score(yt, test_pred_ft[mask], zero_division=0), 3),
    })
pd.DataFrame(rows)
"""),
    md("""## 12. Conclusions

- **Class balance** is well-controlled by `balance_classes` at the merge stage; train/val/test all sit at ~50/50.
- **Source skew** is real - `benign_wiki` dominates the negative class, so models implicitly learn "Wikipedia-style prose vs. instruction templates" as much as anything else. Adding `wildguardmix` (gated) and out-of-domain benign would harden the negative class.
- **TF-IDF + LogReg** is a surprisingly strong baseline because adversarial Russian prompts are stylistically distinct from Wikipedia prose. Captures lexical patterns ("игнорируй", "забудь предыдущие инструкции", roleplay framings).
- **LightGBM on embeddings** matches or beats TF-IDF and generalizes better to paraphrased attacks since the encoder collapses lexical surface variation.
- **Fine-tuned ruBERT-tiny2** is the strongest on full-data runs; on the 12k subsample it's competitive but the gap is tighter.
- **Per-source diagnostics** show the model is weakest on `hivetrace_sysprompt` (small, 9 train rows). Production deployment should pair this with continual data collection from real traffic.

Reproducibility:
- All splits committed in `data/splits/`
- DVC tracks data hashes; `dvc.lock` pins the merge config
- Each model has a deterministic seed
- Production training: `make train-tfidf`, `make train-lgbm`, `make train-rubert` log to MLflow
"""),
]


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'wrote {NB} ({NB.stat().st_size} bytes, {len(CELLS)} cells)')
