"""EP47 - Named Entity Recognition: fine-tune a token classifier, then extract entities.
    python ep47_ner.py                          # fine-tune, evaluate (seqeval), save
    python ep47_ner.py "Sarah joined Google in Paris on 12 May 2025."
CoNLL-2003 has no DATE label: people/orgs/places come from the model, dates from rules.
pip install "transformers>=4.46" datasets torch seqeval accelerate
"""
import re
import sys
import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          pipeline, set_seed)

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
MODEL = "distilbert-base-cased"        # CASED: capital letters are a strong NER clue
DATASET = "tomaarsen/conll2003"        # any dataset with a tokens column + tag column
TOKENS, TAGS = "tokens", "ner_tags"
OUT_DIR = "ep47_ner"
SEED = 42
THRESHOLD = 0.6                        # drop entities the model is unsure about
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
DATE_RE = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b|"
                     rf"\b(?:\d{{1,2}}\s)?(?:{MONTHS})[a-z]*\.?(?:\s\d{{1,2}},?)?\s\d{{4}}\b")

# ── 2. ALIGN LABELS TO SUBWORD TOKENS ─────────────────────────────────────
def align(batch, tok):
    enc = tok(batch[TOKENS], truncation=True, is_split_into_words=True)
    enc["labels"] = []
    for i, tags in enumerate(batch[TAGS]):
        prev, ids = None, []
        for w in enc.word_ids(batch_index=i):
            ids.append(-100 if w is None or w == prev else tags[w])  # first piece only
            prev = w
        enc["labels"].append(ids)
    return enc

# ── 3. SEQEVAL METRICS (entity level, not token level) ────────────────────
def decode(logits, labels, names):
    y_true, y_pred = [], []
    for pr, lb in zip(np.argmax(logits, axis=-1), labels):
        keep = lb != -100
        y_true.append([names[t] for t in lb[keep]])
        y_pred.append([names[t] for t in pr[keep]])
    return y_true, y_pred

def metrics_fn(names):
    def compute(p):
        y_true, y_pred = decode(p.predictions, p.label_ids, names)
        return {"precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
    return compute

# ── 4. TRAIN ──────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset(DATASET)
    names = ds["train"].features[TAGS].feature.names
    enc = ds.map(lambda b: align(b, tok), batched=True, remove_columns=ds["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
    args = TrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", report_to="none", seed=SEED,
        fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=enc["train"],
                      eval_dataset=enc["validation"], processing_class=tok,
                      data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=metrics_fn(names))
    trainer.train()
    # ── 5. EVALUATE ON THE TEST SET ───────────────────────────────────────
    out = trainer.predict(enc["test"])
    print(classification_report(*decode(out.predictions, out.label_ids, names), digits=3))
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

# ── 6. EXTRACT ────────────────────────────────────────────────────────────
def extract(text):
    ner = pipeline("token-classification", model=OUT_DIR, aggregation_strategy="simple",
                   device=0 if torch.cuda.is_available() else -1)
    found = [(e["word"], e["entity_group"], round(float(e["score"]), 3))
             for e in ner(text) if e["score"] >= THRESHOLD]
    found += [(m.group(0), "DATE", "rule") for m in DATE_RE.finditer(text)]
    return found

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for row in extract(" ".join(sys.argv[1:])):
            print(row)
    else:
        train()
