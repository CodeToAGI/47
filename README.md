# EP47 – Named Entity Recognition (NER) Explained  
**Deep Learning Series · Module 9 · Token Classification**

> One label per **token**, not per text.  
> Fine-tune DistilBERT (cased) on CoNLL-2003 + hybrid date extraction in < 100 lines.

---

## What you get

| File | Description |
|------|-------------|
| `ep47_ner.py` | Complete 97-line script (train → evaluate → extract) |
| This README | How to run + challenge |

---

## Quick Start

```bash
# 1. Install
pip install "transformers>=4.46" datasets torch seqeval accelerate

# 2. Train + evaluate (saves model to ./ep47)
python ep47_ner.py

# 3. Extract entities from any text
python ep47_ner.py "Sarah joined Google in Paris on 12 May 2025."
