#!/usr/bin/env python3

import os
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer, util

# Optional FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Supervised classifier
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Optional LLM fallback (causal generation) - use a local HF model if provided
from transformers import AutoModelForCausalLM, AutoTokenizer as AutoTokCausal, pipeline as hf_pipeline
import torch

# -----------------------------
# Config & defaults
# -----------------------------
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"   
DEFAULT_SEQ_MODEL = "roberta-base"          
DEFAULT_CLS_EPOCHS = 3
EMB_CACHE_DIR = "emb_cache"
UNIDB_CACHE = "uniclass_db.npz"             # composite texts + embeddings
FAISS_INDEX_PATH = "uniclass_faiss.index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utilities
# -----------------------------
def mkdirp(p: str):
    os.makedirs(p, exist_ok=True)

def hash_text(t: str) -> str:
    import hashlib
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def embed_texts_with_cache(texts: List[str], model: SentenceTransformer, cache_dir: str = EMB_CACHE_DIR,
                           batch_size: int = 64) -> np.ndarray:
    """
    Cache per-text embedding in cache_dir using MD5 filenames. Returns NxD numpy array.
    """
    mkdirp(cache_dir)
    out = []
    to_compute = []
    idxs_to_compute = []
    for i, t in enumerate(texts):
        h = hash_text(t)
        path = os.path.join(cache_dir, f"{h}.npy")
        if os.path.exists(path):
            emb = np.load(path)
            out.append(emb)
        else:
            out.append(None)
            to_compute.append(t)
            idxs_to_compute.append(i)
    if to_compute:
        embs = model.encode(to_compute, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        for ind, emb in zip(idxs_to_compute, embs):
            out[ind] = emb
            np.save(os.path.join(cache_dir, f"{hash_text(texts[ind])}.npy"), emb)
    return np.vstack(out)

# -----------------------------
# Build Uniclass DB
# -----------------------------
def build_uniclass_items_from_sheet(df: pd.DataFrame, code_col='Code', title_col='Title', table_type='Ss') -> List[Dict]:
    """
    Convert Uniclass dataframe into standardized list of items.
    df must contain code + description/title; ancestry optional.
    """
    items = []
    for _, r in df.iterrows():
        code = str(r.get(code_col, "")).strip()
        title = str(r.get(title_col, "")).strip()

        if not code or code.lower() in ("nan", "none"):
            continue

        items.append({
            "code": code,
            "title": title,
            "table_type": table_type,  # Add table type identifier
            # keep keys for compatibility, but empty
            "description": "",
            "ancestry": [],
            "keywords": [],
            "aliases": [],
            "ifc_hints": []
        })

    return items

def composite_uniclass_text(item: Dict) -> str:
    return f"{item['code']} — {item['title']}"

def build_uniclass_db_and_embeddings(ss_df: pd.DataFrame, ef_df: pd.DataFrame, embed_model_name: str = DEFAULT_EMBED_MODEL,
                                     cache_path: str = UNIDB_CACHE, cache_dir: str = EMB_CACHE_DIR) -> Dict:
    """
    Build a combined Uniclass DB for both Ss and Ef (you can adapt to other tables).
    Returns a dict with:
      db['items'] = list of items (dict)
      db['codes'] = list of codes
      db['texts'] = list of composite texts
      db['embeddings'] = NxD numpy array
    and saves to cache_path.
    """
    s_items = build_uniclass_items_from_sheet(ss_df, code_col='Code', title_col='Title', table_type='Ss')
    e_items = build_uniclass_items_from_sheet(ef_df, code_col='Code', title_col='Title', table_type='Ef')
    items = s_items + e_items
    # build composite texts
    texts = [composite_uniclass_text(it) for it in items]
    codes = [it['code'] for it in items]
    table_types = [it['table_type'] for it in items]  # Track table types

    model = SentenceTransformer(embed_model_name, device=DEVICE)
    embs = embed_texts_with_cache(texts, model, cache_dir=cache_dir, batch_size=64)

    # save
    np.savez_compressed(cache_path, codes=np.array(codes), texts=np.array(texts), 
                        embeddings=embs, table_types=np.array(table_types))
    print(f"Saved Uniclass DB to {cache_path} (N={len(codes)})")
    db = {"items": items, "codes": codes, "texts": texts, "embeddings": embs, 
          "embed_model_name": embed_model_name, "table_types": table_types}
    return db

def load_uniclass_db(cache_path: str = UNIDB_CACHE) -> Dict:
    arr = np.load(cache_path, allow_pickle=True)
    codes = arr['codes'].tolist()
    texts = arr['texts'].tolist()
    embeddings = arr['embeddings']
    table_types = arr['table_types'].tolist() if 'table_types' in arr else ['Unknown'] * len(codes)
    items = []
    for c, t, tt in zip(codes, texts, table_types):
        items.append({"code": c, "composite_text": t, "table_type": tt})
    return {"items": items, "codes": codes, "texts": texts, "embeddings": embeddings, "table_types": table_types}

# -----------------------------
# Optional: build FAISS index for fast NN
# -----------------------------
def build_faiss_index(embeddings: np.ndarray, index_path: str = FAISS_INDEX_PATH):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available. Install faiss-cpu or faiss-gpu.")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)   # use inner-product; normalise vectors for cosine
    arr = embeddings.astype('float32')
    # normalize
    faiss.normalize_L2(arr)
    index.add(arr)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index

def load_faiss_index(index_path: str):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available.")
    return faiss.read_index(index_path)

# -----------------------------
# Supervised classifier: prepare HF dataset
# -----------------------------
def prepare_supervised_dataset(df: pd.DataFrame, text_col: str = 'properties', label_col: str = 'label_code'):
    """
    Expects df rows to have 'merged_text' and 'label_code'
    returns hf_dataset (Dataset), label_encoder
    """
    df2 = df[[text_col, label_col]].dropna().copy()
    if df2.empty:
        return None, None
    le = LabelEncoder()
    df2['label_id'] = le.fit_transform(df2[label_col])
    ds = Dataset.from_pandas(df2.rename(columns={text_col: 'text', 'label_id': 'label'})[['text','label']])
    return ds, le

def train_sequence_classifier(hf_dataset: Dataset, label_encoder: LabelEncoder,
                              model_name: str = DEFAULT_SEQ_MODEL, out_dir: str = "hf_uniclass_model",
                              epochs: int = DEFAULT_CLS_EPOCHS, per_device_batch_size: int = 8):
    """
    Train a sequence classification model with HF Trainer.
    """
    num_labels = len(label_encoder.classes_)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = hf_dataset['train'].map(tokenize_fn, batched=True)
    eval_ds = hf_dataset['test'].map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        logging_dir=f"{out_dir}/logs",
        fp16=True if torch.cuda.is_available() else False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": metric.compute(predictions=preds, references=labels)}
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir, tokenizer, model

# -----------------------------
# Inference helpers: hierarchical backoff & combined scoring
# -----------------------------
from sklearn.metrics.pairwise import cosine_similarity

def keyword_match_score(prop_text: str, keywords: List[str]) -> float:
    if not keywords: return 0.0
    tokens = prop_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in tokens)
    return hits / len(keywords)

def ifc_hint_score(element_type: str, hints: List[str]) -> float:
    if not hints: return 0.0
    et = element_type.lower()
    for h in hints:
        if h.lower() in et:
            return 1.0
    return 0.0

def combined_topk_scoring(prop_emb: np.ndarray, candidate_embs: np.ndarray, topk_idxs: List[int],
                          items: List[Dict], prop_text: str, element_type: str,
                          weights: Dict[str,float] = None) -> List[Tuple[str, float, str]]:
    """
    For topk candidate indices, compute combined score and return sorted list of (code, score, table_type).
    weights: {'sim':0.7,'hint':0.15,'kw':0.15}
    """
    if weights is None:
        weights = {'sim':0.75, 'hint':0.15, 'kw':0.10}
    out = []
    for idx in topk_idxs:
        cand = items[idx]
        cand_emb = candidate_embs[idx]
        sim = float(cosine_similarity(prop_emb.reshape(1,-1), cand_emb.reshape(1,-1))[0,0])
        hscore = ifc_hint_score(element_type, cand.get('ifc_hints',[]))
        kscore = keyword_match_score(prop_text, cand.get('keywords',[]))
        score = weights['sim']*sim + weights['hint']*hscore + weights['kw']*kscore
        out.append((cand['code'], score, cand.get('table_type', 'Unknown')))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# -----------------------------
# Hybrid predict function - MODIFIED FOR BOTH Ss AND Ef
# -----------------------------
class HybridPredictor:
    def __init__(self, classifier_trainer_dir: Optional[str], tokenizer_seq=None, hf_model_seq=None,
                 uniclass_db: Dict = None, faiss_index = None, embed_model_name: str = DEFAULT_EMBED_MODEL,
                 llm_model_name: Optional[str] = None, confidence_threshold: float = 0.6):
        # load embedder
        self.embedder = SentenceTransformer(embed_model_name, device=DEVICE)
        self.uniclass_db = uniclass_db  # dict with items,codes,texts,embeddings
        self.faiss_index = faiss_index
        # classifier
        self.seq_tokenizer = tokenizer_seq
        self.seq_model = hf_model_seq
        if classifier_trainer_dir and not (tokenizer_seq and hf_model_seq):
            # attempt load from dir
            print("Loading classifier from", classifier_trainer_dir)
            self.seq_tokenizer = AutoTokenizer.from_pretrained(classifier_trainer_dir)
            self.seq_model = AutoModelForSequenceClassification.from_pretrained(classifier_trainer_dir).to(DEVICE)
        # LLM fallback (optional)
        self.llm_name = llm_model_name
        if llm_model_name:
            print("Loading causal LLM for fallback:", llm_model_name)
            self.llm_tok = AutoTokCausal.from_pretrained(llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")
            # simple wrapper
            self.llm_pipe = hf_pipeline("text-generation", model=self.llm, tokenizer=self.llm_tok, device=0 if torch.cuda.is_available() else -1)
        else:
            self.llm = None
        self.conf_thresh = confidence_threshold

    def classify_with_model(self, text: str):
        if self.seq_model is None:
            return None, 0.0
        toks = self.seq_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = self.seq_model(**toks).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            # label mapping requires external label_encoder: caller will map index->code
            return best_idx, conf

    def llm_fallback(self, text: str, prompt_template: Optional[str] = None) -> Tuple[Optional[str], float]:
        """
        Use the causal LLM to generate a predicted code. This requires the model to be tuned or a strong general model.
        We wrap in a simple prompt; you should adapt it to your model.
        Returns predicted_code, dummy_score (0-1) based on heuristics.
        """
        if self.llm is None:
            return None, 0.0
        prompt = (prompt_template or "Given the following IFC element properties, return the Uniclass code (exact token) for Ss or Ef that best matches. If unsure, return the most general parent code.\n\nINPUT:\n") + text + "\n\nOUTPUT:"
        # generate short
        out = self.llm_pipe(prompt, max_new_tokens=16, do_sample=False, num_return_sequences=1)[0]['generated_text']
        # naive parse: take last token-like sequence after 'OUTPUT:'
        pred = out.split("OUTPUT:")[-1].strip().split()[0].strip()
        # confidence heuristic: check if pred in DB codes
        conf = 1.0 if self.uniclass_db and pred in self.uniclass_db['codes'] else 0.0
        return pred, conf

    def get_top_similarities(
        self,
        text: str,
        topk: int = 10,
        filter_table: str = None
    ):
        emb = self.embedder.encode([text], convert_to_numpy=True)[0]

        items = self.uniclass_db["items"]
        cand_embs = self.uniclass_db["embeddings"]

        fetch_k = topk * 2 if filter_table else topk

        # ---- retrieve candidates
        if self.faiss_index is not None:
            v = emb.astype("float32")
            faiss.normalize_L2(v.reshape(1, -1))
            _, I = self.faiss_index.search(v.reshape(1, -1), fetch_k)
            idxs = I[0].tolist()
        else:
            sims = util.cos_sim(emb, cand_embs)[0]
            if isinstance(sims, torch.Tensor):
                sims = sims.cpu().numpy()
            idxs = np.argsort(sims)[-fetch_k:][::-1].tolist()

        # ---- FILTER BY TABLE
        if filter_table:
            idxs = [i for i in idxs if items[i]["table_type"] == filter_table]

        if not idxs:
            return []

        # ---- score & return
        return combined_topk_scoring(
            prop_emb=emb,
            candidate_embs=cand_embs,
            topk_idxs=idxs[:topk],
            items=items,
            prop_text=text,
            element_type="",
            weights=None
        )

    def hybrid_predict_both(
        self,
        text: str,
        label_encoder: Optional[LabelEncoder] = None
    ) -> Dict:

        result = {
            "text": text,
            "Ss_code": "",
            "Ss_confidence": 0.0,
            "Ef_code": "",
            "Ef_confidence": 0.0,
            "source": "similarity"
        }

        # ------------------
        # 1) Similarity baseline
        # ------------------
        ss_matches = self.get_top_similarities(text, filter_table="Ss")
        ef_matches = self.get_top_similarities(text, filter_table="Ef")

        if ss_matches:
            result["Ss_code"] = ss_matches[0][0]
            result["Ss_confidence"] = ss_matches[0][1]

        if ef_matches:
            result["Ef_code"] = ef_matches[0][0]
            result["Ef_confidence"] = ef_matches[0][1]

        # ------------------
        # 2) Classifier override
        # ------------------
        if self.seq_model and label_encoder is not None:
            idx, conf = self.classify_with_model(text)

            if conf >= self.conf_thresh:
                try:
                    pred_code = label_encoder.inverse_transform([idx])[0]
                    code_to_table = {i["code"]: i["table_type"] for i in self.uniclass_db["items"]}
                    table = code_to_table.get(pred_code)

                    if table == "Ss" and conf > result["Ss_confidence"]:
                        result["Ss_code"] = pred_code
                        result["Ss_confidence"] = conf
                        result["source"] = "classifier"

                    elif table == "Ef" and conf > result["Ef_confidence"]:
                        result["Ef_code"] = pred_code
                        result["Ef_confidence"] = conf
                        result["source"] = "classifier"

                except Exception:
                    pass

        # ------------------
        # 3) Optional LLM fallback
        # ------------------
        if self.llm and (not result["Ss_code"] or not result["Ef_code"]):
            llm_pred, llm_conf = self.llm_fallback(text)
            if llm_pred:
                code_to_table = {i["code"]: i["table_type"] for i in self.uniclass_db["items"]}
                table = code_to_table.get(llm_pred)

                if table == "Ss" and not result["Ss_code"]:
                    result["Ss_code"] = llm_pred
                    result["Ss_confidence"] = llm_conf
                    result["source"] = "llm"

                elif table == "Ef" and not result["Ef_code"]:
                    result["Ef_code"] = llm_pred
                    result["Ef_confidence"] = llm_conf
                    result["source"] = "llm"

        return result

def run_train_and_build(args):
    # load input files
    print("Loading Uniclass sheets...")
    ss_df = pd.read_excel(args.ss_excel) if args.ss_excel.endswith(('.xls','.xlsx')) else pd.read_csv(args.ss_excel)
    ef_df = pd.read_excel(args.ef_excel) if args.ef_excel.endswith(('.xls','.xlsx')) else pd.read_csv(args.ef_excel)

    # Build Uniclass DB (cache)
    db = build_uniclass_db_and_embeddings(ss_df, ef_df, embed_model_name=args.embed_model, cache_path=args.unidb_cache, cache_dir=args.emb_cache_dir)

    # load ifc-extracted df (expects pickled DataFrame or csv)

    ifc_df = pd.read_excel(args.ifc_extracted)
        # properties column might be JSON strings
    if 'properties' in ifc_df.columns and ifc_df['properties'].dtype == object:
            # try parse JSON strings
        try:
            ifc_df['properties'] = ifc_df['properties'].apply(lambda x: json.loads(x) if isinstance(x,str) else x)
        except Exception:
            pass

    # prepare supervised dataset if labels exist
    if 'label_code' in ifc_df.columns and not ifc_df['label_code'].dropna().empty:
        ds, le = prepare_supervised_dataset(ifc_df, text_col='properties', label_col='label_code')
        print("Training classifier on labeled data (num labels):", len(le.classes_))
        model_dir, tokenizer, model = train_sequence_classifier(ds, le, model_name=args.seq_model, out_dir=args.out_model_dir, epochs=args.epochs, per_device_batch_size=args.batch_size)
        # save label encoder
        import joblib
        joblib.dump(le, os.path.join(args.out_model_dir, "label_encoder.joblib"))
    else:
        print("No label_code column with data found in IFC-extracted file. Skipping classifier training.")
        ds, le = None, None
        model_dir, tokenizer, model = None, None, None

    # Optional: build FAISS index
    faiss_idx = None
    if FAISS_AVAILABLE:
        try:
            faiss_idx = build_faiss_index(db['embeddings'], index_path=args.faiss_index)
        except Exception as e:
            print("FAISS index build failed:", e)
            faiss_idx = None

    # Save DB to path
    np.savez_compressed(args.unidb_cache, codes=np.array(db['codes']), texts=np.array(db['texts']), 
                        embeddings=db['embeddings'], table_types=np.array(db['table_types']))
    print("Uniclass DB ready.")

    print("Training and DB build done. You can now call --predict to run hybrid inference.")
    return db, le, tokenizer, model, faiss_idx

def run_predict(args):
    # load DB
    db = load_uniclass_db(args.unidb_cache)
    faiss_index = None
    if FAISS_AVAILABLE and os.path.exists(args.faiss_index):
        faiss_index = load_faiss_index(args.faiss_index)
    # load classifier
    le = None
    tokenizer = None
    model = None
    if args.classifier_dir and os.path.exists(args.classifier_dir):
        import joblib
        try:
            le = joblib.load(os.path.join(args.classifier_dir,"label_encoder.joblib"))
        except Exception:
            le = None
        tokenizer = AutoTokenizer.from_pretrained(args.classifier_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.classifier_dir).to(DEVICE)
    # if LLM specified
    llm_name = args.llm if args.llm else None

    predictor = HybridPredictor(classifier_trainer_dir=None, tokenizer_seq=tokenizer, hf_model_seq=model,
                                uniclass_db=db, faiss_index=faiss_index, embed_model_name=args.embed_model, llm_model_name=llm_name,
                                confidence_threshold=args.confidence)


    df = pd.read_excel(args.ifc_extracted)
    if 'properties' in df.columns and df['properties'].dtype == object:
        try:
            # Fix: Use ast.literal_eval to parse Python dictionary strings
            import ast
            def parse_properties(value):
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    # Check if it looks like a dictionary string
                    stripped = value.strip()
                    if stripped.startswith('{') and stripped.endswith('}'):
                        try:
                            return ast.literal_eval(stripped)
                        except:
                            # Try JSON as fallback
                            try:
                                import json
                                # Replace single quotes with double quotes for JSON
                                json_str = stripped.replace("'", '"')
                                return json.loads(json_str)
                            except:
                                return {}  # Return empty dict if parsing fails
                    else:
                        # Not a dictionary string, return as-is
                        return value
                else:
                    return value
                
            df['properties'] = df['properties'].apply(parse_properties)
        except Exception as e:
            print(f"Warning: Could not parse properties column: {e}")
            # Keep original code as fallback
            try:
                df['properties'] = df['properties'].apply(lambda x: json.loads(x) if isinstance(x,str) else x)
            except Exception:
                pass
    
    # Load Uniclass sheets for title lookups
    uniclass_df_ss = pd.read_excel(args.ss_excel) if args.ss_excel.endswith(('.xls','.xlsx')) else pd.read_csv(args.ss_excel)
    uniclass_df_ef = pd.read_excel(args.ef_excel) if args.ef_excel.endswith(('.xls','.xlsx')) else pd.read_csv(args.ef_excel)
    
    # Create lookup dictionaries for code to title
    ss_lookup = dict(zip(uniclass_df_ss['Code'].astype(str).str.strip(), 
                         uniclass_df_ss['Title'].astype(str).str.strip()))
    ef_lookup = dict(zip(uniclass_df_ef['Code'].astype(str).str.strip(), 
                         uniclass_df_ef['Title'].astype(str).str.strip()))

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if 'merged_text' in row and pd.notna(row['merged_text']):
            text = str(row['merged_text'])

        elif 'properties' in row and isinstance(row['properties'], dict):
            # flatten properties dict into text
            text = " ".join([f"{k}: {v}" for k, v in row['properties'].items()])

        elif 'properties' in row and pd.notna(row['properties']):
            text = str(row['properties'])

        else:
            continue  # nothing usable, skip row

        text = text.strip()
        if not text:
            continue

        # Use the NEW hybrid_predict_both method
        res = predictor.hybrid_predict_both(text, label_encoder=le)
        
        # Get Ss and Ef predictions
        Ss_Code = res.get('Ss_code', '')
        Ss_Title = ss_lookup.get(Ss_Code, '')
        Ss_confidence = res.get('Ss_confidence', 0.0)
        
        Ef_Code = res.get('Ef_code', '')
        Ef_Title = ef_lookup.get(Ef_Code, '')
        Ef_confidence = res.get('Ef_confidence', 0.0)
        
        # Calculate overall confidence (average of Ss and Ef)
        confidence_rate = (Ss_confidence + Ef_confidence) / 2 if (Ss_confidence + Ef_confidence) > 0 else 0.0
        
        # Create new row with original data + new columns
        new_row = row.to_dict()
        new_row['Ss_Code'] = Ss_Code
        new_row['Ss_Title'] = Ss_Title
        new_row['Ss_confidence'] = Ss_confidence
        new_row['Ef_code'] = Ef_Code
        new_row['Ef_Title'] = Ef_Title
        new_row['Ef_confidence'] = Ef_confidence
        new_row['confidence_rate'] = confidence_rate
    
        
        results.append(new_row)
    
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")
    print(f"Added columns: Ss_Code, Ss_Title, Ss_confidence, Ef_code, Ef_Title, Ef_confidence, confidence_rate")
    return out_df

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ss_excel", type=str, default="data/Uniclass2015_Ss_v1_40.xlsx", help="Ss table CSV/XLS")
    p.add_argument("--ef_excel", type=str, default="data/Uniclass2015_EF_v1_16.xlsx", help="Ef table CSV/XLS")
    p.add_argument("--ifc_extracted", type=str, default="ifc_extracted.pkl", help="Extracted IFC dataframe (pkl/csv) with columns guid,element_type,properties and optional label_code.")
    p.add_argument("--run_train", action="store_true", help="Run training + build DB")
    p.add_argument("--predict", action="store_true", help="Run prediction (requires DB and optional classifier)")
    p.add_argument("--unidb_cache", type=str, default=UNIDB_CACHE, help="Path to save/load Uniclass DB (.npz)")
    p.add_argument("--emb_cache_dir", type=str, default=EMB_CACHE_DIR, help="Embedding cache dir")
    p.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model for embeddings")
    p.add_argument("--seq_model", type=str, default=DEFAULT_SEQ_MODEL, help="seq model backbone for classifier")
    p.add_argument("--out_model_dir", type=str, default="hf_uniclass_model", help="where to save classifier")
    p.add_argument("--epochs", type=int, default=DEFAULT_CLS_EPOCHS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--faiss_index", type=str, default=FAISS_INDEX_PATH)
    p.add_argument("--classifier_dir", type=str, default=None, help="directory of trained HF classifier for prediction")
    p.add_argument("--llm", type=str, default=None, help="optional causal LLM HF model id for fallback (local or HF)")
    p.add_argument("--confidence", type=float, default=0.6, help="classifier confidence threshold for fallback")
    p.add_argument("--output_csv", type=str, default="predictions.csv")
    return p.parse_args()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.run_train:
        # Build Uniclass DB AND train classifier if labeled data exists
        print("Starting training pipeline...")
        run_train_and_build(args)
    if args.predict:
        run_predict(args)