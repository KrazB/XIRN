import ifcopenshell
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import argparse
import os

# --- Helper Functions ---

def extract_ifc(ifc_path):
    """
    Extracts GUID, element type, and property sets from IfcElement entities.
    """
    try:
        m = ifcopenshell.open(ifc_path)
    except Exception as e:
        print(f"Error loading IFC file {ifc_path}: {e}")
        return pd.DataFrame()
    
    rows = []
    for e in m.by_type("IfcElement"):
        guid = e.GlobalId
        etype = e.is_a()
        props = {}
        if e.IsDefinedBy:
            for rel in e.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if pset.is_a("IfcPropertySet"):
                        for p in pset.HasProperties:
                            if p.is_a("IfcPropertySingleValue"):
                                if p.NominalValue:
                                    props[p.Name] = str(p.NominalValue.wrappedValue)
        rows.append({"guid": guid, "element_type": etype, "properties": props})
    return pd.DataFrame(rows)

# Initialize the model outside the matching function to prevent re-loading on every call
# This optimization significantly speeds up the process.
try:
    GLOBAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error initializing SentenceTransformer: {e}. Check internet connection or environment.")
    GLOBAL_MODEL = None


def match(text, emb_bank, codes):
    """
    Performs vector similarity search (cosine similarity) against the embedding bank.
    """
    if GLOBAL_MODEL is None:
        return None, 0.0
        
    # Encode the input text
    vec = GLOBAL_MODEL.encode([text], convert_to_tensor=True)
    
    # Calculate cosine similarity with the entire bank
    sims = util.cos_sim(vec, emb_bank)[0]
    
    # Find the index of the highest similarity score
    idx = sims.argmax().item()
    
    return codes[idx], float(sims[idx])


def parse_args():
    """
    Handles command line arguments for file paths.
    """
    parser = argparse.ArgumentParser(description="Classifies IFC elements using Sentence-Transformers and Uniclass.")
    parser.add_argument("--ifc_path", type=str, required=True, help="Path to the input IFC file (e.g., ./data/model.ifc).")
    parser.add_argument("--ss_path", type=str, required=True, help="Path to the Uniclass Ss Excel file (e.g., ./data/Uniclass2015_Ss.xlsx).")
    parser.add_argument("--ef_path", type=str, required=True, help="Path to the Uniclass Ef Excel file (e.g., ./data/Uniclass2015_EF.xlsx).")
    parser.add_argument("--output_path", type=str, default="IFC_Uniclass_Predictions_Enriched.xlsx", help="Path for the output Excel file.")
    return parser.parse_args()


# --- Main Execution Function ---

def main():
    if GLOBAL_MODEL is None:
        print("Model failed to initialize. Exiting.")
        return

    args = parse_args()
    
    # --- 1. Extract IFC → DataFrame ---
    print(f"Loading IFC file from: {args.ifc_path}")
    df = extract_ifc(args.ifc_path)
    if df.empty:
        print("Could not process IFC file or no IfcElement found. Exiting.")
        return

    # --- 2. Flatten properties to text for embedding ---
    df["merged_text"] = df.apply(lambda r: r["element_type"] + " " +
        " ".join(f"{k}:{v}" for k,v in r["properties"].items()), axis=1)

    # --- 3. Load Uniclass tables ---
    print("Loading Uniclass tables and preparing embeddings...")
    ss = pd.read_excel(args.ss_path)
    ef = pd.read_excel(args.ef_path)

    # Standardize and rename columns for clarity
    ss.columns = [c.lower() for c in ss.columns]
    ef.columns = [c.lower() for c in ef.columns]

    ss = ss[["code","title"]].rename(columns={"code":"ss_code","title":"ss_title"})
    ef = ef[["code","title"]].rename(columns={"code":"ef_code","title":"ef_title"})
    
    # --- 4. Zero-shot embedding + matching ---
    print("Encoding Uniclass titles for similarity matching...")
    ss_emb = GLOBAL_MODEL.encode(ss["ss_title"].tolist(), batch_size=128, show_progress_bar=True, convert_to_tensor=True)
    ef_emb = GLOBAL_MODEL.encode(ef["ef_title"].tolist(), batch_size=128, show_progress_bar=True, convert_to_tensor=True)

    print("Matching IFC elements to Uniclass codes...")
    results_ss = [match(txt, ss_emb, ss["ss_code"].tolist()) for txt in df["merged_text"]]
    results_ef = [match(txt, ef_emb, ef["ef_code"].tolist()) for txt in df["merged_text"]]

    df["pred_ss_code"], df["pred_ss_score"] = zip(*results_ss)
    df["pred_ef_code"], df["pred_ef_score"] = zip(*results_ef)

    # --- 5. Merge Titles back into the main DataFrame ---
    print("Merging Uniclass titles into results...")
    
    # Merge for Ss_Title
    df = df.merge(
        ss[["ss_code", "ss_title"]],
        left_on="pred_ss_code",
        right_on="ss_code",
        how="left"
    ).drop(columns=["ss_code"]).rename(columns={"ss_title": "pred_ss_title"})


    # Merge for Ef_Title
    df = df.merge(
        ef[["ef_code", "ef_title"]],
        left_on="pred_ef_code",
        right_on="ef_code",
        how="left"
    ).drop(columns=["ef_code"]).rename(columns={"ef_title": "pred_ef_title"})

    # --- 6. Output to Excel ---
    print(f"Final output structure complete. Saving to: {args.output_path}")
    df.to_excel(args.output_path, index=False)

if __name__ == "__main__":
    main()