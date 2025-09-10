
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load and prepare reference Uniclass data
@st.cache_data
def load_reference_data():
    df = pd.read_csv("/Users/irenie/Downloads/X_LIB_R1 REOL(X_MAP_R1 REOL).csv", encoding="ISO-8859-1")
    df = df.dropna(subset=["X_OBJ Name", "Uniclass Systems"]).reset_index(drop=True)
    df[['Uniclass_Code', 'Uniclass_Name']] = df['Uniclass Systems'].str.extract(r'^([A-Za-z]{2}_\d{2}(?:_\d{2}){0,3})\s+(.*)$')
    df['Uniclass_Name'] = df['Uniclass_Name'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()

    df["input_text"] = (
        "Name: " + df["X_OBJ Name"].astype(str).fillna("") + " | " +
        "Type: " + df["X_OBJ Type"].astype(str).fillna("") + " | " +
        "Class: " + df["X_OBJ IfcClass"].astype(str).fillna("") + " | " +
        "External: | LoadBearing: | Keyword: | PPK Active Object: | PPK Active Task: | Object Description:"
    )

    grouped = df.groupby("Uniclass_Name")[["input_text", "Uniclass_Code"]].agg({
        "input_text": lambda texts: " ".join(texts),
        "Uniclass_Code": "first"
    }).reset_index()
    return grouped

# Retrieval function
def retrieve_uniclass(input_description, data_grouped, model, top_k=5):
    class_embeddings = model.encode(data_grouped["input_text"].tolist(), convert_to_tensor=True)
    input_embedding = model.encode(input_description, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, class_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        i = int(idx)
        row = data_grouped.iloc[i]
        results.append({
            "Uniclass_Code": row["Uniclass_Code"],
            "Uniclass_Name": row["Uniclass_Name"],
            "Similarity": float(score)
        })
    return pd.DataFrame(results)

# Streamlit UI
st.set_page_config(page_title="Manual Input - Semantic Uniclass Retrieval", layout="centered")
st.title("🧠 Manual Semantic Search for Uniclass Classification")

st.write("Fill in the object details below to retrieve the most similar Uniclass category using BERT.")

# Manual input fields
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("X_OBJ Name")
    obj_type = st.text_input("X_OBJ Type")
    ifc_class = st.text_input("IfcClass.Type")
    is_external = st.text_input("X_IsExternal")
    load_bearing = st.text_input("X_LoadBearing")
with col2:
    keyword = st.text_input("X_OBJ Keyword")
    ppk_obj = st.text_input("PPK Active Object")
    ppk_task = st.text_input("PPK Active Task")
    obj_description = st.text_area("Object Description")

top_k = st.slider("Top N Matches", 1, 10, 5)

if st.button("🔍 Retrieve"):
    # Build the full input_text
    input_text = (
        f"Name: {name} | Type: {obj_type} | Class: {ifc_class} | "
        f"External: {is_external} | LoadBearing: {load_bearing} | "
        f"Keyword: {keyword} | PPK Active Object: {ppk_obj} | "
        f"PPK Active Task: {ppk_task} | Object Description: {obj_description}"
    )

    with st.spinner("Running semantic retrieval..."):
        model = load_model()
        data_grouped = load_reference_data()
        results = retrieve_uniclass(input_text, data_grouped, model, top_k)
        st.success("✅ Retrieval complete.")
        st.dataframe(results, use_container_width=True)
