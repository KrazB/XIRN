# Uniclass Classifier for IFC Data

A hybrid pipeline for classifying IFC elements into Uniclass Ss and Ef codes.

## Installation

```bash
# --- Running on terminal ---
# Clone the repository
git clone
cd uniclass_classifier_hybrid

# Install dependencies
pip install -r requirements.txt

#Extracting data from .ifc file
python ifc_extract_function.py --ifc_path ./data.ifc --output_path ./data_output.xlsx

#Running the Modified Script
python hybrid_uniclass_pipeline.py --predict --classifier_dir hf_uniclass_model --ifc_extracted ./data_output.xlsx
