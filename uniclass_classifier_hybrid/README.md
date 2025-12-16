# Uniclass Classifier for IFC Data

A hybrid pipeline for classifying IFC elements into Uniclass Ss and Ef codes.

## Installation
```bash

# Clone the repository
git clone
cd uniclass_classifier_hybrid

# Install dependencies
pip install -r requirements.txt

# Running modified file_path
#Extracting data from IFC file
python ifc_extract_function.py \
    --ifc_path ./data/input_model.ifc \
    --output_path ./output/predictions.xlsx
#Running the predictions
python hybrid_uniclass_pipeline.py --predict --classifier_dir hf_uniclass_model --ifc_extracted ./output/predictions.xlsx
