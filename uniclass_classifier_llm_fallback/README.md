# Uniclass Classifier for IFC Data

A hybrid pipeline for classifying IFC elements into Uniclass Ss and Ef codes.

## Installation

```bash
# Clone the repository
git clone
cd uniclass_classifier_hybrid

# Install dependencies
pip install -r requirements.txt

#Running the Modified Script
python classify_ifc.py \
    --ifc_path ./data/input_model.ifc \
    --ss_path ./data/Uniclass2015_Ss_v1_40.xlsx \ 
    --ef_path ./data/Uniclass2015_EF_v1_16.xlsx \
    --output_path ./output/predictions.xlsx 