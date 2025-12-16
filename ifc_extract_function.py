import ifcopenshell
import pandas as pd
import argparse
import os

def extract_ifc(ifc_path):
    """
    Extracts GUID, element type, and property sets from IfcElement entities.

    Args:
        ifc_path (_type_): _description_
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
        props = {"element_type":etype}
        if e.IsDefinedBy:
            for rel in e.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if pset.is_a("IfcPropertySet"):
                        for p in pset.HasProperties:
                            if p.is_a("IfcPropertySingleValue"):
                                if p.NominalValue:
                                    props[p.Name] = str(p.NominalValue.wrappedValue)
            rows.append({"guid": guid, "properties": props})
    return pd.DataFrame(rows)
def parse_args():
    """
    Handles command line arguments for file paths
    """
    parser = argparse.ArgumentParser(description= "Extracts IFC elements for prediction")
    parser.add_argument("--ifc_path", type=str, required=True, help="Path to the input IFC file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted IFC data")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Extract IFC data ---
    print(f"Loading IFc file from: {args.ifc_path}")
    df = extract_ifc(args.ifc_path)
    if df.empty:
        print("Could noot process IFC file or no IfcElement found. Exiting.")
        return
    
    # --- Output to Excel ---
    print(f"Data extracted. Saving to: {args.output_path} for prediction.")
    df.to_excel(args.output_path, index=False)

if __name__ == "__main__":
    main()
    