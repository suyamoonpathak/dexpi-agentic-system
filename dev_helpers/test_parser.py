import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.xml_parser import DexpiParser

def main():
    parser = DexpiParser()
    
    file_path = "data/raw/C01V01-HEX.EX02.xml" 

    print(f"Parsing {file_path}...")
    result = parser.parse_file(file_path)
    
    print(f"Equipment Found: {len(result['equipment'])}")
    print(f"Connections Found: {len(result['connections'])}")
    
    if result['equipment']:
        print("\n--- Sample Equipment ---")
        print(json.dumps(result['equipment'][:5], indent=2))
        
    if result['connections']:
        print("\n--- Sample Connections ---")
        print(json.dumps(result['connections'][:5], indent=2))

if __name__ == "__main__":
    main()