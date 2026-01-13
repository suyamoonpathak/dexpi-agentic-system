import sys
import os
from src.ingestion.xml_parser import DexpiParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inspect_valves():
    xml_file = "data/raw/C01V01-HEX.EX03.xml"
    
    parser = DexpiParser()
    data = parser.parse_file(xml_file)
    
    valves = [item for item in data['equipment'] if "Valve" in item['type'] or "Valve" in item['tag']]
    print(f"Found {len(valves)} Valves.")
    
    connections = data['connections']
    valve_ids = {v['id'] for v in valves}
    
    connected_valves = 0
    for conn in connections:
        if conn['source'] in valve_ids or conn['target'] in valve_ids:
            connected_valves += 1
            
    print(f"Valves with explicit connections: {connected_valves}")
    
    if len(valves) > 0 and connected_valves == 0:
        print("FAIL: You have Valves, but they are not connected to anything in the Graph!")
    else:
        print("Valves are connected. The issue is likely Semantic Search.")

if __name__ == "__main__":
    inspect_valves()