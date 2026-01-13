import os
import json

def verify_storage():
    kb_dir = "data/knowledge_base"
    target_string = "Wirkungslinie"
    found = False

    print(f"Scanning {kb_dir} for '{target_string}'...")

    for filename in os.listdir(kb_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(kb_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if target_string in content:
                        print(f"FOUND DATA in: {filename}")
                        start_idx = content.find(target_string)
                        snippet = content[start_idx:start_idx+200]
                        print(f"   Snippet: ...{snippet}...")
                        found = True
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    if not found:
        print("\ ERROR: The data is NOT in the database.")

    else:
        print("\nSUCCESS: The rich text is physically stored on disk.")

if __name__ == "__main__":
    verify_storage()