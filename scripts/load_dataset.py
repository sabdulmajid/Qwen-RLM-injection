"""OOLONG Dataset Loader."""

import os
import json
from datasets import load_dataset


def main():
    print("=" * 60)
    print("OOLONG Dataset Loader")
    print("=" * 60)
    
    print("\n[1/4] Downloading OOLONG-real (dnd config)...")
    ds = load_dataset("oolongbench/oolong-real", "dnd")
    
    print(f"\n[2/4] Dataset loaded successfully!")
    print(f"Available splits: {list(ds.keys())}")
    
    print(f"\n[3/4] Dataset structure:")
    for split_name, split_data in ds.items():
        print(f"  - {split_name}: {len(split_data)} examples")
        if len(split_data) > 0:
            first_example = split_data[0]
            print(f"    Keys: {list(first_example.keys())}")
            print(f"    Context length: {len(first_example.get('context_window_text', ''))} chars")
            print(f"    Question: {first_example.get('question', '')[:100]}...")
            print(f"    Answer: {first_example.get('answer', '')}")
    
    print(f"\n[4/4] Saving samples...")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in ds.items():
        if len(split_data) > 0:
            sample = {
                "split": split_name,
                "num_examples": len(split_data),
                "example": {
                    "question": split_data[0].get("question", ""),
                    "answer": split_data[0].get("answer", ""),
                    "context_preview": split_data[0].get("context_window_text", "")[:2000] + "...",
                    "context_full_length": len(split_data[0].get("context_window_text", ""))
                }
            }
            
            output_path = os.path.join(output_dir, f"oolong_sample_{split_name}.json")
            with open(output_path, "w") as f:
                json.dump(sample, f, indent=2)
            print(f"  Saved: {output_path}")
    
    print(f"\n{'=' * 60}")
    print("Dataset cached successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    main()
