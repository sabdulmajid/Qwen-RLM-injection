"""OOLONG Dataset Exploration and Analysis."""

import os
import json
import numpy as np
from collections import Counter
from datasets import load_dataset


def main():
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    print("=" * 80)
    print("OOLONG DATASET ANALYSIS")
    print("=" * 80)
    
    ds = load_dataset("oolongbench/oolong-real", "dnd")
    
    for split_name in ['validation', 'test']:
        split = ds[split_name]
        print(f"\n{split_name.upper()} SPLIT")
        print(f"Total examples: {len(split)}")
        
        question_types = Counter(split['question_type'])
        print(f"\nQuestion Type Distribution:")
        for qtype, count in question_types.most_common():
            print(f"  {qtype}: {count} ({count/len(split)*100:.1f}%)")
        
        context_lengths = [len(ex['context_window_text']) for ex in split]
        print(f"\nContext Statistics (chars):")
        print(f"  Mean: {np.mean(context_lengths):,.0f}")
        print(f"  Median: {np.median(context_lengths):,.0f}")
        print(f"  Max: {np.max(context_lengths):,}")
        
        token_lengths = [l / 4 for l in context_lengths]
        print(f"\nContext Statistics (estimated tokens):")
        print(f"  Mean: {np.mean(token_lengths):,.0f}")
        print(f"  Max: {np.max(token_lengths):,.0f}")
        
        print(f"\nSample Questions:")
        seen_types = set()
        for ex in split:
            qtype = ex['question_type']
            if qtype not in seen_types:
                print(f"\n  [{qtype}]")
                print(f"    Q: {ex['question'][:100]}...")
                print(f"    A: {ex['answer']}")
                seen_types.add(qtype)
                if len(seen_types) >= 3:
                    break
    
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("1. OOLONG contexts are MASSIVE (up to 1.2M tokens)")
    print("2. Dense reasoning required - info spread across entire context")
    print("3. Perfect testbed for RLM's recursive approach")
    print("=" * 80)


if __name__ == "__main__":
    main()
