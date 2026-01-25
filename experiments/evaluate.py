"""OOLONG Benchmark Evaluation."""

import os
import sys
import json
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from tqdm import tqdm
from src.controller import RLMController
from src.worker import RLMWorker
from src.repl import RLMREPL


def evaluate(num_examples=10, split='validation', controller_gpu="cuda:0", worker_gpu="cuda:0"):
    """Run RLM evaluation on OOLONG benchmark."""
    
    print("=" * 80)
    print(f"OOLONG EVALUATION - RLM")
    print("=" * 80)
    print(f"Examples: {num_examples}, Split: {split}")
    print(f"Controller: {controller_gpu}, Worker: {worker_gpu}")
    print("=" * 80)
    
    print("\n[1/3] Loading OOLONG dataset...")
    ds = load_dataset("oolongbench/oolong-real", "dnd", cache_dir=os.environ.get("HF_HOME"))
    examples = list(ds[split].select(range(num_examples)))
    
    print("\n[2/3] Loading models...")
    controller = RLMController(device=controller_gpu)
    worker = RLMWorker(device=worker_gpu)
    repl = RLMREPL(controller, worker)
    
    print("\n[3/3] Running evaluation...")
    results = []
    
    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        ex_id = example['id']
        question = example['question']
        context = example['context_window_text']
        gold_answer = example['answer']
        question_type = example['question_type']
        
        print(f"\nExample {i+1}/{len(examples)} [ID: {ex_id}]")
        print(f"Type: {question_type}, Question: {question[:80]}...")
        
        start_time = time.time()
        
        try:
            result = repl.run(task=question, document=context, verbose=False)
            elapsed = time.time() - start_time
            
            predicted_answer = result['answer']
            correct = str(gold_answer).strip().lower() == str(predicted_answer).strip().lower()
            
            print(f"Predicted: {predicted_answer}, Gold: {gold_answer}, Correct: {correct}, Time: {elapsed:.1f}s")
            
            results.append({
                'id': ex_id,
                'question_type': question_type,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'correct': correct,
                'success': result['success'],
                'time_seconds': elapsed
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'id': ex_id,
                'question_type': question_type,
                'error': str(e)
            })
    
    accuracy = sum(1 for r in results if r.get('correct', False)) / len(results) if results else 0
    
    print("\n" + "=" * 80)
    print(f"ACCURACY: {accuracy*100:.1f}% ({sum(1 for r in results if r.get('correct', False))}/{len(results)})")
    print("=" * 80)
    
    output_file = f"oolong_results_{split}_{num_examples}ex.json"
    with open(output_file, 'w') as f:
        json.dump({'accuracy': accuracy, 'results': results}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--controller_gpu', type=str, default='cuda:0')
    parser.add_argument('--worker_gpu', type=str, default='cuda:0')
    args = parser.parse_args()
    
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    evaluate(args.num_examples, args.split, args.controller_gpu, args.worker_gpu)
