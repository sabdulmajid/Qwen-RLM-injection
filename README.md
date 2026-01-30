# Recursive Language Models: Dense Reasoning at Scale

**Investigating information flow and reasoning capabilities in Recursive Language Models (RLMs) on ultra-long contexts**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.03908)
[![Dataset](https://img.shields.io/badge/Dataset-OOLONG-blue)](https://huggingface.co/datasets/oolongbench/oolong-real)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What is This?

Traditional language models fail catastrophically on ultra-long contexts (>100K tokens). Even state-of-the-art models with "infinite" context windows suffer from **context rot** — their reasoning degrades as context grows.

**Recursive Language Models (RLMs)** take a different approach: instead of processing everything at once, they use a **Controller-Worker architecture** where:

- **Controller** (the "Brain"): Writes Python code to strategically query parts of the document
- **Worker** (the "Eye"): Processes text chunks on-demand and returns focused summaries  
- **REPL Environment**: Executes the Controller's code with access to `sub_call()` for recursive queries

This project implements and analyzes RLMs on the **OOLONG benchmark** — a dense reasoning task requiring aggregation across Critical Role D&D transcripts spanning up to **1.2 million tokens**.

---

## 🔬 Key Insights

### The Problem: Context Rot in Long Documents

When asked to count dice rolls across 24 episodes of D&D gameplay (~400K tokens), standard LLMs:
- Miss critical information scattered throughout
- Get distracted by irrelevant details
- Produce hallucinated counts

### The RLM Solution: Programmatic Reasoning

Instead of reading everything at once, the RLM Controller generates code like:

```python
total_rolls = 0
chunk_size = 50000

for i in range(0, len(prompt), chunk_size):
    chunk = prompt[i:i+chunk_size]
    result = sub_call(chunk, "Count all dice rolls in this section")
    total_rolls += int(result)

print(f"Total rolls: {total_rolls}")
```

This **recursive decomposition** mirrors how humans solve complex problems: break it down, delegate subtasks, aggregate results.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Controller (Qwen-Coder-14B-Int4)      │
│  • Sees: Task + len(document)           │
│  • Generates: Python code               │
│  • Hardware: GPU 0 (NVIDIA RTX A4500)  │
└─────────────────────────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  REPL Environment    │
         │  • prompt variable   │
         │  • sub_call() func   │
         └─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Worker (Qwen-7B-FP16)                 │
│  • Receives: Text chunks                │
│  • Returns: Extracted information       │
│  • Hardware: GPU 1 (NVIDIA RTX A4500)  │
└─────────────────────────────────────────┘
```

**Why This Split?**
- **Controller** in 4-bit: Efficient code generation, fits in ~10GB
- **Worker** in FP16: Clean activations for mechanistic interpretability analysis
- **Dual-GPU**: Parallel processing, ~30GB total VRAM usage

---

## 📊 Dataset: OOLONG-real (D&D)

- **12,067 examples** (5,995 validation + 6,072 test)
- **Average context**: 383K tokens (~1.5M characters)
- **Maximum context**: 1.2M tokens (~4.8M characters)
- **Task types**:
  - 50% Multi-episode spell counting
  - 43% Multi-episode roll counting
  - 7% Single-episode queries

**Why It's Hard**: Answers require **dense reasoning** — aggregating information from every page. You can't skip or sample. Perfect stress test for RLMs.

---

## 🚀 Getting Started

### Prerequisites

```bash
# Hardware
2x NVIDIA RTX A4500 (20GB each) or equivalent

# Software
Python 3.13+
CUDA 12.8+
PyTorch 2.9+
```

### Installation

```bash
# Clone repository
git clone https://github.com/sabdulmajid/Qwen-RLM-injection.git
cd Qwen-RLM-injection

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set cache directory (optional, for HPC clusters)
export HF_HOME=/path/to/cache
```

### Quick Test

```bash
# Load OOLONG dataset
python scripts/load_dataset.py

# Run basic RLM test
python src/rlm_pipeline.py
```

### Run Evaluation

```bash
# Evaluate on 10 examples
python experiments/evaluate.py --num_examples 10 --split validation

# Full benchmark
sbatch scripts/run_evaluation.sbatch
```

---

## 🔍 Research Questions

1. **Information Flow**: How does information propagate through recursive `sub_call()` boundaries?
2. **Planning Quality**: What coding patterns emerge in the Controller's strategies?
3. **Context Utilization**: Does the RLM actually read relevant sections, or does it hallucinate?
4. **Failure Modes**: When does recursive decomposition break down?

Future work will add **mechanistic interpretability** analysis using activation patching to trace information loss through the recursive stack.

---

## 📈 Preliminary Results

*Evaluation in progress. Results will be added upon completion.*

Key metrics:
- Accuracy on OOLONG validation set
- Average number of `sub_call()` invocations per task
- Code execution success rate
- Runtime per example

---

## 🛠️ Development

### Running on SLURM Cluster

```bash
# Submit job to dualcard partition
sbatch scripts/run_evaluation.sbatch

# Monitor progress
watch -n 5 'squeue -u $USER && tail -20 rlm_*.out'
```

### Modifying the Architecture

- **Different models**: Edit `model_name` in `src/controller.py` or `src/worker.py`
- **Quantization**: Adjust `BitsAndBytesConfig` for different precision
- **REPL sandbox**: Modify allowed functions in `src/repl.py`

---

## 📚 Citation

If you use this code or findings, please cite the original RLM paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, et al.},
  journal={arXiv preprint arXiv:2501.03908},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- **MIT CSAIL** for the RLM paradigm
- **OOLONG Team** for the benchmark dataset
- **Qwen Team** for open-source models
- Compute resources provided by ECE Department, University of Toronto

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Questions?** Open an issue or reach out via [LinkedIn](https://linkedin.com/in/sabdulmajid)
