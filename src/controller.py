"""RLM Controller - The planning agent that generates Python code."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re


class RLMController:
    """Controller model that generates Python code for recursive long-context reasoning."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-14B-Instruct", device="cuda:0"):
        self.device = device
        self.model_name = model_name
        
        print(f"[Controller] Loading {model_name} (4-bit) on {device}...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.environ.get("HF_HOME", None)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            cache_dir=os.environ.get("HF_HOME", None)
        )
        
        self.model.eval()
        print(f"[Controller] Loaded. VRAM: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        
    def plan(self, task_description, prompt_length, max_new_tokens=2048, temperature=0.3):
        """Generate Python code to solve a task using the REPL environment."""
        
        system_prompt = """You are an expert Python programmer with access to a REPL environment.

Available tools:
- `prompt`: A string variable containing a very long document ({prompt_len:,} characters)
- `sub_call(text_chunk, instruction)`: Analyze a text chunk and return results as a string

Your task: Write Python code to answer the user's question about the document.

Rules:
1. You cannot directly read `prompt` (it's too long). Only check `len(prompt)` or slice small chunks
2. Use `sub_call()` to analyze chunks. Examples:
   - sub_call(prompt[:5000], "List all character names")
   - sub_call(prompt[i:i+10000], "Count dice rolls in this section")
3. Store results in variables and aggregate them
4. Print your final answer clearly

Write clean, idiomatic Python code."""

        messages = [
            {"role": "system", "content": system_prompt.format(prompt_len=prompt_length)},
            {"role": "user", "content": f"Task: {task_description}\n\nWrite Python code to solve this:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|im_start|>assistant" in full_output:
            response = full_output.split("<|im_start|>assistant")[-1].strip()
        else:
            response = full_output[len(prompt):].strip()
        
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else response.strip()
