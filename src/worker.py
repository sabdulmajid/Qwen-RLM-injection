"""RLM Worker - The reading agent that processes text chunks."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


class RLMWorker:
    """Worker model that reads text chunks and answers queries."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:0"):
        self.device = device
        self.model_name = model_name
        
        print(f"[Worker] Loading {model_name} on {device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.environ.get("HF_HOME", None)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            cache_dir=os.environ.get("HF_HOME", None)
        )
        
        self.model.eval()
        print(f"[Worker] Loaded. VRAM: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        
    def query(self, text_chunk, instruction, max_new_tokens=512, temperature=0.7):
        """Process a text chunk with a specific instruction."""
        
        messages = [
            {"role": "system", "content": "You are a careful reader. Answer questions based only on the provided text."},
            {"role": "user", "content": f"{instruction}\n\nText:\n{text_chunk}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|im_start|>assistant" in full_output:
            response = full_output.split("<|im_start|>assistant")[-1].strip()
        else:
            response = full_output[len(prompt):].strip()
        
        return response
