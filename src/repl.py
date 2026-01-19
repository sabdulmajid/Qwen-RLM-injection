"""RLM REPL Environment - Connects Controller and Worker for recursive reasoning."""

import os
import sys
from io import StringIO
from contextlib import redirect_stdout
import traceback


class RLMREPL:
    """Python REPL environment where the RLM operates."""
    
    def __init__(self, controller, worker):
        self.controller = controller
        self.worker = worker
        self.prompt = None
        
    def sub_call(self, text_chunk, instruction):
        """The magic function exposed to Controller-generated code."""
        print(f"[sub_call] Processing {len(text_chunk)} chars: '{instruction[:50]}...'")
        return self.worker.query(text_chunk, instruction, max_new_tokens=512, temperature=0.1)
    
    def run(self, task, document, verbose=True):
        """Execute full RLM pipeline."""
        
        self.prompt = document
        
        if verbose:
            print("=" * 80)
            print(f"Task: {task}")
            print(f"Document: {len(document):,} chars (~{len(document)//4:,} tokens)")
            print("=" * 80)
        
        if verbose:
            print("\n[1/2] Controller planning...")
        
        code = self.controller.plan(
            task_description=task,
            prompt_length=len(document),
            temperature=0.3
        )
        
        if verbose:
            print("\n[Generated Code]")
            print("-" * 80)
            print(code)
            print("-" * 80)
        
        if verbose:
            print("\n[2/2] Executing code...")
        
        exec_globals = {
            'prompt': self.prompt,
            'sub_call': self.sub_call,
            'len': len,
            'range': range,
            'print': print,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'sum': sum,
            'min': min,
            'max': max,
        }
        
        output_buffer = StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                exec(code, exec_globals)
            
            execution_log = output_buffer.getvalue()
            
            if verbose:
                print("\n[Execution Output]")
                print("-" * 80)
                print(execution_log)
                print("-" * 80)
            
            lines = execution_log.strip().split('\n')
            answer = lines[-1] if lines else "No output"
            
            return {
                'answer': answer,
                'code': code,
                'execution_log': execution_log,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"EXECUTION ERROR:\n{traceback.format_exc()}"
            
            if verbose:
                print("\n[ERROR]")
                print("-" * 80)
                print(error_msg)
                print("-" * 80)
            
            return {
                'answer': f"Error: {str(e)}",
                'code': code,
                'execution_log': error_msg,
                'success': False
            }
