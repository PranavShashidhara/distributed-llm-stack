import os
import multiprocess as mp  # Use multiprocess for better compatibility
from dotenv import load_dotenv
from datasets import load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login

def run_tokenization(
    model_id="meta-llama/Llama-3.1-8B",
    input_path="data/unified/master_dataset",
    output_path="data/tokenized_master",
    env_path=".env"
):
    # 1. PREVENT HANGS: Force 'spawn' start method
    # This ensures each process gets its own clean memory space
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    load_dotenv(env_path)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    # Llama-3 Template Injection
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] == 'system' %}"
        "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_from_disk(input_path)
    
    def tokenize_fn(batch):
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in batch["messages"]
        ]
        return tokenizer(
            texts, 
            truncation=True, 
            max_length=2048, 
            padding="max_length"
        )

    # 2. OPTIMIZED CORES: Using 8 cores leaves 8 for your OS to stay responsive
    num_cores = 8 
    print(f"--- Starting tokenization (Stable Mode: {num_cores} cores) ---")
    
    tokenized_ds = ds.map(
        tokenize_fn, 
        batched=True, 
        batch_size=500,        # Smaller batches for smoother processing
        num_proc=num_cores,
        writer_batch_size=500, # Flushes to disk twice as often
        remove_columns=ds.column_names
    )
    
    print(f"--- Saving final result to {output_path} ---")
    tokenized_ds.save_to_disk(output_path)
    print(" Success! Your laptop survived.")
    
    return tokenized_ds

if __name__ == "__main__":
    run_tokenization()