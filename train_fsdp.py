import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1. Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B"
DATASET_PATH = "data/tokenized_master"
OUTPUT_DIR = "models/llama-3.1-8b-custom"

def train():
    # 2. Apply Liger Kernels (The 2026 Performance Secret)
    # This replaces standard Llama layers with highly optimized Triton kernels
    apply_liger_kernel_to_llama()

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Pre-tokenized Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)

    # 5. Training Arguments (Optimized for 2-GPU FSDP)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text", # Not strictly used since data is pre-tokenized
        max_seq_length=2048,
        per_device_train_batch_size=4,   # Total batch size will be 8
        gradient_accumulation_steps=4,  # Effective batch size = 32
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,                      # Use bfloat16 for Ampere GPUs
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=1,
        report_to="none",               # Change to "wandb" if you have an account
        gradient_checkpointing=True,    # Save memory at the cost of slight speed
        fsdp="full_shard auto_wrap",    # Critical for 2-GPU splitting
        fsdp_config={
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        },
    )

    # 6. Initialize Model
    # We load the model in full precision; FSDP handles the sharding/mixed precision
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False # Must be False for Gradient Checkpointing
    )

    # 7. Start Training
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(" Launching training...")
    trainer.train()
    
    # 8. Save Final Model
    trainer.save_model(OUTPUT_DIR)
    print(f" Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()