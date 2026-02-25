import os
from datasets import load_dataset, concatenate_datasets

def standardize_format(example, source):
    """
    Normalizes 2026 dataset schemas into a unified ChatML-ready list.
    Corrects for key variations in Gretel SQL and UltraChat.
    """
    messages = []
    
    if source == "finetome":
        # FineTome uses 'conversations' with 'from' and 'value'
        for msg in example["conversations"]:
            role = "assistant" if msg["from"] == "gpt" else msg["from"]
            role = "user" if role == "human" else role
            messages.append({"role": role, "content": msg["value"]})
            
    elif source == "sql":
        # Correcting keys for gretelai/synthetic_text_to_sql
        # Input is 'sql_prompt', output is 'sql'
        system_content = (
            "You are a SQL expert. Use the provided context to generate "
            "accurate SQL queries."
        )
        if "sql_context" in example and example["sql_context"]:
            system_content += f" Context: {example['sql_context']}"
            
        messages.append({"role": "system", "content": system_content})
        # Use .get() to prevent further KeyErrors if columns shift again
        messages.append({"role": "user", "content": example.get("sql_prompt", "")})
        messages.append({"role": "assistant", "content": example.get("sql", "")})
        
    elif source == "ultrachat":
        # UltraChat 200k uses 'messages' directly
        messages = example["messages"]

    return {"messages": messages}

def main():
    target_dir = "data/unified"
    os.makedirs(target_dir, exist_ok=True)
    
    all_datasets = []

    # 1. FineTome
    print("Fetching mlabonne/FineTome-100k...")
    ds_ft = load_dataset("mlabonne/FineTome-100k", split="train")
    ds_ft = ds_ft.map(lambda x: standardize_format(x, "finetome"), remove_columns=ds_ft.column_names)
    all_datasets.append(ds_ft)

    # 2. Gretel SQL
    print("Fetching gretelai/synthetic_text_to_sql...")
    ds_sql = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    # Mapping with corrected keys
    ds_sql = ds_sql.map(lambda x: standardize_format(x, "sql"), remove_columns=ds_sql.column_names)
    all_datasets.append(ds_sql)

    # 3. UltraChat
    print("Fetching HuggingFaceH4/ultrachat_200k...")
    ds_uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds_uc = ds_uc.map(lambda x: standardize_format(x, "ultrachat"), remove_columns=ds_uc.column_names)
    all_datasets.append(ds_uc)

    # 4. Merge and Save
    print("Merging and shuffling master dataset...")
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    
    save_path = os.path.join(target_dir, "master_dataset")
    combined.save_to_disk(save_path)
    
    print("-" * 30)
    print(f"Success. Total samples: {len(combined)}")
    print(f"Artifact stored in: {save_path}")

if __name__ == "__main__":
    main()