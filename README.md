# Distributed LLM Orchestration Platform (D-LOP)

A fine-tuning pipeline for Llama 3.1 8B-Instruct using Unsloth + Liger Kernels, designed to run on Google Colab with A100/H100 GPUs.

---

## Project Structure

```
infra/
├── main_fine_tuning_pipeline.ipynb   # Interactive fine-tuning pipeline (Colab)
├── README.md
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Environment bootstrap script
├── checkpoints/                      # Saved LoRA adapters & model checkpoints
└── data/
    ├── unified/master_dataset        # Preprocessed ChatML dataset (Arrow format)
    └── tokenized_master              # Tokenized dataset ready for training
```

---

## Usage

### Running the Notebook (Colab)

`main_fine_tuning_pipeline.ipynb` is the primary entrypoint. Open it in Google Colab and run cells top to bottom.

**What it does:**
- Loads and preprocesses the Gretel SQL dataset into ChatML format
- Tokenizes with the Llama-3 chat template
- Fine-tunes Llama 3.1 8B-Instruct with Unsloth + Liger Kernels (4-bit QLoRA)
- Evaluates SQL generation quality on holdout samples
- Exports a merged GGUF (Q4_K_M) model for deployment

**Setup:**
1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime: `Runtime → Change runtime type → A100 or H100`
3. Add `HF_TOKEN` to Colab Secrets (🔑 left sidebar)
4. Run all cells top to bottom

---

### Local Environment (Optional)

```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
```

---

## Recommended Hyperparameters

All hyperparameters are configured in Cell 4 of the notebook.

| Parameter | A100 40GB | H100 95GB | Notes |
|-----------|-----------|-----------|-------|
| `BATCH_SIZE` | 8 | 64 | Linear VRAM cost — safe to push high |
| `MAX_SEQ_LENGTH` | 2048 | 2048 | Keep at 2048 — quadratic cost |
| `LORA_R` | 64 | 128 | Higher = more adapter capacity |
| `LEARNING_RATE` | 2e-4 | 8e-4 | Scale with sqrt(batch_size) |
| `GRAD_ACCUM` | 4 | 1 | Reduce as batch size increases |

| Runtime | VRAM | Est. Training Time |
|---------|------|--------------------|
| A100 40GB | 40GB | ~2 hrs |
| H100 95GB | 95GB | ~25 mins |

> ⚠️ Do not increase `MAX_SEQ_LENGTH` beyond 2048 without reducing batch size — attention is O(n²) and will multiply training time dramatically.
- The dataset's longest entry is under 2,100 characters (~525 tokens at ~4 characters per token), making `MAX_SEQ_LENGTH=1024` a safe and efficient choice for this training run.

---

## Dataset

Uses the **Gretel SQL** dataset by default:

| Dataset | Samples | Use case |
|---------|---------|----------|
| `gretelai/synthetic_text_to_sql` | ~100k | SQL generation (default) |

---

## Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| LoRA adapter | `sql-genie-lora/` | Fine-tuned adapter weights |
| GGUF model | `sql-genie-gguf/*.gguf` | Q4_K_M quantized for deployment |
| Checkpoints | `checkpoints/` | Mid-training saves every 200 steps |