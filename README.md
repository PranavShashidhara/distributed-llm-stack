# Distributed LLM Orchestration Platform (D-LOP)

A distributed fine-tuning pipeline for Llama 3.1 8B across multiple GPUs using Fully Sharded Data Parallel (FSDP).

---

## Usage

**1. Bootstrap the environment**
```bash
chmod +x setup.sh && ./setup.sh
```

**2. Activate the virtual environment**
```bash
source venv/bin/activate
```

**3. Add your HuggingFace token to a `.env` file**
```env
HUGGINGFACEHUB_API_TOKEN=hf_...
RUNPOD_API_KEY=...
```

**4. Run the pipeline**
```bash
python preprocessing.py
python tokenize_master.py
accelerate launch --config_file accelerate_config.yaml train_fsdp.py
```

---

## Scripts

**`setup.sh`** — Bootstraps the environment. Creates a virtual environment, installs dependencies, and sets up the required directories.

**`preprocessing.py`** — Pulls three HuggingFace datasets (FineTome, Gretel SQL, UltraChat), normalizes them into a unified ChatML format, and saves the merged dataset to disk.

**`tokenize_master.py`** — Applies the LLaMA 3.1 chat template to the merged dataset and tokenizes it in parallel, producing padded sequences ready for training.

**`train_fsdp.py`** — Runs supervised fine-tuning using FSDP to shard the model across GPUs, with Liger Triton kernels for memory efficiency and gradient checkpointing enabled.

**`accelerate_config.yaml`** — Configures the Accelerate launcher for FSDP with full sharding, BF16 mixed precision, and auto-wrapping of LLaMA decoder layers.

**`provision.py`** — Programmatically spins up and tears down GPU clusters on RunPod via their API.

---

## Structure

```
infra/
├── setup.sh
├── requirements.txt
├── accelerate_config.yaml
├── preprocessing.py
├── tokenize_master.py
├── train_fsdp.py
├── provision.py
├── checkpoints/
└── data/
```