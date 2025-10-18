# 🧰 CLI Reference

Entrypoint: `python crashtransformer.py <command> [options]`

## Commands

- `setup`: Interactive environment configuration
- `run`: Execute the crash analysis pipeline
- `train`: Fine-tune summarization models
- `prepare-data`: Build training datasets from pipeline outputs or synthetic data
- `clean-db`: Clear Neo4j database (removes all data)
- `docs`: Open local HTML documentation
- `help`: Show comprehensive help

## Global

```text
python crashtransformer.py            # interactive menu
python crashtransformer.py help       # detailed help
python crashtransformer.py docs       # build/serve docs
```

## setup

```bash
python crashtransformer.py setup [--interactive] [--create-env]
```

## run

```bash
python crashtransformer.py run \
  (--csv FILE | --xlsx FILE) \
  [--llm_provider PROVIDER] [--llm_model MODEL] [--neo4j_enabled] [--neo4j_uri URI] \
  [--log_level LEVEL] [--log_dir DIR] [--no_logs] \
  [--out_dir DIR] [--append] [--skip_llm] \
  [--model MODEL] [--batch_models MODELS...] [--cost_mode MODE]
```

## train

```bash
python crashtransformer.py train \
  --training_data FILE \
  [--model_name NAME] [--output_dir DIR] \
  [--num_epochs N] [--batch_size N] [--learning_rate FLOAT] \
  [--max_length N] [--max_target_length N] [--fp16] \
  [--early_stopping_patience N]
```

## prepare-data

```bash
python crashtransformer.py prepare-data \
  --source (pipeline|synthetic) \
  [--graphs_file FILE] [--summaries_file FILE] \
  --output FILE [--format (csv|jsonl)] [--num_examples N]
```

## Key Flags

### `--skip_llm`

Skip LLM extraction and use existing graphs from previous runs. Enables cost-efficient multi-model comparison by reusing graph structures.

**Usage:**

```bash
# Phase 1: Generate graphs (with LLM)
python crashtransformer.py run --csv crashes.csv --out_dir artifacts/llm_extraction

# Phase 2: Compare models (reuse graphs, no LLM calls)
python crashtransformer.py run --csv crashes.csv --model facebook/bart-base --out_dir artifacts/bart --skip_llm
python crashtransformer.py run --csv crashes.csv --model t5-base --out_dir artifacts/t5 --skip_llm
```

**Benefits:**

- **Cost Reduction**: ~80% savings on LLM API calls
- **Consistency**: Identical graphs across all model evaluations
- **Fair Comparison**: Models evaluated on same graph structures

## Examples

```bash
python crashtransformer.py run --csv crashes.csv
python crashtransformer.py run --csv crashes.csv --llm_provider anthropic
python crashtransformer.py run --csv crashes.csv --neo4j_enabled --log_level DEBUG
python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base
```
