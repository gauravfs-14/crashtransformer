# 🧰 CLI Reference

Entrypoint: `python crashtransformer.py <command> [options]`

## Commands

- `setup`: Interactive environment configuration
- `run`: Execute the crash analysis pipeline
- `train`: Fine-tune summarization models
- `prepare-data`: Build training datasets from pipeline outputs or synthetic data
- `docs`: Build HTML docs from `docs/*.md` and serve locally
- `validate`: Validate current configuration
- `install`: Install required dependencies
- `examples`: Show usage examples
- `help`: Show help

## Global

```text
python crashtransformer.py            # interactive menu
python crashtransformer.py help       # detailed help
python crashtransformer.py docs       # build/serve docs
```

## setup

```bash
python crashtransformer.py setup [--interactive] [--create-env] [--validate] [--install] [--examples]
```

## run

```bash
python crashtransformer.py run \
  (--csv FILE | --xlsx FILE) \
  [--llm_provider PROVIDER] [--llm_model MODEL] [--neo4j_enabled] [--neo4j_uri URI] \
  [--log_level LEVEL] [--log_dir DIR] [--no_logs] \
  [--out_dir DIR] [--append] \
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

## Examples

```bash
python crashtransformer.py run --csv crashes.csv
python crashtransformer.py run --csv crashes.csv --llm_provider anthropic
python crashtransformer.py run --csv crashes.csv --neo4j_enabled --log_level DEBUG
python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base
```
