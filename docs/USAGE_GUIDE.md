# 📖 CrashTransformer Usage Guide

> **Suggested reading path**: `00_START_HERE.md` → `ENVIRONMENT_SETUP.md` → `INTERACTIVE_SETUP.md` → this guide

Complete guide for using CrashTransformer's main entry point with comprehensive examples and best practices.

## 🚀 Quick Start

```bash
# 1. Setup environment (first time only)
python crashtransformer.py setup

# 2. Run the pipeline
python crashtransformer.py run --csv crashes.csv
```

## 📋 Command Reference

### **Main Commands**

| Command | Description | Usage | Options |
|---------|-------------|-------|---------|
| `setup` | Environment configuration | `python crashtransformer.py setup` | `--interactive`, `--create-env` |
| `run` | Execute pipeline | `python crashtransformer.py run --csv file.csv` | `--llm_provider`, `--neo4j_enabled` |
| `train` | Fine-tune models | `python crashtransformer.py train --training_data data.csv` | `--model_name`, `--output_dir` |
| `prepare-data` | Prepare training data | `python crashtransformer.py prepare-data --source pipeline` | `--output_dir`, `--format` |
| `validate` | Check configuration | `python crashtransformer.py validate` | `--verbose` |
| `install` | Install dependencies | `python crashtransformer.py install` | `--dev`, `--upgrade` |
| `examples` | Show examples | `python crashtransformer.py examples` | `--provider`, `--model` |
| `help` | Show help | `python crashtransformer.py help` | `--command` |

### **Interactive Menu**

```bash
# Run without arguments for interactive menu
python crashtransformer.py

# Choose from:
# 1. Setup environment (recommended for first time)
# 2. Run pipeline
# 3. Show help
# 4. Exit
```

## 🔧 Environment Setup

See `ENVIRONMENT_SETUP.md` and `INTERACTIVE_SETUP.md`.

### **Interactive Setup (Recommended)**

```bash
# Guided setup process
python crashtransformer.py setup

# Direct interactive setup
python crashtransformer.py setup --interactive
```

### **Setup Options**

```bash
# Create .env from template
python crashtransformer.py setup --create-env

# Validate configuration
python crashtransformer.py validate

# Install dependencies
python crashtransformer.py install

# Show usage examples
python crashtransformer.py examples
```

## 🚀 Pipeline Execution

### **Basic Usage**

```bash
# Use environment configuration
python crashtransformer.py run --csv crashes.csv

# Override LLM provider
python crashtransformer.py run --csv crashes.csv --llm_provider anthropic

# Enable Neo4j storage
python crashtransformer.py run --csv crashes.csv --neo4j_enabled

# Debug mode
python crashtransformer.py run --csv crashes.csv --log_level DEBUG
```

### **Advanced Usage**

```bash
# Multiple summarization models
python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base

# Custom output directory
python crashtransformer.py run --csv crashes.csv --out_dir results

# Cost tracking mode
python crashtransformer.py run --csv crashes.csv --cost_mode api

# Custom log directory
python crashtransformer.py run --csv crashes.csv --log_dir custom_logs
```

### **Optimized Two-Phase Pipeline (Cost Efficient)**

For comparing multiple models while minimizing LLM costs:

#### **Phase 1: Generate graphs and LLM summaries (once)**

```bash
# Generate graphs and LLM summaries
python crashtransformer.py run --csv crashes.csv --model facebook/bart-base --out_dir artifacts/llm_extraction
```

#### **Phase 2: Compare models using existing graphs**

```bash
# Compare different models using existing graphs (no LLM calls)
python crashtransformer.py run --csv crashes.csv --model facebook/bart-base --out_dir artifacts/bart_comparison --skip_llm
python crashtransformer.py run --csv crashes.csv --model t5-base --out_dir artifacts/t5_comparison --skip_llm
```

#### **Benefits of Two-Phase Approach**

| Benefit | Description | Impact |
|---------|-------------|---------|
| **💰 Cost Reduction** | Reduces LLM costs by ~80% for multi-model comparisons | Significant savings on large datasets |
| **🔄 Consistent Graphs** | Ensures identical graph structures across all model evaluations | Fair comparison between models |
| **⚡ Faster Processing** | Skip expensive LLM calls in Phase 2 | Faster model comparison |
| **📊 Better Analysis** | Isolates summarization performance from extraction quality | Clearer insights into model differences |

> **Note**: See `CLI_REFERENCE.md` for complete command/flag reference.

## 📊 Input Requirements

See `DATA_SPEC.md` for full details.

### **CSV Format**

Required columns:

- `Crash_ID`: Unique identifier
- `Latitude`, `Longitude`: Geographic coordinates
- `CrashDate`, `DayOfWeek`, `CrashTime`: Temporal data
- `County`, `City`: Location information
- `SAE_Autonomy_Level`: Vehicle autonomy level
- `Crash_Severity`: Severity classification
- `Narrative`: Crash description text

### **Example CSV**

```csv
Crash_ID,Latitude,Longitude,CrashDate,DayOfWeek,CrashTime,County,City,SAE_Autonomy_Level,Crash_Severity,Narrative
19955047,26.15526348,-97.99060556,2023-01-15,Sunday,14:30,Hidalgo,Weslaco,Level 0,Not Injured,"Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."
```

## 📈 Outputs Generated

See `OUTPUTS.md` for structure and examples.

### **File Outputs**

- `artifacts/crash_graphs.jsonl` - Structured graph data
- `artifacts/crash_summaries.jsonl` - Generated summaries
- `artifacts/summaries_metrics.csv` - Quality metrics
- `artifacts/cost_report.json` - Cost analysis
- `logs/crashtransformer-*.log` - Processing logs

### **Neo4j Database (if enabled)**

- Nodes: Crash, Vehicle, Location, Event, Summary
- Relationships: HAS_ENTITY, HAS_EVENT, CAUSES, PARTICIPATED_IN, HIT, HAS_SUMMARY

## 🔒 Configuration

See `ENVIRONMENT_SETUP.md` and `SECURITY.md`.

## 🎯 Use Cases

- Insurance Claims, Transportation Safety, Research Analysis, Compliance Reporting

## 🐛 Troubleshooting

See `TROUBLESHOOTING_FAQ.md`.

## 📚 Additional Resources

- `PROVIDERS_GUIDE.md`, `NEO4J_GUIDE.md`, `COST_PERFORMANCE.md`, `FINE_TUNING_GUIDE.md`
