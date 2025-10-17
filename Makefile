# Makefile for CrashTransformer end-to-end workflow

.PHONY: all test setup data baseline baseline-full baseline-only llm-pipeline llm-pipeline-full llm-pipeline-with-baseline prepare-training-data prepare-training-data-full finetune-bart finetune-bart-full finetune-t5 finetune-t5-full run-finetuned-bart run-finetuned-bart-full run-finetuned-t5 run-finetuned-t5-full visualize-bart visualize-t5 clean help

# Stop on any error
.SHELLFLAGS := -e -o pipefail

PY ?= uv run
ROOT := $(abspath .)

# Data
DATA_XLSX := $(ROOT)/data/data.xlsx
TEST_CSV := $(ROOT)/data/test_data_5rows.csv
SHEET ?= Narr_CrLev

# Centralized artifacts directory
ARTIFACTS_DIR := $(ROOT)/artifacts

# Baseline outputs (test)
BASE_OUT := $(ARTIFACTS_DIR)/baseline
BART_BASE_DIR := $(BASE_OUT)/facebook_bart-base
T5_BASE_DIR := $(BASE_OUT)/t5-base

# Baseline outputs (full)
BASE_OUT_FULL := $(ARTIFACTS_DIR)/baseline_full
BART_FULL_DIR := $(BASE_OUT_FULL)/facebook_bart-base
T5_FULL_DIR := $(BASE_OUT_FULL)/t5-base

# Fine-tuned model dirs
FT_BART_DIR := $(ARTIFACTS_DIR)/fine_tuned_models/bart
FT_T5_DIR := $(ARTIFACTS_DIR)/fine_tuned_models/t5
FT_BART_MODEL := $(FT_BART_DIR)/final_model
FT_T5_MODEL := $(FT_T5_DIR)/final_model

# Finetuned run outputs (test)
FT_BART_OUT := $(ARTIFACTS_DIR)/finetuned_bart
FT_T5_OUT := $(ARTIFACTS_DIR)/finetuned_t5

# Finetuned run outputs (full)
FT_BART_OUT_FULL := $(ARTIFACTS_DIR)/finetuned_bart_full
FT_T5_OUT_FULL := $(ARTIFACTS_DIR)/finetuned_t5_full

# Training data files
TRAINING_DATA_DIR := $(ARTIFACTS_DIR)/training_data
TRAINING_BART := $(TRAINING_DATA_DIR)/training_data_bart.csv
TRAINING_T5 := $(TRAINING_DATA_DIR)/training_data_t5.csv
TRAINING_BART_FULL := $(TRAINING_DATA_DIR)/training_data_bart_full.csv
TRAINING_T5_FULL := $(TRAINING_DATA_DIR)/training_data_t5_full.csv

# Default target runs on FULL data (optimized for cost efficiency)
all: setup llm-pipeline-full baseline-full-optimized prepare-training-data-full finetune-bart-full finetune-t5-full run-finetuned-bart-full run-finetuned-t5-full

# Test target runs on 5-row sample (optimized for cost efficiency)
test: setup data llm-pipeline baseline-only prepare-training-data finetune-bart finetune-t5 run-finetuned-bart run-finetuned-t5

help:
	@echo "Targets:"
	@echo "  all                  - Full pipeline on complete dataset (XLSX, cost-optimized)"
	@echo "  test                 - Full pipeline on 5-row test CSV (cost-optimized)"
	@echo "  setup                - Setup environment"
	@echo "  data                 - Create 5-row test dataset in data/"
	@echo ""
	@echo "LLM Pipeline (Graph Generation):"
	@echo "  llm-pipeline         - Run LLM pipeline to generate crash graphs (test)"
	@echo "  llm-pipeline-full    - Run LLM pipeline to generate crash graphs (full)"
	@echo "  llm-pipeline-with-baseline - Run LLM + baseline models in one go (test)"
	@echo ""
	@echo "Baseline Models:"
	@echo "  baseline             - Run baseline (BART/T5) on 5-row CSV"
	@echo "  baseline-full        - Run baseline (BART/T5) on full XLSX"
	@echo "  baseline-only        - Run baseline models using existing graphs (cost-efficient)"
	@echo "  baseline-full-optimized - Run baseline models using existing graphs (full, cost-efficient)"
	@echo ""
	@echo "Training Data & Fine-tuning:"
	@echo "  prepare-training-data- Build training CSVs from test baseline outputs"
	@echo "  prepare-training-data-full - Build training CSVs from full baseline outputs"
	@echo "  finetune-bart        - Fine-tune facebook/bart-base (test)"
	@echo "  finetune-bart-full   - Fine-tune facebook/bart-base (full)"
	@echo "  finetune-t5          - Fine-tune t5-base (test)"
	@echo "  finetune-t5-full     - Fine-tune t5-base (full)"
	@echo ""
	@echo "Fine-tuned Model Runs:"
	@echo "  run-finetuned-bart   - Run pipeline using fine-tuned BART (test)"
	@echo "  run-finetuned-bart-full - Run pipeline using fine-tuned BART (full)"
	@echo "  run-finetuned-t5     - Run pipeline using fine-tuned T5 (test)"
	@echo "  run-finetuned-t5-full - Run pipeline using fine-tuned T5 (full)"
	@echo ""
	@echo "Utilities:"
	@echo "  visualize-bart       - Create visualizations for BART training"
	@echo "  visualize-t5         - Create visualizations for T5 training"
	@echo "  clean                - Remove generated artifacts and training data"
	@echo ""
	@echo "Cost Optimization:"
	@echo "  💡 Use *-optimized targets to save ~80% on LLM API costs"
	@echo "  💡 Phase 1: Run llm-pipeline to generate graphs once"
	@echo "  💡 Phase 2: Run baseline-only to reuse graphs for model comparison"

setup:
	$(PY) crashtransformer.py setup

data: $(TEST_CSV)

$(TEST_CSV): $(DATA_XLSX)
	@echo "Creating 5-row test CSV at $(TEST_CSV) from $(DATA_XLSX) [sheet=$(SHEET)]"
	@mkdir -p $(dir $(TEST_CSV))
	@echo "import pandas as pd; df = pd.read_excel('$(DATA_XLSX)', sheet_name='$(SHEET)'); df.head(5).to_csv('$(TEST_CSV)', index=False); print('Wrote $(TEST_CSV)')" > /tmp/create_test_data.py
	$(PY) /tmp/create_test_data.py
	@rm -f /tmp/create_test_data.py

# ---------- LLM Pipeline (crash graph generation) ----------

llm-pipeline: data
	@echo "🤖 Running LLM pipeline to generate crash graphs from test data..."
	@mkdir -p $(BASE_OUT)
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --out_dir $(BASE_OUT) \
	  --neo4j_enabled \
	  --log_level INFO || (echo "❌ LLM pipeline failed!" && exit 1)

llm-pipeline-with-baseline: data
	@echo "🤖 Running LLM pipeline with baseline models in one go..."
	@mkdir -p $(BASE_OUT)
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --batch_models facebook/bart-base t5-base \
	  --out_dir $(BASE_OUT) \
	  --neo4j_enabled \
	  --log_level INFO || (echo "❌ LLM pipeline with baseline failed!" && exit 1)

llm-pipeline-full:
	@echo "🤖 Running LLM pipeline to generate crash graphs from full dataset..."
	@mkdir -p $(BASE_OUT_FULL)
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --out_dir $(BASE_OUT_FULL) \
	  --neo4j_enabled \
	  --log_level INFO || (echo "❌ LLM pipeline failed!" && exit 1)

# ---------- Test (5-row) pipeline ----------

baseline: llm-pipeline
	@echo "📊 Running baseline summarization models on generated crash graphs..."
	@mkdir -p $(BASE_OUT)
	@echo "🤖 Running BART baseline model..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model facebook/bart-base \
	  --neo4j_enabled \
	  --out_dir $(BASE_OUT) \
	  --log_level INFO || (echo "❌ BART baseline failed!" && exit 1)
	@echo "🤖 Running T5 baseline model..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model t5-base \
	  --neo4j_enabled \
	  --out_dir $(BASE_OUT) \
	  --log_level INFO || (echo "❌ T5 baseline failed!" && exit 1)

baseline-only:
	@echo "📊 Running baseline summarization models on existing crash graphs (cost-efficient)..."
	@echo "💡 This reuses graphs from previous LLM runs, saving ~80% on API costs"
	@mkdir -p $(BASE_OUT)
	@echo "🤖 Running BART baseline model (reusing existing graphs)..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model facebook/bart-base \
	  --skip_llm \
	  --out_dir $(BASE_OUT) \
	  --log_level INFO || (echo "❌ BART baseline failed!" && exit 1)
	@echo "🤖 Running T5 baseline model (reusing existing graphs)..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model t5-base \
	  --skip_llm \
	  --out_dir $(BASE_OUT) \
	  --log_level INFO || (echo "❌ T5 baseline failed!" && exit 1)

prepare-training-data: baseline
	@mkdir -p $(TRAINING_DATA_DIR)
	@echo "📊 Preparing Training Data..."
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(BART_BASE_DIR)/crash_graphs.jsonl \
	  --summaries_file $(BART_BASE_DIR)/crash_summaries.jsonl \
	  --output $(TRAINING_BART) \
	  --format csv || (echo "❌ BART training data preparation failed!" && exit 1)
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(T5_BASE_DIR)/crash_graphs.jsonl \
	  --summaries_file $(T5_BASE_DIR)/crash_summaries.jsonl \
	  --output $(TRAINING_T5) \
	  --format csv || (echo "❌ T5 training data preparation failed!" && exit 1)

finetune-bart: prepare-training-data
	@mkdir -p $(FT_BART_DIR)
	@echo "🎯 Running Model Training (BART)..."
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART) \
	  --model_name facebook/bart-base \
	  --output_dir $(FT_BART_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ BART fine-tuning failed!" && exit 1)
	@echo "📊 Creating training visualizations..."
	@$(PY) create_visualizations.py $(FT_BART_DIR) || (echo "❌ BART visualization failed!" && exit 1)

finetune-t5: prepare-training-data
	@mkdir -p $(FT_T5_DIR)
	@echo "🎯 Running Model Training (T5)..."
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_T5) \
	  --model_name t5-base \
	  --output_dir $(FT_T5_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ T5 fine-tuning failed!" && exit 1)
	@echo "📊 Creating training visualizations..."
	@$(PY) create_visualizations.py $(FT_T5_DIR) || (echo "❌ T5 visualization failed!" && exit 1)

run-finetuned-bart: finetune-bart data
	@echo "🚀 Running Fine-tuned BART Model..."
	@mkdir -p $(FT_BART_OUT)
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model facebook/bart-base \
	  --fine_tuned_model $(FT_BART_MODEL) \
	  --neo4j_enabled \
	  --out_dir $(FT_BART_OUT) \
	  --log_level INFO || (echo "❌ Fine-tuned BART run failed!" && exit 1)

run-finetuned-t5: finetune-t5 data
	@echo "🚀 Running Fine-tuned T5 Model..."
	@mkdir -p $(FT_T5_OUT)
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model t5-base \
	  --fine_tuned_model $(FT_T5_MODEL) \
	  --neo4j_enabled \
	  --out_dir $(FT_T5_OUT) \
	  --log_level INFO || (echo "❌ Fine-tuned T5 run failed!" && exit 1)

# ---------- Full-data pipeline ----------

baseline-full: llm-pipeline-full
	@echo "📊 Running baseline summarization models on generated crash graphs (full dataset)..."
	@mkdir -p $(BASE_OUT_FULL)
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --batch_models facebook/bart-base t5-base \
	  --neo4j_enabled \
	  --out_dir $(BASE_OUT_FULL) \
	  --log_level INFO || (echo "❌ Baseline models failed!" && exit 1)

baseline-full-optimized: llm-pipeline-full
	@echo "📊 Running baseline summarization models on existing crash graphs (full dataset, cost-efficient)..."
	@echo "💡 This reuses graphs from previous LLM runs, saving ~80% on API costs"
	@mkdir -p $(BASE_OUT_FULL)
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --batch_models facebook/bart-base t5-base \
	  --skip_llm \
	  --neo4j_enabled \
	  --out_dir $(BASE_OUT_FULL) \
	  --log_level INFO || (echo "❌ Baseline models failed!" && exit 1)

prepare-training-data-full: baseline-full
	@mkdir -p $(TRAINING_DATA_DIR)
	@echo "📊 Preparing Training Data (Full Dataset)..."
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(BART_FULL_DIR)/crash_graphs.jsonl \
	  --summaries_file $(BART_FULL_DIR)/crash_summaries.jsonl \
	  --output $(TRAINING_BART_FULL) \
	  --format csv || (echo "❌ BART training data preparation failed!" && exit 1)
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(T5_FULL_DIR)/crash_graphs.jsonl \
	  --summaries_file $(T5_FULL_DIR)/crash_summaries.jsonl \
	  --output $(TRAINING_T5_FULL) \
	  --format csv || (echo "❌ T5 training data preparation failed!" && exit 1)

finetune-bart-full: prepare-training-data-full
	@mkdir -p $(FT_BART_DIR)
	@echo "🎯 Running Model Training (BART - Full Dataset)..."
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART_FULL) \
	  --model_name facebook/bart-base \
	  --output_dir $(FT_BART_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ BART fine-tuning failed!" && exit 1)
	@echo "📊 Creating training visualizations..."
	@$(PY) create_visualizations.py $(FT_BART_DIR) || (echo "❌ BART visualization failed!" && exit 1)

finetune-t5-full: prepare-training-data-full
	@mkdir -p $(FT_T5_DIR)
	@echo "🎯 Running Model Training (T5 - Full Dataset)..."
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_T5_FULL) \
	  --model_name t5-base \
	  --output_dir $(FT_T5_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ T5 fine-tuning failed!" && exit 1)
	@echo "📊 Creating training visualizations..."
	@$(PY) create_visualizations.py $(FT_T5_DIR) || (echo "❌ T5 visualization failed!" && exit 1)

run-finetuned-bart-full: finetune-bart-full
	@echo "🚀 Running Fine-tuned BART Model (Full Dataset)..."
	@mkdir -p $(FT_BART_OUT_FULL)
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model facebook/bart-base \
	  --fine_tuned_model $(FT_BART_MODEL) \
	  --neo4j_enabled \
	  --out_dir $(FT_BART_OUT_FULL) \
	  --log_level INFO || (echo "❌ Fine-tuned BART run failed!" && exit 1)

run-finetuned-t5-full: finetune-t5-full
	@echo "🚀 Running Fine-tuned T5 Model (Full Dataset)..."
	@mkdir -p $(FT_T5_OUT_FULL)
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model t5-base \
	  --fine_tuned_model $(FT_T5_MODEL) \
	  --neo4j_enabled \
	  --out_dir $(FT_T5_OUT_FULL) \
	  --log_level INFO || (echo "❌ Fine-tuned T5 run failed!" && exit 1)

visualize-bart:
	@echo "📊 Creating BART training visualizations..."
	@$(PY) create_visualizations.py $(FT_BART_DIR) || (echo "❌ BART visualization failed!" && exit 1)

visualize-t5:
	@echo "📊 Creating T5 training visualizations..."
	@$(PY) create_visualizations.py $(FT_T5_DIR) || (echo "❌ T5 visualization failed!" && exit 1)

clean:
	rm -rf $(ARTIFACTS_DIR)