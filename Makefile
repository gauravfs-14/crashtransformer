# Makefile for CrashTransformer - Simplified

.PHONY: test full setup clean help

# Stop on any error
.SHELLFLAGS := -e -o pipefail

PY ?= uv run
ROOT := $(abspath .)

# Data files
DATA_XLSX := $(ROOT)/data/data.xlsx
TEST_CSV := $(ROOT)/data/test_data_5rows.csv
SHEET ?= Narr_CrLev

# Artifacts directories
ARTIFACTS_DIR := $(ROOT)/artifacts
TEST_OUT := $(ARTIFACTS_DIR)/test
FULL_OUT := $(ARTIFACTS_DIR)/full

# Fine-tuned model directories
FT_BART_DIR := $(ARTIFACTS_DIR)/fine_tuned_models/bart
FT_T5_DIR := $(ARTIFACTS_DIR)/fine_tuned_models/t5
FT_BART_MODEL := $(FT_BART_DIR)/final_model
FT_T5_MODEL := $(FT_T5_DIR)/final_model

# Training data files
TRAINING_DATA_DIR := $(ARTIFACTS_DIR)/training_data
TRAINING_BART := $(TRAINING_DATA_DIR)/training_data_bart.csv
TRAINING_T5 := $(TRAINING_DATA_DIR)/training_data_t5.csv
TRAINING_BART_FULL := $(TRAINING_DATA_DIR)/training_data_bart_full.csv
TRAINING_T5_FULL := $(TRAINING_DATA_DIR)/training_data_t5_full.csv

# Main targets
test: setup data test-pipeline
full: setup full-pipeline

help:
	@echo "CrashTransformer - Simplified Makefile"
	@echo "====================================="
	@echo ""
	@echo "Main Targets:"
	@echo "  test    - Run complete pipeline on 5-row test data (fast, cost-efficient)"
	@echo "  full    - Run complete pipeline on full dataset (slow, comprehensive)"
	@echo "  setup   - Setup environment and dependencies"
	@echo "  clean   - Remove all generated artifacts and clear Neo4j database"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make test    # Quick test with 5 rows"
	@echo "  make full    # Full run with complete dataset"
	@echo ""
	@echo "💡 Both targets are cost-optimized and reuse graphs to save ~80% on LLM API costs"

# Setup and data preparation
setup:
	$(PY) crashtransformer.py setup

data: $(TEST_CSV)

$(TEST_CSV): $(DATA_XLSX)
	@echo "Creating 5-row test CSV at $(TEST_CSV) from $(DATA_XLSX) [sheet=$(SHEET)]"
	@mkdir -p $(dir $(TEST_CSV))
	@echo "import pandas as pd; df = pd.read_excel('$(DATA_XLSX)', sheet_name='$(SHEET)'); df.head(5).to_csv('$(TEST_CSV)', index=False); print('Wrote $(TEST_CSV)')" > /tmp/create_test_data.py
	$(PY) /tmp/create_test_data.py
	@rm -f /tmp/create_test_data.py

# Test pipeline (5-row sample)
test-pipeline: data
	@echo "🚀 Running complete test pipeline (5 rows)..."
	@mkdir -p $(TEST_OUT)
	
	@echo "🤖 Phase 1: Generating crash graphs with LLM..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --out_dir $(TEST_OUT) \
	  --neo4j_enabled \
	  --log_level INFO || (echo "❌ LLM pipeline failed!" && exit 1)
	
	@echo "📊 Phase 2: Running baseline models (cost-efficient)..."
	@echo "🤖 Running BART baseline model..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model facebook/bart-base \
	  --skip_llm \
	  --out_dir $(TEST_OUT) \
	  --log_level INFO || (echo "❌ BART baseline failed!" && exit 1)
	@echo "🤖 Running T5 baseline model..."
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model t5-base \
	  --skip_llm \
	  --out_dir $(TEST_OUT) \
	  --log_level INFO || (echo "❌ T5 baseline failed!" && exit 1)
	
	@echo "📊 Phase 3: Preparing training data..."
	@mkdir -p $(TRAINING_DATA_DIR)
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(TEST_OUT)/facebook_bart-base/crash_graphs.jsonl \
	  --summaries_file $(TEST_OUT)/facebook_bart-base/crash_summaries.jsonl \
	  --output $(TRAINING_BART) \
	  --format csv || (echo "❌ Training data preparation failed!" && exit 1)
	
	@echo "🎯 Phase 4: Fine-tuning models..."
	@mkdir -p $(FT_BART_DIR)
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART) \
	  --model_name facebook/bart-base \
	  --output_dir $(FT_BART_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ BART fine-tuning failed!" && exit 1)
	@mkdir -p $(FT_T5_DIR)
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART) \
	  --model_name t5-base \
	  --output_dir $(FT_T5_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ T5 fine-tuning failed!" && exit 1)
	
	@echo "🚀 Phase 5: Running fine-tuned models..."
	@mkdir -p $(ARTIFACTS_DIR)/finetuned_bart
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model facebook/bart-base \
	  --fine_tuned_model $(FT_BART_MODEL) \
	  --skip_llm \
	  --out_dir $(ARTIFACTS_DIR)/finetuned_bart \
	  --log_level INFO || (echo "❌ Fine-tuned BART failed!" && exit 1)
	@mkdir -p $(ARTIFACTS_DIR)/finetuned_t5
	@$(PY) crashtransformer.py run \
	  --csv $(TEST_CSV) \
	  --model t5-base \
	  --fine_tuned_model $(FT_T5_MODEL) \
	  --skip_llm \
	  --out_dir $(ARTIFACTS_DIR)/finetuned_t5 \
	  --log_level INFO || (echo "❌ Fine-tuned T5 failed!" && exit 1)
	
	@echo "📊 Creating visualizations..."
	@$(PY) create_visualizations.py $(FT_BART_DIR) || (echo "❌ BART visualization failed!" && exit 1)
	@$(PY) create_visualizations.py $(FT_T5_DIR) || (echo "❌ T5 visualization failed!" && exit 1)
	
	@echo "✅ Test pipeline completed successfully!"

# Full pipeline (complete dataset)
full-pipeline:
	@echo "🚀 Running complete full pipeline (all data)..."
	@mkdir -p $(FULL_OUT)
	
	@echo "🤖 Phase 1: Generating crash graphs with LLM..."
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --out_dir $(FULL_OUT) \
	  --neo4j_enabled \
	  --log_level INFO || (echo "❌ LLM pipeline failed!" && exit 1)
	
	@echo "📊 Phase 2: Running baseline models (cost-efficient)..."
	@echo "🤖 Running BART baseline model (reusing graphs)..."
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model facebook/bart-base \
	  --skip_llm \
	  --out_dir $(FULL_OUT) \
	  --log_level INFO || (echo "❌ BART baseline failed!" && exit 1)
	@echo "🤖 Running T5 baseline model (reusing graphs)..."
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model t5-base \
	  --skip_llm \
	  --out_dir $(FULL_OUT) \
	  --log_level INFO || (echo "❌ T5 baseline failed!" && exit 1)
	
	@echo "📊 Phase 3: Preparing training data..."
	@mkdir -p $(TRAINING_DATA_DIR)
	@$(PY) crashtransformer.py prepare-data \
	  --source pipeline \
	  --graphs_file $(FULL_OUT)/crash_graphs.jsonl \
	  --summaries_file $(FULL_OUT)/crash_summaries.jsonl \
	  --output $(TRAINING_BART_FULL) \
	  --format csv || (echo "❌ Training data preparation failed!" && exit 1)
	
	@echo "🎯 Phase 4: Fine-tuning models..."
	@mkdir -p $(FT_BART_DIR)
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART_FULL) \
	  --model_name facebook/bart-base \
	  --output_dir $(FT_BART_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ BART fine-tuning failed!" && exit 1)
	@mkdir -p $(FT_T5_DIR)
	@$(PY) crashtransformer.py train \
	  --training_data $(TRAINING_BART_FULL) \
	  --model_name t5-base \
	  --output_dir $(FT_T5_DIR) \
	  --num_epochs 3 \
	  --batch_size 4 \
	  --learning_rate 5e-5 || (echo "❌ T5 fine-tuning failed!" && exit 1)
	
	@echo "🚀 Phase 5: Running fine-tuned models..."
	@mkdir -p $(ARTIFACTS_DIR)/finetuned_bart_full
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model facebook/bart-base \
	  --fine_tuned_model $(FT_BART_MODEL) \
	  --skip_llm \
	  --out_dir $(ARTIFACTS_DIR)/finetuned_bart_full \
	  --log_level INFO || (echo "❌ Fine-tuned BART failed!" && exit 1)
	@mkdir -p $(ARTIFACTS_DIR)/finetuned_t5_full
	@$(PY) crashtransformer.py run \
	  --xlsx $(DATA_XLSX) \
	  --model t5-base \
	  --fine_tuned_model $(FT_T5_MODEL) \
	  --skip_llm \
	  --out_dir $(ARTIFACTS_DIR)/finetuned_t5_full \
	  --log_level INFO || (echo "❌ Fine-tuned T5 failed!" && exit 1)
	
	@echo "📊 Creating visualizations..."
	@$(PY) create_visualizations.py $(FT_BART_DIR) || (echo "❌ BART visualization failed!" && exit 1)
	@$(PY) create_visualizations.py $(FT_T5_DIR) || (echo "❌ T5 visualization failed!" && exit 1)
	
	@echo "✅ Full pipeline completed successfully!"

# Utilities
clean:
	@echo "🧹 Cleaning up generated artifacts..."
	rm -rf $(ARTIFACTS_DIR)
	@echo "🗄️ Clearing Neo4j database..."
	@$(PY) crashtransformer.py clean-db || echo "⚠️ Neo4j cleanup failed (database might not be running)"
	@echo "✅ Cleanup completed!"