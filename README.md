# 🔧 CrashTransformer - AI-Powered Crash Analysis Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Neo4j](https://img.shields.io/badge/Neo4j-6.0+-green.svg)](https://neo4j.com/)

CrashTransformer is a sophisticated AI system that processes vehicle crash narratives into structured causal summaries using Large Language Models and transformer-based summarization. It extracts structured crash graphs, generates high-quality summaries, and provides comprehensive analytics through multiple LLM providers and graph database integration.

## ✨ Key Features

- **🤖 Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Groq, Ollama, XAI
- **📊 Advanced Analytics**: Graph extraction, dual summarization, quality metrics
- **🗄️ Graph Database Integration**: Neo4j support for complex pattern analysis
- **💰 Cost Tracking**: Real-time token usage and cost optimization
- **🔧 Flexible Configuration**: Environment-based setup with CLI overrides

## 🚀 Quick Start

```bash
# 1. Setup environment (first time only)
python crashtransformer.py setup

# 2. Run the pipeline
python crashtransformer.py run --csv crashes.csv
```

## 📋 Main Commands

| Command | Description | Example |
|---------|-------------|---------|
| `setup` | Interactive environment configuration | `python crashtransformer.py setup` |
| `run` | Execute crash analysis pipeline | `python crashtransformer.py run --csv crashes.csv` |
| `train` | Fine-tune summarization models | `python crashtransformer.py train --training_data data.csv` |
| `prepare-data` | Prepare training data from outputs | `python crashtransformer.py prepare-data --source pipeline` |
| `clean-db` | Clear Neo4j database | `python crashtransformer.py clean-db` |
| `docs` | Open local HTML documentation | `python crashtransformer.py docs` |
| `help` | Show comprehensive help | `python crashtransformer.py help` |

## 🎯 Core Features

### **🤖 Multi-Provider LLM Support**

| Provider | Models | Speed | Cost | Quality | Best For |
|----------|--------|-------|------|---------|----------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-3.5-turbo | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced performance |
| **Anthropic** | Claude-3 Opus, Sonnet, Haiku | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | High-quality analysis |
| **Google** | Gemini-2.0-Flash, Gemini-1.5-Pro | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Cost-effective processing |
| **Groq** | Llama-3.1-70b, Mixtral-8x7b | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast inference |
| **Ollama** | Local models (Llama3, Mistral) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Privacy-focused |
| **XAI** | Grok-Beta, Grok-2 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Latest models |

### **📊 Advanced Analytics**

- **🧠 Graph Extraction**: LLM-powered structured entities, events, and causal relationships
- **📝 Dual Summarization**: Both LLM and fine-tuned model summaries for comprehensive analysis
- **⚡ Single LLM Call**: Efficient API usage - one call generates both graph and summary
- **📈 Quality Metrics**: Precision, recall, faithfulness, hallucination rates, ROUGE, BLEU, BERTScore
- **💰 Cost Tracking**: Real-time token usage, runtime, and monetary costs for all providers
- **📊 Performance Analytics**: Processing times, throughput analysis, and model comparison

### **🗄️ Data Storage & Integration**

- **📁 File Outputs**: JSON, JSONL, CSV formats with structured data
- **🕸️ Graph Database**: Neo4j integration for complex pattern analysis and queries
- **📝 Comprehensive Logging**: Timestamped, structured logs with detailed processing information
- **💵 Cost Reports**: Detailed financial analysis and optimization recommendations

## 🔧 Environment Setup

### **Interactive Setup (Recommended)**

```bash
# Guided setup process
python crashtransformer.py setup

# Direct interactive setup
python crashtransformer.py setup --interactive
```

### **Manual Setup**

```bash
# Create .env from template
python crashtransformer.py setup --create-env

# Validate configuration
python crashtransformer.py validate

# Install dependencies
python crashtransformer.py install
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
```

## 📊 Input Requirements

### **CSV/XLSX Format**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Crash_ID` | String | ✅ | Unique crash identifier |
| `Latitude` | Float | ✅ | Geographic latitude coordinate |
| `Longitude` | Float | ✅ | Geographic longitude coordinate |
| `CrashDate` | String | ✅ | Date of crash (YYYY-MM-DD format) |
| `DayOfWeek` | String | ✅ | Day of week (MON, TUE, etc.) |
| `CrashTime` | String | ✅ | Time of crash (HH:MM format) |
| `County` | String | ✅ | County name |
| `City` | String | ✅ | City name |
| `SAE_Autonomy_Level` | String | ✅ | Vehicle autonomy level (0-5) |
| `Crash_Severity` | String | ✅ | Severity classification |
| `Narrative` | String | ✅ | Crash description text |

### **Example Input**

```csv
Crash_ID,Latitude,Longitude,CrashDate,DayOfWeek,CrashTime,County,City,SAE_Autonomy_Level,Crash_Severity,Narrative
19955047,26.15526348,-97.99060556,2023-01-15,Sunday,14:30,Hidalgo,Weslaco,Level 0,Not Injured,"Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."
19955369,32.92553538,-96.80385791,2023-01-16,Monday,15:37,Dallas,Dallas,Level 1,Injured,"Unit 1 was traveling eastbound on Main Street. Unit 2 was traveling northbound on Oak Avenue. Unit 1 failed to yield at the stop sign and collided with Unit 2 at the intersection."
```

## 📈 Outputs Generated

### **1. Structured Graph Data**

The system extracts structured crash graphs with entities, events, and causal relationships:

```json
{
  "crash": {
    "crash_id": "19955047",
    "latitude": 26.15526348,
    "longitude": -97.99060556,
    "city": "Weslaco",
    "crash_severity": "Not Injured",
    "raw_narrative": "Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."
  },
  "entities": [
    {"id": "19955047:U1", "label": "VEHICLE", "unit_id": "1", "name": "Unit 1"},
    {"id": "19955047:U2", "label": "VEHICLE", "unit_id": "2", "name": "Unit 2"},
    {"id": "19955047:L1", "label": "ROAD", "name": "S. Texas Blvd", "city": "Weslaco"}
  ],
  "events": [
    {"id": "19955047:E1", "type": "VIOLATION", "label": "Failure to Control Speed", "evidence_span": "Unit 1 failed to control speed"},
    {"id": "19955047:E2", "type": "COLLISION", "label": "Rear-End Collision", "evidence_span": "struck Unit 2 on the back end"}
  ],
  "relationships": [
    {"start": "19955047:U1", "end": "19955047:E1", "type": "PARTICIPATED_IN", "properties": {"role": "agent"}},
    {"start": "19955047:E1", "end": "19955047:E2", "type": "CAUSES", "properties": {"marked": true, "connective": "failed to control speed"}},
    {"start": "19955047:U1", "end": "19955047:U2", "type": "HIT", "properties": {"impact_config": "rear_end"}}
  ]
}
```

### **2. Causal Summaries**

High-quality summaries with comprehensive quality metrics:

```text
Input: "Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."

Output: "Unit 1 failed to control speed and struck the stationary Unit 2 from behind, causing a rear-end collision."

Quality Metrics: {
  "causal_precision": 0.95,
  "causal_recall": 0.90,
  "causal_f1": 0.92,
  "span_faithfulness": 0.88,
  "hallucination_rate": 0.05,
  "compression_ratio": 0.15,
  "combined_score": 0.89,
  "rouge_rouge1_f1": 0.87,
  "bertscore_f1": 0.91
}
```

### **3. File Outputs**

| File | Description | Format |
|------|-------------|--------|
| `crash_graphs.jsonl` | Structured graph data | JSONL |
| `crash_summaries.jsonl` | Generated summaries with metrics | JSONL |
| `summaries_metrics.csv` | Quality metrics and performance data | CSV |
| `cost_report.json` | Cost analysis and optimization | JSON |
| `logs/crashtransformer-*.log` | Processing logs | Text |

### **4. Neo4j Graph Database**

Advanced graph database integration for complex pattern analysis:

- **Nodes**: `Crash`, `Vehicle`, `Location`, `Event`, `Summary`
- **Relationships**: `HAS_ENTITY`, `HAS_EVENT`, `CAUSES`, `PARTICIPATED_IN`, `HIT`, `HAS_SUMMARY`

See [NEO4J_GUIDE.md](docs/NEO4J_GUIDE.md) for comprehensive Cypher queries and analytics.

## 🔒 Security & Configuration

### **Environment Variables**

All sensitive configuration is managed through `.env` file:

```bash
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
ENABLE_NEO4J=false

# Logging Configuration
LOG_DIR=logs
LOG_LEVEL=INFO
ENABLE_LOGGING=true
```

### **Security Features**

- **🔐 Hidden Input**: API keys never displayed in terminal
- **🔐 Secure Storage**: Configuration saved to `.env` file
- **🔐 Validation**: Automatic configuration checking
- **🔐 No CLI Secrets**: Sensitive data only via environment variables

## 📊 Performance & Cost Optimization

### **Provider Selection Guide**

| Provider | Speed | Cost | Quality | Use Case |
|----------|-------|------|---------|----------|
| **Groq** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast processing |
| **OpenAI** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| **Anthropic** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | High quality |
| **Google** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Cost-effective |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Local processing |

### **Cost Tracking**

- **Token Usage**: Input/output tokens for each stage
- **API Costs**: Real-time cost calculation
- **GPU Costs**: Local processing costs
- **Optimization**: Cost per 1,000 summaries analysis

## 🛠️ Development & Customization

### **Project Structure**

```text
crashtransformer/
├── crashtransformer.py              # Main entry point
├── src/                             # Source code
│   ├── main_pipeline.py             # Pipeline orchestration
│   ├── setup_env.py                # Environment setup
│   ├── prepare_training_data.py    # Training data preparation
│   ├── train_model.py              # Model fine-tuning
│   └── utils/                       # Core modules
│       ├── crash_graph_llm.py      # LLM integration & structured output
│       ├── causal_plan_summarizer.py # Summarization & quality metrics
│       ├── cost_tracker.py         # Cost tracking & optimization
│       ├── neo4j_io.py             # Graph database integration
│       ├── llm_providers.py        # Multi-provider support
│       ├── advanced_metrics.py     # NLP quality metrics
│       ├── fine_tuning.py          # Model fine-tuning utilities
│       └── config.py               # Configuration management
├── docs/                           # Documentation
│   ├── NEO4J_GUIDE.md              # Neo4j integration guide
│   ├── USAGE_GUIDE.md              # Usage documentation
│   └── ...                         # Additional guides
├── data/                           # Sample data
├── logs/                           # Processing logs
└── artifacts/                      # Output directory
```

### **Key Components**

| Component | Description | Key Features |
|-----------|-------------|---------------|
| **crash_graph_llm.py** | LLM integration | Structured output, multi-provider support |
| **causal_plan_summarizer.py** | Summarization | Quality metrics, BART/T5 models |
| **neo4j_io.py** | Graph database | CRUD operations, constraints, indexes |
| **llm_providers.py** | Provider abstraction | OpenAI, Anthropic, Google, Groq, Ollama, XAI |
| **cost_tracker.py** | Cost management | Token tracking, cost optimization |
| **advanced_metrics.py** | Quality assessment | ROUGE, BLEU, BERTScore, semantic similarity |

### **Customization Options**

- **🔧 Custom Models**: Add new LLM providers via `LLMProviderFactory`
- **📊 Custom Metrics**: Extend quality scoring in `advanced_metrics.py`
- **📁 Custom Outputs**: Add new output formats in `main_pipeline.py`
- **⚙️ Custom Processing**: Extend pipeline stages with new modules
- **🎯 Fine-tuning**: Custom model training with `train_model.py`

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**

   ```bash
   ❌ Error importing setup module
   ```

   **Solution**: Ensure you're in the correct directory with `src/` folder

2. **Configuration Errors**

   ```bash
   ❌ Configuration validation failed
   ```

   **Solution**: Run `python crashtransformer.py setup` to configure

3. **API Key Issues**

   ```bash
   ❌ OPENAI_API_KEY not set
   ```

   **Solution**: Run interactive setup and enter valid API key

### **Debug Mode**

```bash
# Enable debug logging
python crashtransformer.py run --csv crashes.csv --log_level DEBUG

# Check configuration
python crashtransformer.py validate
```

## 📚 Documentation

| Document | Description | Purpose |
|----------|-------------|---------|
| **[NEO4J_GUIDE.md](docs/NEO4J_GUIDE.md)** | Neo4j integration & Cypher queries | Graph database analytics |
| **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** | Complete usage documentation | Getting started guide |
| **[ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)** | Configuration setup | Environment configuration |
| **[PROVIDERS_GUIDE.md](docs/PROVIDERS_GUIDE.md)** | LLM provider comparison | Provider selection |
| **[COST_PERFORMANCE.md](docs/COST_PERFORMANCE.md)** | Cost optimization guide | Performance tuning |
| **[FINE_TUNING_GUIDE.md](docs/FINE_TUNING_GUIDE.md)** | Model fine-tuning | Custom model training |

## 🎯 Use Cases

| Industry | Use Case | Benefits |
|----------|----------|----------|
| **🏢 Insurance** | Automated crash report analysis | Faster claims processing, fraud detection |
| **🚗 Transportation** | Safety pattern analysis | Accident prevention, infrastructure planning |
| **⚖️ Legal** | Evidence extraction and summarization | Case preparation, documentation |
| **🔬 Research** | Large-scale crash data analysis | Statistical insights, trend analysis |
| **📋 Compliance** | Regulatory reporting | Automated documentation, audit trails |

## 🚀 Getting Started

### **Quick Setup**

```bash
# 1. Clone the repository
git clone https://github.com/your-org/crashtransformer.git
cd crashtransformer

# 2. Setup environment
python crashtransformer.py setup

# 3. Prepare your data (ensure CSV has required columns)
# 4. Run the pipeline
python crashtransformer.py run --csv crashes.csv

# 5. Check results in artifacts/ directory
```

### **Advanced Setup**

```bash
# Enable Neo4j for graph analytics
python crashtransformer.py run --csv crashes.csv --neo4j_enabled

# Use specific LLM provider
python crashtransformer.py run --csv crashes.csv --llm_provider anthropic

# Compare multiple models
python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base
```

## 📞 Support & Community

### **Getting Help**

- 📖 **Documentation**: Check `docs/` folder for comprehensive guides
- 🆘 **CLI Help**: Run `python crashtransformer.py help`
- 📝 **Logs**: Review processing logs in `logs/` directory
- 🐛 **Issues**: Report bugs and feature requests on GitHub

### **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## CrashTransformer

Transforming crash narratives into actionable insights with AI! 🚗💥🤖
