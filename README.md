# 🔧 CrashTransformer - AI-Powered Crash Analysis Pipeline

CrashTransformer is a sophisticated AI system that processes vehicle crash narratives into structured causal summaries using Large Language Models and transformer-based summarization.

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
| `validate` | Validate current configuration | `python crashtransformer.py validate` |
| `install` | Install required dependencies | `python crashtransformer.py install` |
| `examples` | Show usage examples | `python crashtransformer.py examples` |
| `help` | Show comprehensive help | `python crashtransformer.py help` |

## 🎯 Features

### **🤖 Multi-Provider LLM Support**

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude-3 Opus, Sonnet, Haiku
- **Google**: Gemini-1.5-Pro, Gemini-1.5-Flash
- **Groq**: Llama-3.1, Mixtral-8x7b (Fast inference)
- **Ollama**: Local models (Llama3, Mistral, CodeLlama)
- **XAI**: Grok-Beta, Grok-2

### **📊 Advanced Analytics**

- **Graph Extraction**: Structured entities, events, relationships
- **Causal Summarization**: Plan-conditioned summary generation
- **Quality Metrics**: Precision, recall, faithfulness, hallucination rates
- **Cost Tracking**: Token usage, runtime, monetary costs
- **Performance Analytics**: Processing times, throughput analysis

### **🗄️ Data Storage**

- **File Outputs**: JSON, JSONL, CSV formats
- **Graph Database**: Neo4j integration for complex queries
- **Comprehensive Logging**: Timestamped, structured logs
- **Cost Reports**: Detailed financial analysis

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

Required columns:

- `Crash_ID`: Unique identifier
- `Latitude`, `Longitude`: Geographic coordinates
- `CrashDate`, `DayOfWeek`, `CrashTime`: Temporal data
- `County`, `City`: Location information
- `SAE_Autonomy_Level`: Vehicle autonomy level
- `Crash_Severity`: Severity classification
- `Narrative`: Crash description text

### **Example Input**

```csv
Crash_ID,Latitude,Longitude,CrashDate,DayOfWeek,CrashTime,County,City,SAE_Autonomy_Level,Crash_Severity,Narrative
19955047,26.15526348,-97.99060556,2023-01-15,Sunday,14:30,Hidalgo,Weslaco,Level 0,Not Injured,"Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."
```

## 📈 Outputs Generated

### **1. Structured Graph Data**

```json
{
  "crash": {
    "crash_id": "19955047",
    "latitude": 26.15526348,
    "longitude": -97.99060556,
    "city": "Weslaco",
    "crash_severity": "Not Injured"
  },
  "entities": [
    {"id": "19955047:U1", "label": "Vehicle", "unit_id": "U1"},
    {"id": "19955047:U2", "label": "Vehicle", "unit_id": "U2"}
  ],
  "events": [
    {"id": "19955047:E1", "type": "Violation", "attributes": {"reason": "failed to control speed"}},
    {"id": "19955047:E2", "type": "Collision", "attributes": {"impact_config": "rear_end"}}
  ],
  "relationships": [
    {"start": "19955047:E1", "end": "19955047:E2", "type": "CAUSES", "properties": {"marked": true}}
  ]
}
```

### **2. Causal Summaries**

```
Input: "Unit 2 was stationary in the northbound lane. Unit 1 failed to control speed and struck Unit 2 on the back end."

Output: "Unit 1 failed to control speed and struck the stationary Unit 2 from behind, causing a rear-end collision."

Metrics: {
  "causal_precision": 0.95,
  "causal_recall": 0.90,
  "causal_f1": 0.92,
  "span_faithfulness": 0.88,
  "hallucination_rate": 0.05,
  "compression_ratio": 0.15,
  "combined_score": 0.89
}
```

### **3. File Outputs**

- `crash_graphs.jsonl` - Structured graph data
- `crash_summaries.jsonl` - Generated summaries
- `summaries_metrics.csv` - Quality metrics
- `cost_report.json` - Cost analysis
- `logs/` - Processing logs

### **4. Neo4j Graph Database**

- **Nodes**: Crash, Vehicle, Location, Event, Summary
- **Relationships**: HAS_ENTITY, HAS_EVENT, CAUSES, PARTICIPATED_IN, HIT, HAS_SUMMARY

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

```
crashtransformer/
├── crashtransformer.py          # Main entry point
├── src/                         # Source code
│   ├── main_pipeline.py         # Pipeline orchestration
│   ├── setup_env.py            # Environment setup
│   ├── utils/                   # Core modules
│   │   ├── crash_graph_llm.py  # LLM integration
│   │   ├── causal_plan_summarizer.py  # Summarization
│   │   ├── cost_tracker.py     # Cost tracking
│   │   ├── neo4j_io.py         # Database integration
│   │   ├── llm_providers.py    # Multi-provider support
│   │   └── config.py           # Configuration management
│   └── misc/                   # Utilities
│       └── logger.py           # Logging system
├── env.example                 # Environment template
└── README.md                   # This file
```

### **Customization**

- **Custom Models**: Add new LLM providers
- **Custom Metrics**: Extend quality scoring
- **Custom Outputs**: Add new output formats
- **Custom Processing**: Extend pipeline stages

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

- **`INTERACTIVE_SETUP.md`** - Environment setup guide
- **`ENVIRONMENT_SETUP.md`** - Configuration documentation
- **`src/`** - Source code and modules

## 🎯 Use Cases

1. **Insurance Claims**: Automated crash report analysis
2. **Transportation Safety**: Pattern analysis and prevention
3. **Legal Analysis**: Evidence extraction and summarization
4. **Research**: Large-scale crash data analysis
5. **Compliance**: Regulatory reporting and documentation

## 🚀 Getting Started

1. **Clone the repository**
2. **Run setup**: `python crashtransformer.py setup`
3. **Prepare data**: Ensure CSV has required columns
4. **Run pipeline**: `python crashtransformer.py run --csv crashes.csv`
5. **Check results**: Review outputs in `artifacts/` directory

## 📞 Support

For issues and questions:

- Check documentation in `docs/` folder
- Run `python crashtransformer.py help`
- Review logs in `logs/` directory

---

**CrashTransformer** - Transforming crash narratives into actionable insights with AI! 🚗💥🤖
