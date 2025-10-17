# 🧭 Interactive Setup Guide

Use this when first installing CrashTransformer or when reconfiguring credentials and defaults.

## 🚀 Start the setup

```bash
python crashtransformer.py setup
# or
python crashtransformer.py setup --interactive
```

What the wizard does:

- Selects LLM provider and model
- Securely captures API keys (hidden input; stored in `.env`)
- Optionally enables Neo4j (URI, user, password)
- Configures logging and output directories
- Validates configuration and can install dependencies

## 🔐 LLM provider and API keys

Supported providers: OpenAI, Anthropic, Google (Gemini), Groq, Ollama, XAI (Grok).
The wizard asks for provider, model, and key as needed. Keys are written to `.env` and never echoed.

## 🗄️ Neo4j (optional)

If enabled, provide:

- `NEO4J_URI` (e.g., `bolt://localhost:7687`)
- `NEO4J_USER`, `NEO4J_PASSWORD`

Neo4j nodes/relationships are upserted; basic constraints are created. You can also enable via `--neo4j_enabled` per run.

## 📝 Logging and outputs

- `LOG_DIR` default `logs/`
- `LOG_LEVEL` default `INFO` (use `DEBUG` when troubleshooting)
- `OUTPUT_DIR` default `artifacts/`

All can be overridden on the CLI.

## ✅ Validate and install

```bash
python crashtransformer.py validate
python crashtransformer.py install
```

## 📁 Example .env

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ENABLE_NEO4J=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
LOG_DIR=logs
LOG_LEVEL=INFO
ENABLE_LOGGING=true
OUTPUT_DIR=artifacts
```

Keep `.env` out of version control. Share `.env.example` instead.

## 🧪 Smoke test

```bash
python crashtransformer.py run --csv data/test_data_5rows.csv --log_level INFO
# or create a 5-row sample from data.xlsx
make data
```

## ❓ Help

```bash
python crashtransformer.py help
```
