# ⚙️ Environment & Configuration

Everything is configured via environment variables in a `.env` file. Create it interactively or from the example.

## 📦 Create and validate

```bash
# Interactive
python crashtransformer.py setup

# From template
cp .env.example .env

# Validate
python crashtransformer.py validate

# Install dependencies
python crashtransformer.py install
```

## 🔑 LLM configuration

```bash
LLM_PROVIDER=openai            # openai|anthropic|google|gemini|groq|ollama|grok
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
XAI_API_KEY=...
```

Notes:

- Only the variables required by your chosen provider need to be set.
- For Ollama, ensure the daemon is running locally and models are pulled.

## 🗄️ Neo4j (optional)

```bash
ENABLE_NEO4J=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
```

## 📝 Logging

```bash
ENABLE_LOGGING=true
LOG_DIR=logs
LOG_LEVEL=INFO     # DEBUG|INFO|WARNING|ERROR
```

## 📤 Outputs

```bash
OUTPUT_DIR=artifacts
```

## 🧰 CLI overrides

Every setting can be overridden per run, e.g.:

```bash
python crashtransformer.py run --csv crashes.csv \
  --llm_provider anthropic --llm_model claude-3-haiku-20240307 \
  --neo4j_enabled --neo4j_uri bolt://localhost:7687 \
  --out_dir results --log_level DEBUG
```

## 🔒 Security guidance

- Do not pass API keys via CLI flags.
- Store secrets only in `.env` or a secret manager.
- Avoid committing `.env`; use `.env.example` for sharing safe defaults.
