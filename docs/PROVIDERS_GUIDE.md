# 🤖 LLM Providers & Models

CrashTransformer supports multiple LLM providers for graph extraction and various HF models for summarization. This guide explains setup and selection.

## Providers

Supported providers and required vars:

- OpenAI
  - `LLM_PROVIDER=openai`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (e.g., `gpt-4o-mini`)
- Anthropic
  - `LLM_PROVIDER=anthropic`
  - `ANTHROPIC_API_KEY`
  - Model (e.g., `claude-3-haiku-20240307`)
- Google (Gemini)
  - `LLM_PROVIDER=google` or `gemini`
  - `GOOGLE_API_KEY`
  - Model (e.g., `gemini-1.5-pro`)
- Groq
  - `LLM_PROVIDER=groq`
  - `GROQ_API_KEY`
  - Model (e.g., `llama-3.1-70b`)
- Ollama (local)
  - `LLM_PROVIDER=ollama`
  - Daemon running locally with pulled models (e.g., `llama3`)
- XAI (Grok)
  - `LLM_PROVIDER=grok`
  - `XAI_API_KEY`

Select per run with flags:

```bash
python crashtransformer.py run --csv crashes.csv --llm_provider anthropic --llm_model claude-3-haiku-20240307
```

## Summarization models

By default, the pipeline uses HF seq2seq models (BART/T5). You can also provide a fine-tuned checkpoint.

```bash
# baseline
python crashtransformer.py run --csv crashes.csv --model facebook/bart-base

# multiple
python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base

# fine-tuned
python crashtransformer.py run --csv crashes.csv --fine_tuned_model fine_tuned_models/bart/final_model
```

## Selection guidance

- Speed vs cost vs quality (rough guide):
  - Groq: fastest, good cost, solid quality
  - OpenAI: balanced
  - Anthropic: highest quality, higher latency/cost
  - Google: cost-effective, strong quality
  - Ollama: local only, zero API cost, quality depends on local model

Tips:

- Start with OpenAI or Groq for reliability/speed.
- Use Anthropic for the best extraction quality on complex narratives.
- Use Ollama for offline/local workflows; ensure prompts and schema fit model limits.

## Cost tracking

Enable/inspect cost tracking via `summaries_metrics.csv` and `cost_report.json`. See `COST_PERFORMANCE.md`.
