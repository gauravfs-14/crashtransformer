# CrashTransformer

> Start with [index.md](index.md) for the ordered guide.

## What this project does

- Converts free-text police crash narratives into:
  - Structured crash graphs (entities, events, causal edges) using LLM extraction
  - High-quality LLM-generated summaries for comparison and training
  - Concise causal summaries from fine-tuned models ranked by quality metrics
  - Optional storage in Neo4j for graph querying
  - Cost/logging/metrics artifacts for analysis

Key flow is orchestrated in `src/main_pipeline.py`:

```python
def run_one(row: Dict[str, Any], cps: CausalPlanSummarizer, track_tokens: bool = True, logger=None, llm_provider: str = None, llm_model: str = None, llm_api_key: str = None) -> Dict[str, Any]:
    """
    Returns a dict with graph, plan, best summary, metrics, and usage stats
    for both extraction and summarization stages.
    """
```

### How it works (pipeline stages)

- Crash graph extraction with an LLM:
  - Few-shot prompt and a structured-output schema (`CrashGraph` via Pydantic).
  - Multi-provider support through a provider factory.

```python
def create_structured_llm(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL, api_key: str = None, **kwargs):
    ... 
    return llm.with_structured_output(CrashGraph), llm_provider
```

```python
def analyze_crash_with_usage(narrative_or_formatted: str, metadata: Dict[str, Any], logger=None, provider: str = None, model: str = None, api_key: str = None):
    """
    Wrap the structured extractor with a token and time tracker.
    Returns (CrashGraph, LLMUsage)
    Falls back to zeros if callbacks are not available.
    """
```

- Plan-conditioned summarization with a HuggingFace seq2seq model (BART/T5 by default):
  - Builds causal plan from graph edges labeled `CAUSES`, then generates multiple summaries and scores them.

```python
def build_plan_from_graph(graph_json: Dict[str, Any]) -> Plan:
    rels = graph_json.get("relationships", [])
    # keep only causal edges, render concise plan lines
    edges = [r for r in rels if r.get("type") == "CAUSES"]
    lines = [f"{i}) cause -> effect" for i, _ in enumerate(edges, start=1)]
    return Plan(lines=lines, edges=edges)
```

```python
class CausalPlanSummarizer:
    def summarize(self, graph_json: Dict[str, Any], num_candidates: int = 3) -> Dict[str, Any]:
        plan = build_plan_from_graph(graph_json)
        candidates = self.gen.generate(
            graph_json["crash"].get("raw_narrative", ""),
            plan,
            num_return_sequences=num_candidates,
            num_beams=max(4, num_candidates),
        )
        scored = [{"summary": s, "metrics": score_summary(s, plan, graph_json["crash"].get("raw_narrative", ""))} for s in candidates]
        best = max(scored, key=lambda x: x["metrics"]["combined_score"]) if scored else None
        return {"plan_lines": plan.lines, "candidates": scored, "best": best}
```

- Aggregation and outputs:
  - Writes JSONL/JSON artifacts and a CSV of metrics and usage.

```python
# write aggregated JSONL
_write_jsonl(os.path.join(out_dir, "crash_graphs.jsonl"), graphs_agg)
_write_jsonl(os.path.join(out_dir, "crash_summaries.jsonl"), summaries_agg)

# write CSV with metrics and usage
csv_path = os.path.join(out_dir, "summaries_metrics.csv")
```

- Optional Neo4j persistence:

```python
class Neo4jSink:
    def ensure_constraints(self):
        cyphers = [
            "CREATE CONSTRAINT crash_id IF NOT EXISTS FOR (c:Crash) REQUIRE c.crash_id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
        ]
```

```cypher
MERGE (s:Summary {crash_id: $cid})
SET s.best_summary = $best_summary,
    s.plan_lines = $plan_lines,
    s.metrics = $metrics
WITH s
MERGE (c:Crash {crash_id: $cid})
MERGE (c)-[:HAS_SUMMARY]->(s)
```

- Provider abstraction and usage tracking (OpenAI, Anthropic, Google/Gemini, Groq, Ollama, XAI/Grok):

```python
class LLMProviderFactory:
    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "ollama": OllamaProvider,
        "groq": GroqProvider,
        "grok": GrokProvider,
    }
```

- Config, logging, cost tracking:

```python
class ConfigManager:
    def _load_from_env(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = self._get_llm_model()
        self.neo4j_enabled = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
```

```python
class CostTracker:
    """Tracks time, tokens, and computes costs for API or local models."""
```

```python
class CrashTransformerLogger:
    """Simple logger that writes structured logs to the logs/ directory."""
```

### What you input

- A CSV or XLSX with required columns:
  - `Crash_ID`, `Latitude`, `Longitude`, `CrashDate`, `DayOfWeek`, `CrashTime`, `County`, `City`, `SAE_Autonomy_Level`, `Crash_Severity`, `Narrative`.

```python
needed = [
    "Crash_ID", "Latitude", "Longitude", "CrashDate", "DayOfWeek", "CrashTime",
    "County", "City", "SAE_Autonomy_Level", "Crash_Severity", "Narrative"
]
```

- Environment variables (via `.env`) for provider keys, logging, Neo4j, etc. Resolved by `ConfigManager`.

### What you get out

- Files under `artifacts/<model_name>/`:
  - `crash_graphs.jsonl` and `crash_graphs.json`: structured graphs per crash.
  - `crash_summaries.jsonl` and `crash_summaries.json`: plan lines, best summary, usage.
  - `summaries_metrics.csv`: per-crash metrics plus usage/cost columns.
  - `cost_report.json`: if cost tracking for summarizer is enabled.
- Optional Neo4j nodes/relationships for querying in Cypher.

### What you can do with it

- Batch-convert crash narratives into machine-queriable graphs and high-quality summaries.
- Run across different LLM providers for extraction; swap summarization models or fine-tuned checkpoints.
- Persist graphs and summaries to Neo4j for analytics (e.g., most common violation→collision chains).
- Track and optimize costs, compare models, compute advanced NLP metrics (if deps installed).

### Typical workflow

1. Configure environment
   - Create `.env` from `.env.example` and set `LLM_PROVIDER`, `OPENAI_API_KEY` or provider-specific key, optional `ENABLE_NEO4J=true` and DB credentials.
   - Or run the interactive setup via the CLI wrapper (see README).
2. Prepare input data
   - Ensure CSV/XLSX contains the required columns; large files supported.
3. Run the pipeline
   - Basic:

     ```bash
     python crashtransformer.py run --csv path/to/crashes.csv
     ```

   - Choose extraction provider/model:

     ```bash
     python crashtransformer.py run --csv crashes.csv --llm_provider anthropic --llm_model claude-3-haiku-20240307
     ```

   - Enable Neo4j:

     ```bash
     python crashtransformer.py run --csv crashes.csv --neo4j_enabled --neo4j_uri bolt://localhost:7687
     ```

   - Run multiple summarization models:

     ```bash
     python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base
     ```

4. Inspect outputs
   - Check `artifacts/<model_name>/crash_graphs.jsonl` and `crash_summaries.jsonl`.
   - Review `summaries_metrics.csv` and `cost_report.json`.
   - If Neo4j enabled, query the graph database.
5. Optional: fine-tune summarization
   - Prepare training data from outputs (via `prepare_training_data.py`).
   - Train:

     ```bash
     python crashtransformer.py train --training_data path/to/data.csv --model_name facebook/bart-base --output_dir fine_tuned_models/bart
     ```

   - Re-run pipeline with `--fine_tuned_model fine_tuned_models/bart`.

### Command surface (from entrypoint)

```text
MAIN COMMANDS:
  setup, run, train, prepare-data, docs, validate, install, examples, help
```

### Execution details worth noting

- Extraction prompt is few-shot; schema is enforced via LangChain structured outputs (`CrashGraph`).
- Usage and API costs are recorded for extraction when provider supports it; summarizer tokens are approximated via tokenizer encoding in `run_one`.

```python
sum_input_tokens = len(tok.encode(prompt))
sum_output_tokens = len(tok.encode(result["best"]["summary"])) if result.get("best") else 0
```

- Metrics include custom causal PR/F1 aligned with the extracted plan, plus optional ROUGE/BLEU/BERTScore/semantic similarity if dependencies are installed.

### Summary

- Inputs: CSV/XLSX with crash metadata + narrative, `.env` with provider keys.
- Process: LLM extracts a structured crash graph → plan-conditioned HF model generates and scores summaries → artifacts written; optional Neo4j upserts.
- Outputs: JSONL/JSON graphs, summaries with metrics, CSV metrics table, logs, cost report, and optional Neo4j graph.
- Usage: `python crashtransformer.py run --csv your_data.csv [--llm_provider ...] [--neo4j_enabled] [--batch_models ...]`.

---

## 📖 Navigation

**Previous:** [Start Here](index.md) - Get started with CrashTransformer  
**Next:** [Environment Setup](ENVIRONMENT_SETUP.md) - Configure your environment
