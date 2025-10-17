# 📤 Outputs & Artifacts

The pipeline writes artifacts under `artifacts/` (or `--out_dir`). Structure may include a model-specific subdirectory.

## Files

- `crash_graphs.jsonl`: One JSON object per crash with structured entities, events, relationships, and LLM summaries
- `crash_summaries.jsonl`: Per-crash plan lines, candidate summaries with metrics, best summary, and LLM summary
- `summaries_metrics.csv`: Tabular metrics and usage/cost columns for both LLM and baseline models
- `cost_report.json`: Aggregate cost analysis (if cost tracking enabled)
- `logs/`: Timestamped processing logs

## Key Features

### LLM-Generated Labels

- **Single LLM Call**: Efficient single API call generates both graph and summary
- **Graph Construction**: LLM extracts structured entities, events, and causal relationships
- **Initial Summary**: LLM generates high-quality summaries that serve as labels for metrics computation
- **Comparison Baseline**: LLM summaries are used to compare against fine-tuned model performance
- **Cost Optimization**: Reduced API costs by eliminating duplicate LLM calls

### Complete Artifacts

- **Structured Graphs**: Full entity-relationship graphs with metadata
- **Dual Summaries**: Both LLM and fine-tuned model summaries for comparison
- **Quality Metrics**: Comprehensive evaluation metrics for both approaches

## Graph JSON example

```json
{
  "crash": {"crash_id": "19955047", "latitude": 26.1552, "longitude": -97.9906, "city": "Weslaco", "crash_severity": "Not Injured"},
  "entities": [{"id": "19955047:U1", "label": "Vehicle", "unit_id": "U1"}],
  "events": [{"id": "19955047:E1", "type": "Violation", "attributes": {"reason": "failed to control speed"}}],
  "relationships": [{"start": "19955047:E1", "end": "19955047:E2", "type": "CAUSES", "properties": {"marked": true}}]
}
```

## Summary block example

```json
{
  "plan_lines": ["1) failed to control speed -> rear-end collision"],
  "candidates": [{"summary": "U1 failed to control speed and rear-ended U2.", "metrics": {"combined_score": 0.89}}],
  "best": {"summary": "U1 failed to control speed and rear-ended U2.", "metrics": {"combined_score": 0.89}},
  "llm_summary": "Unit 1 failed to control speed and struck the stationary Unit 2 from behind, causing a rear-end collision.",
  "extract_runtime_sec": 2.34,
  "extract_total_tokens": 1250,
  "extract_cost_usd": 0.0015,
  "extract_provider": "openai",
  "extract_model": "gpt-4o-mini"
}
```

## Neo4j

If enabled, nodes and relationships are upserted with constraints and a `Summary` node linked via `HAS_SUMMARY`.
