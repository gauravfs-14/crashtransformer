# 📤 Outputs & Artifacts

The pipeline writes artifacts under `artifacts/` (or `--out_dir`). Structure may include a model-specific subdirectory.

## Files

- `crash_graphs.jsonl`: One JSON object per crash with structured entities, events, relationships
- `crash_summaries.jsonl`: Per-crash plan lines, candidate summaries with metrics, and best summary
- `summaries_metrics.csv`: Tabular metrics and usage/cost columns
- `cost_report.json`: Aggregate cost analysis (if cost tracking enabled)
- `logs/`: Timestamped processing logs

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
  "best": {"summary": "U1 failed to control speed and rear-ended U2.", "metrics": {"combined_score": 0.89}}
}
```

## Neo4j

If enabled, nodes and relationships are upserted with constraints and a `Summary` node linked via `HAS_SUMMARY`.
