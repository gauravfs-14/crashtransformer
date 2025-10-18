# 💰 Cost & Performance Guide

This guide helps you manage token costs and runtime performance.

## Cost tracking

Artifacts:

- `summaries_metrics.csv`: per-crash token usage and costs (when available)
- `cost_report.json`: aggregated totals and breakdowns

CLI:

```bash
python crashtransformer.py run --csv crashes.csv --cost_mode api
```

## Provider selection

High-level guidance:

- Groq: fastest, good cost, solid quality
- OpenAI: balanced across dimensions
- Anthropic: highest quality, higher latency/cost
- Google: cost-effective with strong quality
- Ollama: zero API cost; model quality varies; local hardware dependent

## Practical tips

- Use shorter narratives where possible; trim unrelated text
- Batch processing by grouping similar lengths to reduce variance
- Prefer smaller-but-strong models (e.g., GPT-4o-mini) when quality is adequate
- Use `--batch_models` to compare quality/cost trade-offs
- Enable logging at `INFO`; use `DEBUG` only when necessary

## Summarization performance

- Reduce `--max_length` and `--max_target_length` when fine-tuning
- Use `--fp16` for faster training on supported GPUs
- Monitor GPU/CPU utilization; avoid contention

## Neo4j

- If remote, run the DB close to the pipeline to reduce network latency
- Create indexes/constraints on high-cardinality properties

## Reporting

- Track cost per 1,000 summaries over time
- Compare providers/models using `summaries_metrics.csv`

---

## 📖 Navigation

**Previous:** [Outputs](OUTPUTS.md) - Output structure and artifacts  
**Next:** [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) - Model training
