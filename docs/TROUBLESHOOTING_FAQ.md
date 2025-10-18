# 🛠️ Troubleshooting & FAQ

## Common issues

### Import error: setup/pipeline

```text
❌ Error importing setup module
```

- Ensure you're in the project root with `src/` present
- Run `python crashtransformer.py help` to verify the entrypoint

### Configuration validation failed

```text
❌ Configuration validation failed
```

- Run `python crashtransformer.py setup` and complete prompts
- Confirm `.env` contains the needed variables for your provider

### Missing API key

```text
❌ OPENAI_API_KEY not set
```

- Run interactive setup; keys are stored in `.env`
- Avoid passing secrets via CLI flags

### CSV file not found

```text
❌ CSV file not found
```

- Check the path and filename
- On macOS/Linux, verify case-sensitivity and working directory

### Neo4j connection refused

- Ensure Neo4j is running and `NEO4J_URI` is correct
- Check firewall/ports and credentials

### Slow processing / high cost

- Use a faster/cheaper provider (e.g., Groq or GPT-4o-mini)
- Trim narratives; avoid excessive context
- Disable DEBUG logs in normal operation

## FAQ

**Q: Which columns are required in the CSV?**
A: See [DATA_SPEC.md](DATA_SPEC.md); `Crash_ID`, coordinates, date/time, location fields, `SAE_Autonomy_Level`, `Crash_Severity`, `Narrative`.

**Q: Where do I find outputs?**
A: Under `artifacts/` by default, or `--out_dir` if set. See [OUTPUTS.md](OUTPUTS.md).

**Q: Can I run locally without API keys?**
A: Yes, with `LLM_PROVIDER=ollama` and local models installed. Quality depends on the chosen model.

**Q: How do I persist to Neo4j?**
A: Set `ENABLE_NEO4J=true` in `.env` or pass `--neo4j_enabled`. See [NEO4J_GUIDE.md](NEO4J_GUIDE.md).

**Q: How do I fine-tune the summarizer?**
A: Use [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md). Prepare data, train, then pass `--fine_tuned_model` to `run`.

**Q: How do I open documentation?**
A: `python crashtransformer.py docs` opens the local HTML documentation in your browser.
