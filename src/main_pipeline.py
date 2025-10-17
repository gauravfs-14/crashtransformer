# main_pipeline.py

import os
import json
import time
import csv
from typing import Dict, Any, Iterable, List

from utils import analyze_crash_with_usage, analyze_crash_with_summary_and_usage, CausalPlanSummarizer, Neo4jSink, config
from misc import get_logger

try:
    from utils import CostTracker
    HAVE_COST = True
except Exception:
    HAVE_COST = False

def format_extraction_input(row: Dict[str, Any]) -> str:
    return (
        f"Crash id: {row.get('Crash_ID')}\n"
        f"Latitude: {row.get('Latitude')}, Longitude: {row.get('Longitude')}\n"
        f"CrashDate: {row.get('CrashDate')}, DayOfWeek: {row.get('DayOfWeek')}, CrashTime: {row.get('CrashTime')}\n"
        f"County: {row.get('County')}, City: {row.get('City')}\n"
        f"SAE_Autonomy_Level: {row.get('SAE_Autonomy_Level')}, "
        f"Crash_Severity: {row.get('Crash_Severity')}\n"
        f"Narrative: {row.get('Narrative')}"
    )


def run_one(row: Dict[str, Any], cps: CausalPlanSummarizer, track_tokens: bool = True, logger=None, llm_provider: str = None, llm_model: str = None, llm_api_key: str = None, skip_llm: bool = False) -> Dict[str, Any]:
    """
    Returns a dict with graph, plan, best summary, metrics, and usage stats
    for both extraction and summarization stages.
    """
    crash_id = row.get('Crash_ID', 'unknown')
    
    # A) LLM generates both graph and summary in one call
    if logger:
        logger.log_crash_processing(crash_id, "llm_extraction", "Starting LLM graph and summary extraction")
    formatted = format_extraction_input(row)
    
    try:
        print(f"🔧 DEBUG: Calling analyze_crash_with_summary_and_usage with provider: {llm_provider}, model: {llm_model}")
        if logger:
            logger.info(f"Calling analyze_crash_with_summary_and_usage with provider: {llm_provider}, model: {llm_model}")
        graph_obj, llm_summary, llm_usage = analyze_crash_with_summary_and_usage(formatted, row, logger=logger, provider=llm_provider, model=llm_model, api_key=llm_api_key)
        graph_json = json.loads(graph_obj.json())
    except Exception as e:
        if logger:
            logger.error(f"LLM extraction failed for crash {crash_id}: {str(e)}")
        else:
            print(f"❌ LLM extraction failed for crash {crash_id}: {str(e)}")
        raise
    
    if logger:
        logger.log_llm_call(crash_id, llm_usage.provider, llm_usage.model or "unknown", 
                           llm_usage.total_tokens, llm_usage.total_cost_usd)

    # B) Baseline model summarization for comparison
    if logger:
        logger.log_crash_processing(crash_id, "baseline_summarization", "Starting baseline model summarization")
    t0 = time.time()
    
    try:
        result = cps.summarize(graph_json, num_candidates=4)
    except Exception as e:
        if logger:
            logger.error(f"Baseline summarization failed for crash {crash_id}: {str(e)}")
        else:
            print(f"❌ Baseline summarization failed for crash {crash_id}: {str(e)}")
        raise
    
    t1 = time.time()
    
    if logger and result.get("best"):
        best_score = result["best"].get("metrics", {}).get("combined_score", 0.0)
        logger.log_summary_generation(crash_id, "plan_conditioned", 4, best_score)

    # C) summarizer token counts
    sum_input_tokens = sum_output_tokens = None
    if track_tokens:
        try:
            tok = cps.gen.tok
            plan_block = "\n".join(result["plan_lines"]) if result["plan_lines"] else "1) main cause -> main outcome"
            prompt = (
                "You are a crash analyst. Write a concise causal summary in 1 to 3 sentences. "
                "Cover the listed edges and keep claims faithful to the narrative.\n"
                "Plan:\n"
                f"{plan_block}\n"
                "Narrative:\n"
                f"{graph_json['crash'].get('raw_narrative', '').strip()}\n"
                "Summary:"
            )
            sum_input_tokens = len(tok.encode(prompt))
            sum_output_tokens = len(tok.encode(result["best"]["summary"])) if result.get("best") else 0
        except Exception:
            pass

    return {
        "Crash_ID": row.get("Crash_ID"),
        # raw graph
        "graph": graph_json,
        # plan and best summary (baseline model)
        "plan_lines": result["plan_lines"],
        "best_summary": result["best"]["summary"] if result.get("best") else "",
        "metrics": result["best"]["metrics"] if result.get("best") else {},
        # LLM summary for comparison and training data
        "llm_summary": llm_summary,
        "llm_runtime_sec": llm_usage.runtime_sec,
        "llm_total_tokens": llm_usage.total_tokens,
        "llm_prompt_tokens": llm_usage.prompt_tokens,
        "llm_completion_tokens": llm_usage.completion_tokens,
        "llm_cost_usd": llm_usage.total_cost_usd,
        "llm_provider": llm_usage.provider,
        "llm_model": llm_usage.model,
        # baseline model usage
        "summarizer_runtime_sec": t1 - t0,
        "summarizer_input_tokens": sum_input_tokens,
        "summarizer_output_tokens": sum_output_tokens,
    }


def _write_jsonl(path: str, rows: List[dict], append: bool = False) -> None:
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_batch(rows: Iterable[Dict[str, Any]],
              model_name: str = "facebook/bart-base",
              fine_tuned_model_path: str = None,
              cost_mode: str = "local",
              gpu_hourly_rate: float = 1.50,
              api_price_in: float = None,
              api_price_out: float = None,
              out_dir: str = "artifacts",
              logger=None,
              neo4j_sink=None,
              llm_provider: str = None,
              llm_model: str = None,
              llm_api_key: str = None) -> List[Dict[str, Any]]:

    out_dir = os.path.join(out_dir, model_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    
    if logger:
        logger.info(f"Starting batch processing with model: {model_name}")
        logger.info(f"Output directory: {out_dir}")
        if neo4j_sink:
            logger.info("Neo4j database storage enabled")

    cps = CausalPlanSummarizer(
        model_name=model_name, 
        logger=logger, 
        fine_tuned_model_path=fine_tuned_model_path
    )

    tracker = None
    if HAVE_COST:
        if cost_mode == "api":
            tracker = CostTracker(
                model_name=f"PlanCond-{model_name}",
                price_per_1k_input=api_price_in,
                price_per_1k_output=api_price_out,
                logger=logger
            )
        else:
            tracker = CostTracker(
                model_name=f"PlanCond-{model_name}",
                gpu_hourly_rate=gpu_hourly_rate,
                logger=logger
            )

    results = []
    graphs_agg = []
    summaries_agg = []

    successful = 0
    failed = 0
    total_cost = 0.0
    
    for row in rows:
        try:
            out = run_one(row, cps, track_tokens=True, logger=logger, llm_provider=llm_provider, llm_model=llm_model, llm_api_key=llm_api_key)
            results.append(out)
            successful += 1
            total_cost += out.get("extract_cost_usd", 0.0)
        except Exception as e:
            failed += 1
            if logger:
                logger.error(f"Failed to process crash {row.get('Crash_ID', 'unknown')}: {str(e)}")
            else:
                print(f"❌ Failed to process crash {row.get('Crash_ID', 'unknown')}: {str(e)}")
            
            # If we have too many failures, stop the pipeline
            if failed > successful and failed >= 3:
                error_msg = f"Too many failures ({failed} failures, {successful} successful). Stopping pipeline."
                if logger:
                    logger.error(error_msg)
                else:
                    print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            continue

        cid = out["Crash_ID"]

        # aggregated raw graph with LLM summary
        graph_with_summary = {"Crash_ID": cid, **out["graph"]}
        graph_with_summary["llm_summary"] = out["llm_summary"]  # Add LLM summary to graph
        graphs_agg.append(graph_with_summary)

        # aggregated summary record
        summary_record = {
            "Crash_ID": cid,
            "plan_lines": out["plan_lines"],
            "best_summary": out["best_summary"],
            "llm_summary": out["llm_summary"],  # Add LLM summary
            "metrics": out["metrics"],
            # extraction usage
            "extract_runtime_sec": out["llm_runtime_sec"],
            "extract_total_tokens": out["llm_total_tokens"],
            "extract_prompt_tokens": out["llm_prompt_tokens"],
            "extract_completion_tokens": out["llm_completion_tokens"],
            "extract_cost_usd": out["llm_cost_usd"],
            "extract_provider": out["llm_provider"],
            "extract_model": out["llm_model"],
            # summarizer usage
            "summarizer_runtime_sec": out["summarizer_runtime_sec"],
            "summarizer_input_tokens": out["summarizer_input_tokens"],
            "summarizer_output_tokens": out["summarizer_output_tokens"],
        }
        summaries_agg.append(summary_record)

        # Store in Neo4j if enabled
        if neo4j_sink:
            try:
                if logger:
                    logger.log_crash_processing(cid, "neo4j_storage", "Storing crash graph in Neo4j")
                neo4j_sink.upsert_crash_graph(out["graph"])
                
                if logger:
                    logger.log_crash_processing(cid, "neo4j_storage", "Storing summary in Neo4j")
                neo4j_sink.upsert_summary(summary_record)
                
                if logger:
                    logger.info(f"Successfully stored crash {cid} in Neo4j database")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to store crash {cid} in Neo4j: {str(e)}")
                else:
                    print(f"Warning: Failed to store crash {cid} in Neo4j: {str(e)}")

        # summarizer cost tracker per item
        if HAVE_COST and tracker is not None:
            tracker.log_summary(
                narrative_id=str(cid),
                input_tokens=out.get("summarizer_input_tokens") or 0,
                output_tokens=out.get("summarizer_output_tokens") or 0,
                start_time=time.time() - out["summarizer_runtime_sec"],
                end_time=time.time()
            )

    # write aggregated JSONL
    _write_jsonl(os.path.join(out_dir, "crash_graphs.jsonl"), graphs_agg)
    _write_jsonl(os.path.join(out_dir, "crash_summaries.jsonl"), summaries_agg)

    # optional JSON arrays
    with open(os.path.join(out_dir, "crash_graphs.json"), "w", encoding="utf-8") as f:
        json.dump(graphs_agg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "crash_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries_agg, f, indent=2, ensure_ascii=False)

    # CSV for quick analysis
    csv_path = os.path.join(out_dir, "summaries_metrics.csv")
    if results:
        # Get all available metrics from the first result
        all_metrics = set()
        for r in results:
            metrics = r.get("metrics", {})
            all_metrics.update(metrics.keys())
        
        # Define base fieldnames
        base_fieldnames = [
            "Crash_ID",
            "best_summary",
            # extraction usage (keep old names for compatibility)
            "extract_runtime_sec",
            "extract_total_tokens",
            "extract_prompt_tokens",
            "extract_completion_tokens",
            "extract_cost_usd",
            "extract_provider",
            "extract_model",
            # summarizer usage
            "summarizer_runtime_sec",
            "summarizer_input_tokens",
            "summarizer_output_tokens",
        ]
        
        # Add all metrics dynamically
        fieldnames = base_fieldnames + sorted(list(all_metrics))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                m = r.get("metrics", {})
                
                # Create row with base fields
                row = {
                    "Crash_ID": r["Crash_ID"],
                    "best_summary": r["best_summary"],
                    # extraction usage
                    "extract_runtime_sec": r.get("llm_runtime_sec"),
                    "extract_total_tokens": r.get("llm_total_tokens"),
                    "extract_prompt_tokens": r.get("llm_prompt_tokens"),
                    "extract_completion_tokens": r.get("llm_completion_tokens"),
                    "extract_cost_usd": r.get("llm_cost_usd"),
                    "extract_provider": r.get("llm_provider"),
                    "extract_model": r.get("llm_model"),
                    # summarizer usage
                    "summarizer_runtime_sec": r.get("summarizer_runtime_sec"),
                    "summarizer_input_tokens": r.get("summarizer_input_tokens"),
                    "summarizer_output_tokens": r.get("summarizer_output_tokens"),
                }
                
                # Add all metrics dynamically
                for metric_name in all_metrics:
                    row[metric_name] = m.get(metric_name)
                
                w.writerow(row)

    if HAVE_COST and tracker is not None:
        tracker.export(os.path.join(out_dir, "cost_report.json"))
    
    if logger:
        total_crashes = successful + failed
        logger.log_batch_completion(total_crashes, successful, failed, total_cost)

    # Check if we have any successful results
    if not results:
        error_msg = "No successful processing results. Pipeline failed completely."
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Check if we have too many failures
    if failed > successful:
        error_msg = f"Pipeline had more failures ({failed}) than successes ({successful}). This indicates a systematic issue."
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Run CrashTransformer pipeline end to end with comprehensive logging.")
    parser.add_argument("--csv", type=str, help="Path to input CSV with columns: Crash_ID, Latitude, Longitude, CrashDate, DayOfWeek, CrashTime, County, City, SAE_Autonomy_Level, Crash_Severity, Narrative")
    parser.add_argument("--xlsx", type=str, help="Path to input XLSX file with columns: Crash_ID, Latitude, Longitude, CrashDate, DayOfWeek, CrashTime, County, City, SAE_Autonomy_Level, Crash_Severity, Narrative")
    parser.add_argument("--model", type=str, default="facebook/bart-base", help="HF model name for plan-conditioned summarizer, e.g., facebook/bart-base or t5-base")
    parser.add_argument("--fine_tuned_model", type=str, default=None, help="Path to fine-tuned model directory")
    parser.add_argument("--worksheet", type=str, default="Narr_CrLev", help="Worksheet name in XLSX file")
    parser.add_argument("--out_dir", type=str, default=None, help="Root output directory (default: from env or 'artifacts')")
    parser.add_argument("--cost_mode", type=str, default="local", choices=["local", "api"], help="How to compute summarizer cost")
    parser.add_argument("--gpu_rate", type=float, default=1.50, help="GPU hourly rate if cost_mode=local")
    parser.add_argument("--api_price_in", type=float, default=None, help="Input token price (USD per 1k) if cost_mode=api")
    parser.add_argument("--api_price_out", type=float, default=None, help="Output token price (USD per 1k) if cost_mode=api")
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL outputs instead of overwriting")
    parser.add_argument("--batch_models", type=str, nargs="*", help="Optional list of models to run, overrides --model if provided")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM extraction and use existing graphs")
    
    # Logger arguments (non-sensitive)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory (default: from env)")
    parser.add_argument("--log_level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Override log level (default: from env)")
    parser.add_argument("--no_logs", action="store_true", help="Disable logging completely")
    
    # Neo4j arguments (non-sensitive)
    parser.add_argument("--neo4j_uri", type=str, default=None, help="Override Neo4j URI (default: from env)")
    parser.add_argument("--neo4j_enabled", action="store_true", help="Enable Neo4j database storage (overrides env)")
    
    # LLM Provider arguments (non-sensitive)
    parser.add_argument("--llm_provider", type=str, default=None, 
                       choices=["openai", "anthropic", "google", "gemini", "groq", "ollama", "grok"],
                       help="Override LLM provider (default: from env)")
    parser.add_argument("--llm_model", type=str, default=None, 
                       help="Override LLM model (default: from env)")
    parser.add_argument("--ollama_base_url", type=str, default=None, 
                       help="Override Ollama base URL (default: from env)")

    args = parser.parse_args()

    def df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
        needed = ["Crash_ID", "Latitude", "Longitude", "CrashDate", "DayOfWeek", "CrashTime",
                  "County", "City", "SAE_Autonomy_Level", "Crash_Severity", "Narrative"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return df[needed].to_dict(orient="records")

    if args.batch_models and len(args.batch_models) > 0:
        models = args.batch_models
    else:
        models = [args.model]

    if args.csv:
        df = pd.read_csv(args.csv)
        rows = df_to_rows(df)
    elif args.xlsx and args.worksheet:
        df = pd.read_excel(args.xlsx, sheet_name=args.worksheet)
        rows = df_to_rows(df)
    else:
        # small sanity demo if no CSV is provided
        rows = [
            {
                "Crash_ID": "19955047",
                "Latitude": 26.15526348,
                "Longitude": -97.99060556,
                "CrashDate": "1/1/24",
                "DayOfWeek": "MON",
                "CrashTime": "2:06 AM",
                "County": "Hidalgo",
                "City": "Weslaco",
                "SAE_Autonomy_Level": "2",
                "Crash_Severity": "Not Injured",
                "Narrative": "unit 2 was stationary in the northbound lane ... unit 1 failed to control speed and struck unit 2 on the back end."
            },
            {
                "Crash_ID": "19955369",
                "Latitude": 32.92553538,
                "Longitude": -96.80385791,
                "CrashDate": "1/2/24",
                "DayOfWeek": "TUE",
                "CrashTime": "3:37 PM",
                "County": "Dallas",
                "City": "Dallas",
                "SAE_Autonomy_Level": "1",
                "Crash_Severity": "Not Injured",
                "Narrative": "***bwc available***unit 1 was traveling w/b ... unit 1 failed to control speed and collided with unit 2. unit 1's fd collided with unit 2's bd."
            },
            {
                "Crash_ID": "19956614",
                "Latitude": 29.69813481,
                "Longitude": -95.20085379,
                "CrashDate": "1/3/24",
                "DayOfWeek": "WED",
                "CrashTime": "6:49 AM",
                "County": "Harris",
                "City": "Pasadena",
                "SAE_Autonomy_Level": "1",
                "Crash_Severity": "Not Injured",
                "Narrative": "unit 1 was in a private drive ... failed to yield the right of way from a private drive and collided with unit 2."
            }
        ]

    # Load configuration from environment with CLI overrides
    llm_config = config.get_llm_config(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key=None  # Always use environment variables for API keys
    )
    
    # Use centralized artifacts directory if not specified
    if args.out_dir is None:
        args.out_dir = config.output_dir
    
    db_config = config.get_database_config(
        uri=args.neo4j_uri,
        enabled=args.neo4j_enabled
    )
    
    logging_config = config.get_logging_config(
        level=args.log_level,
        directory=args.log_dir
    )
    
    # Validate configuration
    validation = config.validate_config()
    if not validation["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        exit(1)
    
    # Initialize logger
    if args.no_logs or not logging_config.enabled:
        logger = None
        print("Logging disabled")
    else:
        logger = get_logger(log_dir=logging_config.directory, log_level=logging_config.level)
        logger.info("CrashTransformer Pipeline started")
        logger.info(f"Log directory: {logging_config.directory}")
        logger.info(f"Log level: {logging_config.level}")
        logger.info(f"LLM Provider: {llm_config.provider}")
        logger.info(f"LLM Model: {llm_config.model}")
        print(f"📝 Logging enabled - Log file: {logger.get_log_file_path()}")
        print(f"🤖 LLM Provider: {llm_config.provider} ({llm_config.model})")

    # Initialize Neo4j if enabled
    neo4j_sink = None
    if db_config.enabled:
        try:
            neo4j_sink = Neo4jSink(db_config.uri, db_config.user, db_config.password)
            neo4j_sink.ensure_constraints()
            if logger:
                logger.info(f"Neo4j connection established: {db_config.uri}")
            print(f"🗄️ Neo4j database enabled: {db_config.uri}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to connect to Neo4j: {str(e)}")
            print(f"❌ Failed to connect to Neo4j: {str(e)}")
            print("Continuing without Neo4j storage...")
            neo4j_sink = None
    
    # run one or many models
    for m in models:
        try:
            # when appending, we just re-use the same files and append JSONL lines
            # the CSV is still overwritten per run to avoid mixed models in one table
            results = run_batch(
                rows=rows,
                model_name=m,
                fine_tuned_model_path=args.fine_tuned_model,
                cost_mode=args.cost_mode,
                gpu_hourly_rate=args.gpu_rate,
                api_price_in=args.api_price_in,
                api_price_out=args.api_price_out,
                out_dir=args.out_dir,
                logger=logger,
                neo4j_sink=neo4j_sink,
                llm_provider=llm_config.provider,
                llm_model=llm_config.model,
                llm_api_key=llm_config.api_key
            )
        except Exception as e:
            error_msg = f"Pipeline failed for model {m}: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(f"❌ {error_msg}")
            print(f"❌ Pipeline failed: {str(e)}")
            exit(1)

        # If you want true append semantics for JSONL, call _write_jsonl with append=True inside run_batch.
        # Here we only print a tiny summary for quick feedback.
        if results:
            print(json.dumps({
                "model": m,
                "items": len(results),
                "first_keys": list(results[0].keys())
            }, indent=2))
    
    # Cleanup Neo4j connection
    if neo4j_sink:
        try:
            neo4j_sink.close()
            if logger:
                logger.info("Neo4j connection closed")
            print("🗄️ Neo4j connection closed")
        except Exception as e:
            if logger:
                logger.error(f"Error closing Neo4j connection: {str(e)}")
            print(f"Warning: Error closing Neo4j connection: {str(e)}")

def main():
    """Main entry point for the pipeline"""
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="CrashTransformer Pipeline")
    parser.add_argument("--csv", help="Input CSV file")
    parser.add_argument("--xlsx", help="Input XLSX file")
    parser.add_argument("--llm_provider", help="LLM provider")
    parser.add_argument("--llm_model", help="LLM model")
    parser.add_argument("--neo4j_enabled", action="store_true", help="Enable Neo4j")
    parser.add_argument("--neo4j_uri", help="Neo4j URI")
    parser.add_argument("--log_level", default="INFO", help="Log level")
    parser.add_argument("--log_dir", help="Log directory")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--model", help="Summarization model")
    parser.add_argument("--batch_models", nargs="+", help="Multiple models")
    parser.add_argument("--fine_tuned_model", help="Fine-tuned model path")
    parser.add_argument("--cost_mode", help="Cost calculation mode")
    
    args = parser.parse_args()
    
    # Load configuration and set LLM parameters if not provided
    from .utils.config import config
    if args.llm_provider is None:
        args.llm_provider = config.get_llm_config().provider
    if args.llm_model is None:
        args.llm_model = config.get_llm_config().model
    
    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.xlsx:
        df = pd.read_excel(args.xlsx)
    else:
        print("Error: Must specify either --csv or --xlsx")
        return
    
    rows = df.to_dict('records')
    
    # Handle multiple models
    if args.batch_models:
        for model in args.batch_models:
            print(f"Running pipeline with model: {model}")
            run_batch(
                rows=rows,
                model_name=model,
                fine_tuned_model_path=args.fine_tuned_model,
                out_dir=args.out_dir or "artifacts",
                llm_provider=args.llm_provider,
                llm_model=args.llm_model
            )
    else:
        # Single model
        model_name = args.model or "facebook/bart-base"
        print(f"Running pipeline with model: {model_name}")
        run_batch(
            rows=rows,
            model_name=model_name,
            fine_tuned_model_path=args.fine_tuned_model,
            out_dir=args.out_dir or "artifacts",
            llm_provider=args.llm_provider,
            llm_model=args.llm_model
        )

if __name__ == "__main__":
    main()
