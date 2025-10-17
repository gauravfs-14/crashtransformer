import json
from typing import List, Dict, Optional

class CostTracker:
    """
    Tracks inference time, token usage, and computes cost per 1,000 summaries.
    Supports both API-based (token-priced per 1M tokens) and local GPU-based models.
    """

    def __init__(
        self,
        model_name: str,
        price_per_1k_input: Optional[float] = None,
        price_per_1k_output: Optional[float] = None,
        gpu_hourly_rate: Optional[float] = None,
        logger=None
    ):
        """
        Args:
            model_name: name of model ("BART", "Llama3", "Ollama", etc.)
            price_per_1k_input: API input token price per 1M tokens (USD)
            price_per_1k_output: API output token price per 1M tokens (USD)
            gpu_hourly_rate: cost of GPU per hour (USD) for local models
        """
        self.model_name = model_name
        self.price_in = price_per_1k_input
        self.price_out = price_per_1k_output
        self.gpu_rate = gpu_hourly_rate
        self.records: List[Dict] = []
        self.logger = logger

    def log_summary(
        self,
        narrative_id: str,
        input_tokens: int,
        output_tokens: int,
        start_time: float,
        end_time: float
    ):
        """Log one summarization event."""
        runtime = end_time - start_time
        self.records.append({
            "narrative_id": narrative_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "runtime_sec": runtime
        })
        
        if self.logger:
            self.logger.info(f"Cost tracking - Narrative {narrative_id}: {input_tokens} input tokens, {output_tokens} output tokens, {runtime:.2f}s runtime")

    def compute_costs(self) -> Dict[str, float]:
        """Compute total and per-1k summary costs."""
        total_input_tokens = sum(r["input_tokens"] for r in self.records)
        total_output_tokens = sum(r["output_tokens"] for r in self.records)
        total_runtime = sum(r["runtime_sec"] for r in self.records)
        n = len(self.records)

        result = {"model": self.model_name, "summaries": n}

        # Case 1: API-based (token-priced)
        if self.price_in is not None and self.price_out is not None:
            total_cost = (
                (total_input_tokens / 1000000) * self.price_in
                + (total_output_tokens / 1000000) * self.price_out
            )
            cost_per_1k_summaries = (total_cost / n) * 1000
            result.update({
                "total_cost_usd": total_cost,
                "cost_per_1k_summaries_usd": cost_per_1k_summaries,
                "total_tokens": total_input_tokens + total_output_tokens,
            })

        # Case 2: Local GPU-based
        elif self.gpu_rate is not None:
            total_hours = total_runtime / 3600
            total_cost = total_hours * self.gpu_rate
            cost_per_1k_summaries = (total_cost / n) * 1000
            result.update({
                "total_cost_usd": total_cost,
                "cost_per_1k_summaries_usd": cost_per_1k_summaries,
                "total_runtime_hours": total_hours,
            })
        else:
            result["error"] = "No pricing info provided."

        return result

    def export(self, path="cost_report.json"):
        cost_data = self.compute_costs()
        with open(path, "w") as f:
            json.dump(cost_data, f, indent=4)
        
        if self.logger:
            self.logger.info(f"Cost report exported to {path}")
            self.logger.info(f"Total cost: ${cost_data.get('total_cost_usd', 0):.2f}")
            self.logger.info(f"Cost per 1k summaries: ${cost_data.get('cost_per_1k_summaries_usd', 0):.2f}")
        else:
            print(f"✅ Saved cost summary to {path}")
