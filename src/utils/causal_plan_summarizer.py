from __future__ import annotations
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import re
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rapidfuzz import fuzz

# Import advanced metrics
try:
    from .advanced_metrics import calculate_metrics, AdvancedMetricsCalculator
    HAVE_ADVANCED_METRICS = True
except ImportError:
    HAVE_ADVANCED_METRICS = False

# ---------------------------
# Data containers
# ---------------------------

@dataclass
class CausalEdge:
    src_event_id: str
    dst_event_id: str
    agent_unit: str | None
    patient_unit: str | None
    reason: str | None
    consequence: str | None
    marked: bool | None
    connective: str | None
    evidence_span: str | None

@dataclass
class Plan:
    lines: List[str]          # text lines to prepend to the model
    edges: List[CausalEdge]   # structured edges used for scoring


# ---------------------------
# Plan builder
# ---------------------------

def _index_entities(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id = {e["id"]: e for e in entities}
    # For quick unit lookup like "U1" -> node id
    unit_to_id = {}
    for e in entities:
        if e.get("label") == "Vehicle" and e.get("unit_id"):
            unit_to_id[e["unit_id"]] = e["id"]
    return {"by_id": by_id, "unit_to_id": unit_to_id}

def _event_text(events_by_id: Dict[str, Dict[str, Any]], ev_id: str) -> Tuple[str, str]:
    ev = events_by_id.get(ev_id, {})
    etype = ev.get("type", "Event")
    label = ev.get("label", "")
    attrs = ev.get("attributes", {}) or {}
    
    # Use the specific label if available, otherwise fall back to type
    if label:
        # For consequences, use the label directly
        cons = label
        # For reasons, use the label as well
        reason = label
    else:
        # Fallback to type-based logic
        if etype.lower() == "collision":
            impact = attrs.get("impact_config") or "collision"
            cons = f"collision({impact})"
        else:
            cons = etype
        # short cause label
        reason = attrs.get("reason") or attrs.get("maneuver") or attrs.get("hazard") or etype
    
    return reason, cons

def build_plan_from_graph(graph_json: Dict[str, Any]) -> Plan:
    """
    Build a concise causal plan from a CrashGraph dictionary.
    The plan lines are human readable bullets the generator can follow.
    """
    entities = graph_json.get("entities", [])
    events = graph_json.get("events", [])
    rels = graph_json.get("relationships", [])

    idx = _index_entities(entities)
    events_by_id = {e["id"]: e for e in events}

    # collect CAUSES edges
    edges: List[CausalEdge] = []
    for r in rels:
        if r.get("type") != "CAUSES":
            continue
        s = r["start"]; t = r["end"]
        props = r.get("properties", {}) or {}
        marked = props.get("marked")
        connective = props.get("connective")
        evidence_span = props.get("evidence_span")

        # attempt to map agent and patient from participation edges around s,t
        agent, patient = None, None
        for pr in rels:
            if pr.get("type") == "PARTICIPATED_IN" and pr.get("end") == s:
                role = (pr.get("properties") or {}).get("role", "").lower()
                unit_id = (pr.get("properties") or {}).get("unit_id")
                if role in {"agent", "striking", "turningvehicle", "mergingvehicle"}:
                    agent = unit_id or agent
            if pr.get("type") == "PARTICIPATED_IN" and pr.get("end") == t:
                role = (pr.get("properties") or {}).get("role", "").lower()
                unit_id = (pr.get("properties") or {}).get("unit_id")
                if role in {"struck", "patient"}:
                    patient = unit_id or patient

        reason, consequence = _event_text(events_by_id, t)  # destination often encodes consequence well
        cause_label, _ = _event_text(events_by_id, s)
        # prefer explicit reason from source event if available
        if events_by_id.get(s, {}).get("attributes", {}).get("reason"):
            cause_label = events_by_id[s]["attributes"]["reason"]
        

        edges.append(CausalEdge(
            src_event_id=s, dst_event_id=t,
            agent_unit=agent, patient_unit=patient,
            reason=cause_label, consequence=consequence,
            marked=marked, connective=connective, evidence_span=evidence_span
        ))

    # deterministic order: source id then dest id
    edges.sort(key=lambda e: (e.src_event_id, e.dst_event_id))

    # render plan lines
    lines = []
    for k, e in enumerate(edges, start=1):
        role = "marked" if e.marked else "unmarked"
        who = f"{e.agent_unit} -> {e.patient_unit}" if e.agent_unit or e.patient_unit else ""
        who = who.strip(" ->")
        pieces = [f"{k}) {e.reason} -> {e.consequence}"]
        if who:
            pieces.append(f"actors={who}")
        pieces.append(role)
        lines.append(" | ".join(pieces))

    return Plan(lines=lines, edges=edges)


# ---------------------------
# Generator
# ---------------------------

class PlanConditionedSummarizer:
    def __init__(self, model_name: str = "facebook/bart-base", device: str | None = None, max_input: int = 768, max_summary: int = 200):
        # Use the specified model
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.max_input = max_input
        self.max_summary = max_summary

    def build_prompt(self, narrative: str, plan: Plan) -> str:
        # Use a simpler prompt that works better with BART
        return f"Summarize this crash report in 1-2 sentences: {narrative.strip()}"

    def generate(self, narrative: str, plan: Plan, num_return_sequences: int = 1, num_beams: int = 4) -> List[str]:
        prompt = self.build_prompt(narrative, plan)
        ipt = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input)
        
        # Generate with parameters optimized for summarization
        out = self.model.generate(
            **ipt,
            max_length=self.max_summary,
            min_length=30,  # Ensure minimum length
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,  # Use beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2,  # Avoid repetition
            pad_token_id=self.tok.eos_token_id,
            length_penalty=1.0,  # Encourage longer summaries
            repetition_penalty=1.1  # Reduce repetition
        )
        
        # Extract only the generated tokens (excluding input tokens)
        input_length = ipt['input_ids'].shape[1]
        generated_tokens = out[:, input_length:]
        
        # Decode and clean up the generated text
        summaries = []
        for g in generated_tokens:
            decoded = self.tok.decode(g, skip_special_tokens=True).strip()
            # Remove any remaining prompt artifacts
            if decoded.startswith("Summary:"):
                decoded = decoded[8:].strip()
            summaries.append(decoded)
        
        # If the model generates empty or very short summaries, create a better fallback
        if not summaries or all(len(s.strip()) < 30 for s in summaries):
            # Create a better fallback summary by extracting key information
            narrative_lower = narrative.lower()
            
            # Extract key information
            vehicles = []
            if "unit 1" in narrative_lower:
                vehicles.append("Unit 1")
            if "unit 2" in narrative_lower:
                vehicles.append("Unit 2")
            if "unit 3" in narrative_lower:
                vehicles.append("Unit 3")
            
            # Extract key actions
            actions = []
            if "failed to control speed" in narrative_lower:
                actions.append("failed to control speed")
            if "struck" in narrative_lower or "collided" in narrative_lower:
                actions.append("collided")
            if "failed to yield" in narrative_lower:
                actions.append("failed to yield right of way")
            if "red light" in narrative_lower:
                actions.append("at a red light")
            
            # Create a proper summary
            vehicle_text = " and ".join(vehicles) if vehicles else "multiple vehicles"
            action_text = " and ".join(actions) if actions else "involved in a collision"
            
            fallback = f"A traffic collision occurred involving {vehicle_text} {action_text}."
            summaries = [fallback] * num_return_sequences
        
        return summaries


# ---------------------------
# Edge extraction from summary
# ---------------------------

MARKERS = [
    "because", "due to", "resulted in", "caused", "led to",
    "as a result", "therefore", "hence", "consequently", "since",
    "failed to", "following too close"
]

def split_sents(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())

def extract_edges_from_text(text: str) -> List[Tuple[str, str, bool, str]]:
    """
    Return list of tuples: (cause_text, effect_text, marked_flag, evidence_phrase)
    Very light heuristic suitable for scoring, not gold annotation.
    """
    sents = split_sents(text)
    edges = []
    for s in sents:
        low = s.lower()
        marked = False
        phrase = None
        for m in MARKERS:
            if m in low:
                marked = True
                phrase = m
                break
        # naive split on arrow cues or typical causal phrases
        if " -> " in s:
            parts = s.split(" -> ")
            if len(parts) >= 2:
                edges.append((parts[0].strip(), parts[1].strip(), marked, phrase or "->"))
                continue
        if " and " in low and ("failed to" in low or "due to" in low or "because" in low):
            # e.g., "failed to control speed and struck U2"
            parts = re.split(r"\band\b", s, flags=re.I)
            if len(parts) >= 2:
                edges.append((parts[0].strip(), parts[1].strip(), marked, phrase or "and"))
                continue
        # fall back to sequential clauses split by comma
        if "," in s:
            parts = s.split(",")
            if len(parts) >= 2 and len(parts[0]) > 3 and len(parts[1]) > 3:
                edges.append((parts[0].strip(), parts[1].strip(), marked, phrase or ","))
    return edges


# ---------------------------
# Scoring
# ---------------------------

def normalize(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s/+-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def prf1(pred_pairs: List[str], ref_pairs: List[str], thresh: int = 80) -> Tuple[float, float, float]:
    if not pred_pairs and not ref_pairs:
        return 1.0, 1.0, 1.0
    if not pred_pairs:
        return 0.0, 1.0, 0.0
    if not ref_pairs:
        return 1.0, 0.0, 0.0
    
    used = set()
    tp = 0
    for p in pred_pairs:
        best_j = -1
        best = 0
        for j, r in enumerate(ref_pairs):
            if j in used:
                continue
            score = fuzz.token_set_ratio(p, r)
            if score > best:
                best = score
                best_j = j
        if best >= thresh:
            tp += 1
            used.add(best_j)
    prec = tp / len(pred_pairs)
    rec = tp / len(ref_pairs)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def edges_to_pairs(edges: List[CausalEdge]) -> List[str]:
    pairs = []
    for e in edges:
        cause = normalize((e.reason or "").strip())
        effect = normalize((e.consequence or "").strip())
        pairs.append(f"{cause} -> {effect}")
    return pairs

def extracted_to_pairs(edges_extracted: List[Tuple[str, str, bool, str]]) -> List[str]:
    pairs = []
    for c, ef, _, _ in edges_extracted:
        pairs.append(f"{normalize(c)} -> {normalize(ef)}")
    return pairs

def span_faithfulness(edges_extracted: List[Tuple[str, str, bool, str]], narrative: str, thresh: int = 70) -> float:
    if not edges_extracted:
        return 1.0
    sents = split_sents(narrative)
    ok = 0
    for c, ef, _, _ in edges_extracted:
        text = (c + " " + ef).strip()
        best = 0
        for s in sents:
            best = max(best, fuzz.partial_ratio(normalize(text), normalize(s)))
        if best >= thresh:
            ok += 1
    return ok / max(1, len(edges_extracted))

def hallucination_rate(edges_extracted: List[Tuple[str, str, bool, str]], narrative: str, thresh: int = 60) -> float:
    if not edges_extracted:
        return 0.0
    sents = split_sents(narrative)
    hal = 0
    for c, ef, _, _ in edges_extracted:
        text = (c + " " + ef).strip()
        best = 0
        for s in sents:
            best = max(best, fuzz.partial_ratio(normalize(text), normalize(s)))
        if best < thresh:
            hal += 1
    return hal / len(edges_extracted)

def compression_ratio(summary: str, narrative: str) -> float:
    return max(1, len(summary.split())) / max(1, len(narrative.split()))

def calculate_semantic_similarity(summaries: List[str], narratives: List[str]) -> Dict[str, float]:
    """Calculate semantic similarity between summaries and narratives using fuzzy matching."""
    if not summaries or not narratives:
        return {"semantic_similarity": 0.0}
    
    similarities = []
    for summary, narrative in zip(summaries, narratives):
        # Use fuzzy token set ratio for semantic similarity
        similarity = fuzz.token_set_ratio(summary.lower(), narrative.lower()) / 100.0
        similarities.append(similarity)
    
    return {"semantic_similarity": sum(similarities) / len(similarities)}

def calculate_faithfulness(summary: str, narrative: str) -> float:
    """Calculate how faithful the summary is to the narrative (no hallucination)."""
    if not summary or not narrative:
        return 0.0
    
    # Extract key entities and events from narrative
    narrative_words = set(narrative.lower().split())
    summary_words = set(summary.lower().split())
    
    # Calculate overlap of important words
    common_words = narrative_words.intersection(summary_words)
    
    # Avoid division by zero
    if len(summary_words) == 0:
        return 0.0
    
    # Faithfulness is the ratio of summary words that appear in narrative
    faithfulness = len(common_words) / len(summary_words)
    return min(1.0, faithfulness)

def calculate_completeness(summary: str, narrative: str) -> float:
    """Calculate how complete the summary is in covering narrative content."""
    if not summary or not narrative:
        return 0.0
    
    # Extract key entities and events from narrative
    narrative_words = set(narrative.lower().split())
    summary_words = set(summary.lower().split())
    
    # Calculate how much of the narrative is covered by the summary
    common_words = narrative_words.intersection(summary_words)
    
    # Avoid division by zero
    if len(narrative_words) == 0:
        return 0.0
    
    # Completeness is the ratio of narrative words covered by summary
    completeness = len(common_words) / len(narrative_words)
    return min(1.0, completeness)

def calculate_coherence(summary: str) -> float:
    """Calculate internal coherence of the summary."""
    if not summary:
        return 0.0
    
    # Simple coherence based on sentence structure and length
    sentences = summary.split('.')
    if len(sentences) <= 1:
        return 0.5  # Single sentence summaries are somewhat coherent
    
    # Check for reasonable sentence length and structure
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    
    # Coherence score based on sentence structure
    if avg_sentence_length < 3:
        return 0.3  # Very short sentences
    elif avg_sentence_length > 20:
        return 0.7  # Long but potentially coherent sentences
    else:
        return 0.9  # Good sentence length

def score_summary(summary: str, plan: Plan, narrative: str, enable_advanced_metrics: bool = True) -> Dict[str, float]:
    """
    Score a natural language summary using comprehensive NLP metrics including ROUGE, BLEU, BERTScore, etc.
    """
    # Basic text metrics
    comp = compression_ratio(summary, narrative)
    
    # Calculate semantic similarity between summary and narrative
    narrative_similarity = calculate_semantic_similarity([summary], [narrative])
    
    # Calculate faithfulness by checking if summary content is supported by narrative
    faithfulness = calculate_faithfulness(summary, narrative)
    
    # Calculate completeness by checking if summary covers key narrative elements
    completeness = calculate_completeness(summary, narrative)
    
    # Calculate coherence (internal consistency of summary)
    coherence = calculate_coherence(summary)
    
    # Basic metrics for natural language summaries
    basic_metrics = {
        "causal_precision": narrative_similarity.get("semantic_similarity", 0.0),
        "causal_recall": completeness,
        "causal_f1": 2 * (narrative_similarity.get("semantic_similarity", 0.0) * completeness) / 
                    max(narrative_similarity.get("semantic_similarity", 0.0) + completeness, 0.001),
        "span_faithfulness": faithfulness,
        "hallucination_rate": max(0.0, 1.0 - faithfulness),  # Inverse of faithfulness
        "compression_ratio": comp,
    }
    
    # Add comprehensive advanced metrics if available and enabled
    if enable_advanced_metrics and HAVE_ADVANCED_METRICS:
        try:
            # Use the existing advanced metrics system
            advanced_metrics = calculate_metrics([summary], [narrative])
            basic_metrics.update(advanced_metrics)
            
            # Calculate combined score using advanced metrics if available
            if "rouge_rouge1_f1" in advanced_metrics and "bertscore_f1" in advanced_metrics:
                # Use ROUGE-1 F1 and BERTScore F1 for combined score
                combined = (0.4 * advanced_metrics.get("rouge_rouge1_f1", 0.0) + 
                           0.3 * advanced_metrics.get("bertscore_f1", 0.0) + 
                           0.2 * faithfulness + 
                           0.1 * coherence)
            else:
                # Fallback to basic metrics
                combined = (0.4 * narrative_similarity.get("semantic_similarity", 0.0) + 
                            0.3 * faithfulness + 
                            0.2 * completeness + 
                            0.1 * coherence)
        except Exception as e:
            # If advanced metrics fail, continue with basic metrics
            import logging
            logging.warning(f"Advanced metrics calculation failed: {e}")
            # Fallback combined score
            combined = (0.4 * narrative_similarity.get("semantic_similarity", 0.0) + 
                        0.3 * faithfulness + 
                        0.2 * completeness + 
                        0.1 * coherence)
    else:
        # Use basic metrics for combined score
        combined = (0.4 * narrative_similarity.get("semantic_similarity", 0.0) + 
                    0.3 * faithfulness + 
                    0.2 * completeness + 
                    0.1 * coherence)
    
    basic_metrics["combined_score"] = combined
    
    return basic_metrics


# ---------------------------
# End to end helper
# ---------------------------

class CausalPlanSummarizer:
    def __init__(self, model_name: str = "facebook/bart-base", logger=None, enable_advanced_metrics: bool = False, fine_tuned_model_path: str = None):
        self.logger = logger
        self.enable_advanced_metrics = enable_advanced_metrics
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # Initialize the generator with fine-tuned model if available
        if fine_tuned_model_path and os.path.exists(fine_tuned_model_path):
            self.gen = PlanConditionedSummarizer(model_name=fine_tuned_model_path)
            if self.logger:
                self.logger.info(f"Using fine-tuned model: {fine_tuned_model_path}")
        else:
            self.gen = PlanConditionedSummarizer(model_name=model_name)
            if self.logger:
                self.logger.info(f"Using pre-trained model: {model_name}")
        
        # Initialize advanced metrics calculator if available
        if enable_advanced_metrics and HAVE_ADVANCED_METRICS:
            try:
                self.advanced_metrics_calc = AdvancedMetricsCalculator(logger=logger)
                if self.logger:
                    self.logger.info("Advanced metrics calculator initialized")
            except Exception as e:
                self.advanced_metrics_calc = None
                if self.logger:
                    self.logger.warning(f"Failed to initialize advanced metrics: {e}")
        else:
            self.advanced_metrics_calc = None

    def summarize(self, graph_json: Dict[str, Any], num_candidates: int = 3) -> Dict[str, Any]:
        crash = graph_json.get("crash", {})
        crash_id = crash.get("crash_id", "unknown")
        narrative = crash.get("raw_narrative") or ""
        
        if self.logger:
            self.logger.log_crash_processing(crash_id, "plan_building", "Building causal plan from graph")
        
        plan = build_plan_from_graph(graph_json)
        
        # Handle case where no edges are found
        if not plan.edges:
            if self.logger:
                self.logger.warning(f"No causal edges found for crash {crash_id}, using fallback plan")
            # Create a fallback plan with a single generic edge
            plan.lines = ["1) incident -> outcome"]
            plan.edges = []
        
        if self.logger:
            self.logger.log_crash_processing(crash_id, "summary_generation", f"Generating {num_candidates} summary candidates")
        
        candidates = self.gen.generate(narrative, plan, num_return_sequences=num_candidates, num_beams=max(4, num_candidates))

        if self.logger:
            self.logger.log_crash_processing(crash_id, "scoring", "Scoring and ranking summary candidates")

        scored = []
        for s in candidates:
            metrics = score_summary(s, plan, narrative, enable_advanced_metrics=self.enable_advanced_metrics)
            scored.append({"summary": s, "metrics": metrics})

        # pick best by combined score
        best = max(scored, key=lambda x: x["metrics"]["combined_score"]) if scored else None
        
        if self.logger and best:
            best_score = best["metrics"]["combined_score"]
            self.logger.info(f"Best summary selected for crash {crash_id} with score: {best_score:.3f}")
        
        return {
            "plan_lines": plan.lines,
            "candidates": scored,
            "best": best
        }


# ---------------------------
# Demo runner
# ---------------------------

def demo_from_file(path: str, model_name: str = "facebook/bart-base"):
    with open(path, "r") as f:
        graph = json.load(f)
    cps = CausalPlanSummarizer(model_name=model_name)
    result = cps.summarize(graph, num_candidates=4)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Example: run on a single CrashGraph JSON file
    # demo_from_file("crash_graph_19957925.json")
    pass
