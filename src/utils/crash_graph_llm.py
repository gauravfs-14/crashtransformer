# crash_graph_llm.py

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import time


from .llm_providers import LLMProviderFactory, LLMUsage

try:
    from langchain_community.callbacks.manager import get_openai_callback
    HAVE_OPENAI_CB = True
except Exception:
    HAVE_OPENAI_CB = False

# -------------------------------------------------------
# 1. Define structured output schema for crash narratives
# -------------------------------------------------------

class CrashEntity(BaseModel):
    """Entity node such as vehicle, driver, or location"""
    id: str
    label: str
    unit_id: Optional[str] = None
    mention_text: Optional[str] = None
    name: Optional[str] = None
    road: Optional[str] = None
    block: Optional[str] = None
    city: Optional[str] = None
    confidence: Optional[float] = None


class CrashEvent(BaseModel):
    """Event node like collision or violation"""
    id: str
    label: str
    type: str
    attributes: Dict[str, Any]
    confidence: Optional[float] = None
    evidence_span: Optional[str] = None


class CrashRelationship(BaseModel):
    """Edge connecting nodes"""
    start: str
    end: str
    type: str
    properties: Dict[str, Any]


class CrashMetadata(BaseModel):
    """Crash metadata information"""
    crash_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    crash_date: Optional[str] = None
    day_of_week: Optional[str] = None
    crash_time: Optional[str] = None
    county: Optional[str] = None
    city: Optional[str] = None
    sae_autonomy_level: Optional[str] = None
    crash_severity: Optional[str] = None
    raw_narrative: str
    source: str = "police_report"


class CrashGraph(BaseModel):
    """Complete graph structure"""
    crash: CrashMetadata
    entities: List[CrashEntity]
    events: List[CrashEvent]
    relationships: List[CrashRelationship]

# LLMUsage is now imported from llm_providers module


# -------------------------------------------------------
# 2. Create the LLM and structured output wrapper
# -------------------------------------------------------

# Default LLM configuration (can be overridden)
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

def create_structured_llm(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL, api_key: str = None, **kwargs):
    """Create a structured LLM instance with the specified provider"""
    try:
        llm_provider = LLMProviderFactory.create_provider(provider, model, api_key, **kwargs)
        llm = llm_provider.get_llm()
        return llm.with_structured_output(CrashGraph), llm_provider
    except Exception as e:
        raise ValueError(f"Failed to create LLM with provider {provider}: {e}")

# Create default structured LLM (lazy initialization)
structured_llm = None
default_llm_provider = None

def get_default_structured_llm():
    """Get or create the default structured LLM"""
    global structured_llm, default_llm_provider
    if structured_llm is None:
        try:
            structured_llm, default_llm_provider = create_structured_llm()
        except Exception:
            # Fallback to basic ChatOpenAI if provider system fails
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm = llm.with_structured_output(CrashGraph)
            default_llm_provider = None
    return structured_llm, default_llm_provider


# -------------------------------------------------------
# 3. Few-shot examples
# -------------------------------------------------------

TOOL_NAME = "emit_crash_graph"

examples = [
    # ===== Example 1 =====
    HumanMessage(
        content="Crash 19955047: Unit 2 was stationary at a red light on S. Texas Blvd. Unit 1 failed to control speed and struck Unit 2 on the back end.",
        name="example_user",
    ),
    AIMessage(
        content="",
        name="example_assistant",
        tool_calls=[
            {
                "name": TOOL_NAME,
                "args": {
                    "crash": {
                        "crash_id": "19955047",
                        "latitude": 26.15526348,
                        "longitude": -97.99060556,
                        "crash_date": "1/1/24",
                        "day_of_week": "MON",
                        "crash_time": "2:06 AM",
                        "county": "Hidalgo",
                        "city": "Weslaco",
                        "sae_autonomy_level": "2",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 2 was stationary... Unit 1 failed to control speed and struck Unit 2.",
                    },
                    "entities": [
                        {"id": "19955047:U1", "label": "Vehicle", "unit_id": "U1"},
                        {"id": "19955047:U2", "label": "Vehicle", "unit_id": "U2"},
                        {"id": "19955047:L1", "label": "Location", "name": "600 block of S. Texas Blvd", "city": "Weslaco"},
                    ],
                    "events": [
                        {"id": "19955047:E1", "label": "Event", "type": "Violation", "attributes": {"reason": "failed to control speed"}},
                        {"id": "19955047:E2", "label": "Event", "type": "Collision", "attributes": {"impact_config": "rear_end"}},
                    ],
                    "relationships": [
                        {"start": "19955047:U1", "end": "19955047:E1", "type": "PARTICIPATED_IN", "properties": {"role": "Agent"}},
                        {"start": "19955047:U1", "end": "19955047:U2", "type": "HIT", "properties": {"impact_config": "fd-bd"}},
                        {"start": "19955047:E1", "end": "19955047:E2", "type": "CAUSES", "properties": {"confidence": 0.9}},
                    ],
                },
                "id": "tool-1",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-1"),

    # ===== Example 2 =====
    HumanMessage(
        content="Crash 19955369: Unit 1 failed to control speed and collided with Unit 2 on LBJ Fwy.",
        name="example_user",
    ),
    AIMessage(
        content="",
        name="example_assistant",
        tool_calls=[
            {
                "name": TOOL_NAME,
                "args": {
                    "crash": {
                        "crash_id": "19955369",
                        "latitude": 32.92553538,
                        "longitude": -96.80385791,
                        "crash_date": "1/2/24",
                        "day_of_week": "TUE",
                        "crash_time": "3:37 PM",
                        "county": "Dallas",
                        "city": "Dallas",
                        "sae_autonomy_level": "1",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 1 failed to control speed and collided with Unit 2 on LBJ Fwy.",
                    },
                    "entities": [
                        {"id": "19955369:U1", "label": "Vehicle", "unit_id": "U1"},
                        {"id": "19955369:U2", "label": "Vehicle", "unit_id": "U2"},
                        {"id": "19955369:L1", "label": "Location", "name": "6000 block of Lyndon B Johnson Fwy", "city": "Dallas"},
                    ],
                    "events": [
                        {"id": "19955369:E1", "label": "Event", "type": "Violation", "attributes": {"reason": "failed to control speed"}},
                        {"id": "19955369:E2", "label": "Event", "type": "Collision", "attributes": {"impact_config": "rear_end"}},
                    ],
                    "relationships": [
                        {"start": "19955369:U1", "end": "19955369:E1", "type": "PARTICIPATED_IN", "properties": {"role": "Agent"}},
                        {"start": "19955369:E1", "end": "19955369:E2", "type": "CAUSES", "properties": {"marked": True}},
                        {"start": "19955369:U1", "end": "19955369:U2", "type": "HIT", "properties": {"impact_config": "fd-bd"}},
                    ],
                },
                "id": "tool-2",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-2"),

    # ===== Example 3 =====
    HumanMessage(
        content="Crash 19956614: Unit 1 failed to yield from a private drive and collided with Unit 2 on Harris near Pasadena Blvd.",
        name="example_user",
    ),
    AIMessage(
        content="",
        name="example_assistant",
        tool_calls=[
            {
                "name": TOOL_NAME,
                "args": {
                    "crash": {
                        "crash_id": "19956614",
                        "latitude": 29.69813481,
                        "longitude": -95.20085379,
                        "crash_date": "1/3/24",
                        "day_of_week": "WED",
                        "crash_time": "6:49 AM",
                        "county": "Harris",
                        "city": "Pasadena",
                        "sae_autonomy_level": "1",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 1 failed to yield from private drive and collided with Unit 2.",
                    },
                    "entities": [
                        {"id": "19956614:U1", "label": "Vehicle", "unit_id": "U1"},
                        {"id": "19956614:U2", "label": "Vehicle", "unit_id": "U2"},
                        {"id": "19956614:L1", "label": "Location", "name": "900 Harris near Pasadena Blvd", "city": "Pasadena"},
                    ],
                    "events": [
                        {"id": "19956614:E1", "label": "Event", "type": "Violation", "attributes": {"reason": "failed to yield"}},
                        {"id": "19956614:E2", "label": "Event", "type": "Collision", "attributes": {"impact_config": "side_impact"}},
                    ],
                    "relationships": [
                        {"start": "19956614:U1", "end": "19956614:E1", "type": "PARTICIPATED_IN", "properties": {"role": "Agent"}},
                        {"start": "19956614:E1", "end": "19956614:E2", "type": "CAUSES", "properties": {"marked": True}},
                        {"start": "19956614:U1", "end": "19956614:U2", "type": "HIT", "properties": {"impact_config": "fr-rbq"}},
                    ],
                },
                "id": "tool-3",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-3"),
]


# -------------------------------------------------------
# 4. Prompt template for few-shot extraction
# -------------------------------------------------------

system_prompt = (
    "You are an expert transportation crash data analyst. "
    "Given a crash narrative and metadata, extract entities (vehicles, locations), "
    "events (violations, collisions), and causal relationships into a graph. "
    "Return your answer as a structured JSON tool call under the schema 'CrashGraph'."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{examples}"),
    ("human", "{input}")
])

# Lazy initialization for the chain
def get_structured_crash_llm():
    """Get the structured crash LLM chain"""
    llm, _ = get_default_structured_llm()
    return prompt | llm


# -------------------------------------------------------
# 5. Function to invoke anywhere
# -------------------------------------------------------

def analyze_crash(narrative: str, metadata: Dict[str, Any], logger=None, provider: str = None, model: str = None, api_key: str = None):
    """Invoke the structured crash graph LLM on new data"""
    crash_id = metadata.get('Crash_ID', 'unknown')
    crash_text = f"Crash {crash_id}: {narrative}"
    
    # If no provider is specified, try to get it from environment
    if not provider:
        from .config import config
        # Reload config to get latest environment variables
        config.reload()
        llm_config = config.get_llm_config()
        provider = llm_config.provider
        model = model or llm_config.model
        api_key = api_key or llm_config.api_key
    
    if logger:
        logger.log_crash_processing(crash_id, "llm_invocation", f"Calling structured LLM with provider: {provider or DEFAULT_PROVIDER}")
    
    # Use custom provider if specified, otherwise use default
    if provider and provider != DEFAULT_PROVIDER:
        try:
            custom_llm, custom_provider = create_structured_llm(provider, model or DEFAULT_MODEL, api_key)
            
            # Create the prompt with examples and input
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("placeholder", "{examples}"),
                ("human", "{input}")
            ])
            
            # Format the prompt with examples and input
            formatted_prompt = prompt.format(examples=examples, input=crash_text)
            result = custom_llm.invoke(formatted_prompt)
        except Exception as e:
            if logger:
                logger.error(f"Failed to use custom provider {provider}: {e}")
            # Fallback to default
            llm, _ = get_default_structured_llm()
            result = llm.invoke({
                "examples": examples,
                "input": crash_text
            })
    else:
        llm, _ = get_default_structured_llm()
        result = llm.invoke({
            "examples": examples,
            "input": crash_text
        })
    
    if logger:
        logger.info(f"Graph extraction completed for crash {crash_id}")
    
    return result

def analyze_crash_with_usage(narrative_or_formatted: str, metadata: Dict[str, Any], logger=None, provider: str = None, model: str = None, api_key: str = None):
    """
    Wrap the structured extractor with a token and time tracker.
    Returns (CrashGraph, LLMUsage)
    Falls back to zeros if callbacks are not available.
    """
    start = time.time()
    
    # If no provider is specified, try to get it from environment
    if not provider:
        from .config import config
        # Reload config to get latest environment variables
        config.reload()
        llm_config = config.get_llm_config()
        provider = llm_config.provider
        model = model or llm_config.model
        api_key = api_key or llm_config.api_key
    
    # Use custom provider if specified
    if provider and provider != DEFAULT_PROVIDER:
        try:
            if logger:
                logger.info(f"Creating LLM provider: {provider} with model: {model}")
                logger.info(f"API key provided: {'Yes' if api_key else 'No'}")
                logger.info(f"API key value: {api_key[:10] if api_key else 'None'}...")
            
            llm_provider = LLMProviderFactory.create_provider(provider, model or DEFAULT_MODEL, api_key)
            if logger:
                logger.info(f"Created provider: {type(llm_provider).__name__}")
            custom_llm = llm_provider.get_llm().with_structured_output(CrashGraph)
            
            crash_id = metadata.get('Crash_ID', 'unknown')
            crash_text = f"Crash {crash_id}: {narrative_or_formatted}"
            
            if logger:
                logger.info(f"Invoking LLM for crash {crash_id}")
            
            # Create the prompt with examples and input
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("placeholder", "{examples}"),
                ("human", "{input}")
            ])
            
            # Format the prompt with examples and input
            formatted_prompt = prompt.format(examples=examples, input=crash_text)
            result = custom_llm.invoke(formatted_prompt)
            
            end = time.time()
            usage = llm_provider.get_usage_stats(result, start, end)
            return result, usage
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to use custom provider {provider}: {e}")
            else:
                print(f"❌ Failed to use custom provider {provider}: {e}")
            # Don't fallback, raise the error
            raise
    
    # Default method with OpenAI callbacks
    if HAVE_OPENAI_CB and (not provider or provider == DEFAULT_PROVIDER):
        # OpenAI priced usage automatically computed here
        with get_openai_callback() as cb:
            result = analyze_crash(narrative_or_formatted, metadata, logger=logger, provider=provider, model=model, api_key=api_key)
        end = time.time()
        usage = LLMUsage(
            provider="openai",
            model=getattr(getattr(result, "__dict__", {}).get("_lc_kwargs", {}), "get", lambda *_: None)("model", None) if hasattr(result, "__dict__") else None,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            total_tokens=cb.total_tokens,
            total_cost_usd=float(cb.total_cost),
            runtime_sec=end - start,
        )
        return result, usage
    else:
        # If using other providers or no callback, still return a usage stub with runtime
        result = analyze_crash(narrative_or_formatted, metadata, logger=logger, provider=provider, model=model, api_key=api_key)
        end = time.time()
        usage = LLMUsage(
            provider=provider or "unknown",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end - start,
        )
        return result, usage

# -------------------------------------------------------
# 6. Example usage
# -------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "Crash_ID": "19957925",
        "Latitude": 29.66488181,
        "Longitude": -95.13898515,
        "CrashDate": "1/3/24",
        "DayOfWeek": "WED",
        "CrashTime": "2:18 PM",
        "County": "Harris",
        "City": "Pasadena",
        "SAE_Autonomy_Level": "1",
        "Crash_Severity": "Not Injured",
        "Narrative": (
            "Unit 2 was traveling eastbound on Spencer Hwy when "
            "Unit 1's tire detached, rolled ahead, and struck Unit 2. "
            "Unit 2 attempted to avoid but could not due to traffic."
        ),
    }

    graph = analyze_crash(sample["Narrative"], sample)
    print(graph.json(indent=2))


def analyze_crash_with_summary_and_usage(narrative_or_formatted: str, metadata: Dict[str, Any], logger=None, provider: str = None, model: str = None, api_key: str = None) -> Tuple[CrashGraph, str, LLMUsage]:
    """Generate both crash graph and summary in one LLM call"""
    start = time.time()
    
    # Generate the crash graph first
    graph_obj, graph_usage = analyze_crash_with_usage(narrative_or_formatted, metadata, logger, provider, model, api_key)
    
    # Extract the narrative for summary generation
    if isinstance(narrative_or_formatted, str):
        narrative = narrative_or_formatted
    else:
        # Extract narrative from formatted input
        narrative = narrative_or_formatted.get('Narrative', '')
    
    # Generate summary using the same LLM
    try:
        # Create LLM provider for summary generation
        llm_provider = LLMProviderFactory.create_provider(
            provider=provider or "openai",
            model_name=model or "gpt-4o-mini", 
            api_key=api_key
        )
        
        # Create summary prompt
        summary_prompt = f"Summarize this crash report in 1-2 sentences: {narrative}"
        
        # Generate summary
        summary_response = llm_provider.get_llm().invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
        # Create combined usage stats
        end = time.time()
        combined_usage = LLMUsage(
            provider=provider or "unknown",
            model=model or "unknown",
            prompt_tokens=graph_usage.prompt_tokens,
            completion_tokens=graph_usage.completion_tokens,
            total_tokens=graph_usage.total_tokens,
            total_cost_usd=graph_usage.total_cost_usd,
            runtime_sec=end - start
        )
        
        return graph_obj, summary, combined_usage
        
    except Exception as e:
        if logger:
            logger.error(f"Summary generation failed: {e}")
        # Return empty summary if generation fails
        end = time.time()
        combined_usage = LLMUsage(
            provider=provider or "unknown",
            model=model or "unknown",
            prompt_tokens=graph_usage.prompt_tokens,
            completion_tokens=graph_usage.completion_tokens,
            total_tokens=graph_usage.total_tokens,
            total_cost_usd=graph_usage.total_cost_usd,
            runtime_sec=end - start
        )
        return graph_obj, "", combined_usage
