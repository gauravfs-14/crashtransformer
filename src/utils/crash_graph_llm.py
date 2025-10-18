# crash_graph_llm.py

from typing import List, Optional, Dict, Any, Tuple, Literal
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
    label: Literal["VEHICLE", "DRIVER", "ROAD", "LOCATION", "TRAFFIC_CONTROL", "PEDESTRIAN", "CYCLIST", "OTHER"]
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
    type: Literal["VIOLATION", "COLLISION", "VEHICLE_STATE", "VEHICLE_MOVEMENT", "TRAFFIC_VIOLATION", "DISTRACTED_DRIVING", "SPEED_VIOLATION", "LANE_CHANGE", "TURNING", "STOPPING", "OTHER"]
    attributes: Dict[str, Any]
    confidence: Optional[float] = None
    evidence_span: Optional[str] = None


class CrashRelationship(BaseModel):
    """Edge connecting nodes"""
    start: str
    end: str
    type: Literal["PARTICIPATED_IN", "CAUSES", "HIT", "FOLLOWS", "PRECEDES", "LOCATED_AT", "DRIVES", "PASSENGER_IN", "WITNESS_TO", "INVOLVED_IN", "OTHER"]
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
    summary: Optional[str] = None


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
    # ===== Example 1: Rear-End Collision =====
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
                        "crash_date": "2024-01-01",
                        "day_of_week": "MON",
                        "crash_time": "02:06:00",
                        "county": "Hidalgo",
                        "city": "Weslaco",
                        "sae_autonomy_level": "2",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 2 was stationary at a red light on S. Texas Blvd. Unit 1 failed to control speed and struck Unit 2 on the back end.",
                        "source": "police_report"
                    },
                    "entities": [
                        {"id": "19955047:U1", "label": "VEHICLE", "unit_id": "1", "name": "Unit 1"},
                        {"id": "19955047:U2", "label": "VEHICLE", "unit_id": "2", "name": "Unit 2"},
                        {"id": "19955047:L1", "label": "ROAD", "name": "S. Texas Blvd", "city": "Weslaco"}
                    ],
                    "events": [
                        {"id": "19955047:E1", "label": "Stationary at Red Light", "type": "VEHICLE_STATE", "attributes": {}, "evidence_span": "Unit 2 was stationary at a red light"},
                        {"id": "19955047:E2", "label": "Failure to Control Speed", "type": "VIOLATION", "attributes": {}, "evidence_span": "Unit 1 failed to control speed"},
                        {"id": "19955047:E3", "label": "Rear-End Collision", "type": "COLLISION", "attributes": {}, "evidence_span": "struck Unit 2 on the back end"}
                    ],
                    "relationships": [
                        {"start": "19955047:U2", "end": "19955047:E1", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955047:U1", "end": "19955047:E2", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955047:E2", "end": "19955047:E3", "type": "CAUSES", "properties": {}},
                        {"start": "19955047:U1", "end": "19955047:U2", "type": "HIT", "properties": {}}
                    ],
                },
                "id": "tool-1",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-1"),

    # ===== Example 2: Intersection Collision =====
    HumanMessage(
        content="Crash 19955369: Unit 1 was traveling eastbound on Main Street. Unit 2 was traveling northbound on Oak Avenue. Unit 1 failed to yield at the stop sign and collided with Unit 2 at the intersection.",
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
                        "crash_date": "2024-01-02",
                        "day_of_week": "TUE",
                        "crash_time": "15:37:00",
                        "county": "Dallas",
                        "city": "Dallas",
                        "sae_autonomy_level": "1",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 1 was traveling eastbound on Main Street. Unit 2 was traveling northbound on Oak Avenue. Unit 1 failed to yield at the stop sign and collided with Unit 2 at the intersection.",
                        "source": "police_report"
                    },
                    "entities": [
                        {"id": "19955369:U1", "label": "VEHICLE", "unit_id": "1", "name": "Unit 1"},
                        {"id": "19955369:U2", "label": "VEHICLE", "unit_id": "2", "name": "Unit 2"},
                        {"id": "19955369:L1", "label": "ROAD", "name": "Main Street", "city": "Dallas"},
                        {"id": "19955369:L2", "label": "ROAD", "name": "Oak Avenue", "city": "Dallas"}
                    ],
                    "events": [
                        {"id": "19955369:E1", "label": "Eastbound Travel", "type": "VEHICLE_MOVEMENT", "attributes": {}, "evidence_span": "Unit 1 was traveling eastbound on Main Street"},
                        {"id": "19955369:E2", "label": "Northbound Travel", "type": "VEHICLE_MOVEMENT", "attributes": {}, "evidence_span": "Unit 2 was traveling northbound on Oak Avenue"},
                        {"id": "19955369:E3", "label": "Right-of-Way Violation", "type": "VIOLATION", "attributes": {}, "evidence_span": "Unit 1 failed to yield at the stop sign"},
                        {"id": "19955369:E4", "label": "Intersection Collision", "type": "COLLISION", "attributes": {}, "evidence_span": "collided with Unit 2 at the intersection"}
                    ],
                    "relationships": [
                        {"start": "19955369:U1", "end": "19955369:E1", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955369:U2", "end": "19955369:E2", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955369:U1", "end": "19955369:E3", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955369:E3", "end": "19955369:E4", "type": "CAUSES", "properties": {}},
                        {"start": "19955369:U1", "end": "19955369:U2", "type": "HIT", "properties": {}}
                    ],
                },
                "id": "tool-2",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-2"),

    # ===== Example 3: Lane Change Collision =====
    HumanMessage(
        content="Crash 19956614: Unit 1 was traveling southbound on Interstate 35 in the left lane. Unit 2 was traveling southbound on Interstate 35 in the right lane. Unit 1 attempted to change lanes without checking blind spots and collided with Unit 2. The driver admitted to being distracted by a cell phone.",
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
                        "crash_date": "2024-01-03",
                        "day_of_week": "WED",
                        "crash_time": "06:49:00",
                        "county": "Harris",
                        "city": "Houston",
                        "sae_autonomy_level": "1",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 1 was traveling southbound on Interstate 35 in the left lane. Unit 2 was traveling southbound on Interstate 35 in the right lane. Unit 1 attempted to change lanes without checking blind spots and collided with Unit 2. The driver admitted to being distracted by a cell phone.",
                        "source": "police_report"
                    },
                    "entities": [
                        {"id": "19956614:U1", "label": "VEHICLE", "unit_id": "1", "name": "Unit 1"},
                        {"id": "19956614:U2", "label": "VEHICLE", "unit_id": "2", "name": "Unit 2"},
                        {"id": "19956614:L1", "label": "ROAD", "name": "Interstate 35", "city": "Houston"},
                        {"id": "19956614:D1", "label": "DRIVER", "name": "Unit 1 Driver"}
                    ],
                    "events": [
                        {"id": "19956614:E1", "label": "Southbound Travel Left Lane", "type": "VEHICLE_MOVEMENT", "attributes": {}, "evidence_span": "Unit 1 was traveling southbound on Interstate 35 in the left lane"},
                        {"id": "19956614:E2", "label": "Southbound Travel Right Lane", "type": "VEHICLE_MOVEMENT", "attributes": {}, "evidence_span": "Unit 2 was traveling southbound on Interstate 35 in the right lane"},
                        {"id": "19956614:E3", "label": "Distracted Driving", "type": "VIOLATION", "attributes": {}, "evidence_span": "admitted to being distracted by a cell phone"},
                        {"id": "19956614:E4", "label": "Unsafe Lane Change", "type": "VIOLATION", "attributes": {}, "evidence_span": "attempted to change lanes without checking blind spots"},
                        {"id": "19956614:E5", "label": "Lane Change Collision", "type": "COLLISION", "attributes": {}, "evidence_span": "collided with Unit 2"}
                    ],
                    "relationships": [
                        {"start": "19956614:U1", "end": "19956614:E1", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19956614:U2", "end": "19956614:E2", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19956614:D1", "end": "19956614:E3", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19956614:U1", "end": "19956614:E4", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19956614:E3", "end": "19956614:E4", "type": "CAUSES", "properties": {}},
                        {"start": "19956614:E4", "end": "19956614:E5", "type": "CAUSES", "properties": {}},
                        {"start": "19956614:U1", "end": "19956614:U2", "type": "HIT", "properties": {}}
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
    "Extract entities (vehicles, locations), events (violations, collisions), and causal relationships from crash narratives. "
    "Focus on CAUSES relationships between events. "
    "Include evidence_span for each event and confidence scores. "
    "Return structured JSON following the CrashGraph schema."
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
    crash_id = str(metadata.get('Crash_ID', 'unknown'))  # Ensure string type
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
            
            crash_id = str(metadata.get('Crash_ID', 'unknown'))  # Ensure string type
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
            
            # Check if result is None
            if result is None:
                raise ValueError(f"LLM returned None for crash {crash_id}. This might be due to content filtering or API issues.")
            
            # Additional check for structured output
            if not hasattr(result, 'json') and not hasattr(result, 'dict'):
                raise ValueError(f"LLM result is not a valid structured output for crash {crash_id}. Result type: {type(result)}")
            
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


# Create examples with summaries
examples_with_summary = [
    # ===== Example 1: Rear-End Collision with Summary =====
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
                        "crash_date": "2024-01-01",
                        "day_of_week": "MON",
                        "crash_time": "02:06:00",
                        "county": "Hidalgo",
                        "city": "Weslaco",
                        "sae_autonomy_level": "2",
                        "crash_severity": "Not Injured",
                        "raw_narrative": "Unit 2 was stationary at a red light on S. Texas Blvd. Unit 1 failed to control speed and struck Unit 2 on the back end.",
                        "source": "police_report"
                    },
                    "entities": [
                        {"id": "19955047:U1", "label": "VEHICLE", "unit_id": "1", "name": "Unit 1"},
                        {"id": "19955047:U2", "label": "VEHICLE", "unit_id": "2", "name": "Unit 2"},
                        {"id": "19955047:L1", "label": "ROAD", "name": "S. Texas Blvd", "city": "Weslaco"}
                    ],
                    "events": [
                        {"id": "19955047:E1", "label": "Stationary at Red Light", "type": "VEHICLE_STATE", "attributes": {}, "evidence_span": "Unit 2 was stationary at a red light"},
                        {"id": "19955047:E2", "label": "Failure to Control Speed", "type": "VIOLATION", "attributes": {}, "evidence_span": "Unit 1 failed to control speed"},
                        {"id": "19955047:E3", "label": "Rear-End Collision", "type": "COLLISION", "attributes": {}, "evidence_span": "struck Unit 2 on the back end"}
                    ],
                    "relationships": [
                        {"start": "19955047:U2", "end": "19955047:E1", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955047:U1", "end": "19955047:E2", "type": "PARTICIPATED_IN", "properties": {}},
                        {"start": "19955047:E2", "end": "19955047:E3", "type": "CAUSES", "properties": {}},
                        {"start": "19955047:U1", "end": "19955047:U2", "type": "HIT", "properties": {}}
                    ],
                    "summary": "A rear-end collision occurred when Unit 1 failed to control speed and struck Unit 2, which was stationary at a red light on S. Texas Blvd."
                },
                "id": "tool-1",
            }
        ],
    ),
    ToolMessage(content="", tool_call_id="tool-1"),
]


def analyze_crash_with_summary_and_usage(narrative_or_formatted: str, metadata: Dict[str, Any], logger=None, provider: str = None, model: str = None, api_key: str = None) -> Tuple[CrashGraph, str, LLMUsage]:
    """Generate both crash graph and summary in one LLM call"""
    start = time.time()
    
    # Extract the narrative for the combined prompt
    if isinstance(narrative_or_formatted, str):
        narrative = narrative_or_formatted
    else:
        # Extract narrative from formatted input
        narrative = narrative_or_formatted.get('Narrative', '')
    
    crash_id = str(metadata.get('Crash_ID', 'unknown'))  # Ensure string type
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
    
    try:
        # Use the existing structured LLM approach
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
                formatted_prompt = prompt.format(examples=examples_with_summary, input=crash_text)
                result = custom_llm.invoke(formatted_prompt)
            except Exception as e:
                if logger:
                    logger.error(f"Failed to use custom provider {provider}: {e}")
                # Fallback to default
                llm, _ = get_default_structured_llm()
                result = llm.invoke({
                    "examples": examples_with_summary,
                    "input": crash_text
                })
        else:
            llm, _ = get_default_structured_llm()
            result = llm.invoke({
                "examples": examples_with_summary,
                "input": crash_text
            })
        
        if logger:
            logger.log_crash_processing(crash_id, "llm_response", "Received structured LLM response")
        
        # Extract summary from structured output if available, otherwise generate simple one
        if hasattr(result, 'summary') and result.summary:
            summary = result.summary
        else:
            summary = _generate_summary_from_graph(result, narrative)
        
        end = time.time()
        
        # Extract token usage from LLM response if available
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Try to extract token usage from the result
        if hasattr(result, 'usage_metadata'):
            usage_metadata = result.usage_metadata
            prompt_tokens = getattr(usage_metadata, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_metadata, 'candidates_tokens', 0)
            total_tokens = getattr(usage_metadata, 'total_tokens', prompt_tokens + completion_tokens)
            # Calculate cost based on provider
            if provider == "google":
                # Google pricing: $0.000075 per 1K input tokens, $0.0003 per 1K output tokens
                total_cost = (prompt_tokens * 0.000075 + completion_tokens * 0.0003) / 1000
        elif hasattr(result, 'response_metadata'):
            # Alternative metadata structure
            metadata = result.response_metadata
            prompt_tokens = metadata.get('prompt_tokens', 0)
            completion_tokens = metadata.get('completion_tokens', 0)
            total_tokens = metadata.get('total_tokens', prompt_tokens + completion_tokens)
            total_cost = metadata.get('total_cost', 0.0)
        
        usage = LLMUsage(
            provider=provider or DEFAULT_PROVIDER,
            model=model or DEFAULT_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            runtime_sec=end - start
        )
        
        if logger:
            logger.log_llm_call(crash_id, usage.provider, usage.model, usage.total_tokens, usage.total_cost_usd)
        
        return result, summary, usage
        
    except Exception as e:
        if logger:
            logger.error(f"LLM call failed for crash {crash_id}: {str(e)}")
        else:
            print(f"❌ LLM call failed for crash {crash_id}: {str(e)}")
        raise


def _generate_summary_from_graph(graph: CrashGraph, narrative: str) -> str:
    """Generate a simple summary from the crash graph"""
    try:
        # Extract key information from the graph
        vehicles = [e for e in graph.entities if e.label == "Vehicle"]
        events = [e for e in graph.events if e.type in ["Violation", "Collision"]]
        
        if not vehicles or not events:
            return f"Vehicle crash occurred: {narrative[:100]}..."
        
        # Create a simple summary
        vehicle_ids = [v.unit_id for v in vehicles if v.unit_id]
        event_types = [e.type for e in events]
        
        if "Collision" in event_types and "Violation" in event_types:
            return f"Vehicle collision involving {', '.join(vehicle_ids)} with traffic violation and collision."
        elif "Collision" in event_types:
            return f"Vehicle collision involving {', '.join(vehicle_ids)}."
        elif "Violation" in event_types:
            return f"Traffic violation involving {', '.join(vehicle_ids)}."
        else:
            return f"Vehicle incident involving {', '.join(vehicle_ids)}."
            
    except Exception:
        # Fallback to simple narrative truncation
        return f"Vehicle crash: {narrative[:100]}..."
