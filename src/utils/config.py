# config.py

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv(override=True)

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None

@dataclass
class DatabaseConfig:
    """Database configuration"""
    uri: str
    user: str
    password: str
    enabled: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    enabled: bool = True
    level: str = "INFO"
    directory: str = "logs"

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    cost_mode: str = "local"
    gpu_hourly_rate: float = 1.50
    batch_size: int = 100
    max_workers: int = 4

class ConfigManager:
    """Manages configuration from environment variables with CLI overrides"""
    
    def __init__(self):
        self._load_from_env()
    
    def reload(self):
        """Reload configuration from environment variables"""
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # LLM Configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = self._get_llm_model()
        self.llm_api_key = self._get_llm_api_key()
        
        # Database Configuration
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_enabled = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
        
        # Logging Configuration
        self.logging_enabled = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_dir = os.getenv("LOG_DIR", "logs")
        
        # Processing Configuration
        self.cost_mode = os.getenv("COST_MODE", "local")
        self.gpu_hourly_rate = float(os.getenv("GPU_HOURLY_RATE", "1.50"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        
        # Output Configuration
        self.output_dir = os.getenv("OUTPUT_DIR", "artifacts")
        self.default_summarization_model = os.getenv("DEFAULT_SUMMARIZATION_MODEL", "facebook/bart-base")
        
        # Cost Tracking
        self.cost_tracking_enabled = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
    
    def _get_llm_model(self) -> str:
        """Get LLM model based on provider"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        
        model_mapping = {
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "google": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            "gemini": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            "groq": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            "grok": os.getenv("XAI_MODEL", "grok-beta"),
            "ollama": os.getenv("OLLAMA_MODEL", "llama3"),
        }
        
        return model_mapping.get(provider, "gpt-4o-mini")
    
    def _get_llm_api_key(self) -> Optional[str]:
        """Get API key based on provider"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "grok": "XAI_API_KEY",
            "ollama": None,  # No API key needed
        }
        
        env_var = key_mapping.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def get_llm_config(self, provider: str = None, model: str = None, api_key: str = None) -> LLMConfig:
        """Get LLM configuration with optional overrides"""
        return LLMConfig(
            provider=provider or self.llm_provider,
            model=model or self.llm_model,
            api_key=api_key or self.llm_api_key,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") if (provider or self.llm_provider) == "ollama" else None,
            input_price=self._get_pricing(provider or self.llm_provider, "input"),
            output_price=self._get_pricing(provider or self.llm_provider, "output")
        )
    
    def _get_pricing(self, provider: str, price_type: str) -> Optional[float]:
        """Get pricing information for provider"""
        if price_type == "input":
            price_mapping = {
                "openai": "OPENAI_INPUT_PRICE",
                "anthropic": "ANTHROPIC_INPUT_PRICE", 
                "google": "GOOGLE_INPUT_PRICE",
                "groq": "GROQ_INPUT_PRICE"
            }
        else:  # output
            price_mapping = {
                "openai": "OPENAI_OUTPUT_PRICE",
                "anthropic": "ANTHROPIC_OUTPUT_PRICE",
                "google": "GOOGLE_OUTPUT_PRICE", 
                "groq": "GROQ_OUTPUT_PRICE"
            }
        
        env_var = price_mapping.get(provider)
        if env_var:
            price = os.getenv(env_var)
            return float(price) if price else None
        return None
    
    def get_database_config(self, uri: str = None, user: str = None, password: str = None, enabled: bool = None) -> DatabaseConfig:
        """Get database configuration with optional overrides"""
        return DatabaseConfig(
            uri=uri or self.neo4j_uri,
            user=user or self.neo4j_user,
            password=password or self.neo4j_password,
            enabled=enabled if enabled is not None else self.neo4j_enabled
        )
    
    def get_logging_config(self, enabled: bool = None, level: str = None, directory: str = None) -> LoggingConfig:
        """Get logging configuration with optional overrides"""
        return LoggingConfig(
            enabled=enabled if enabled is not None else self.logging_enabled,
            level=level or self.log_level,
            directory=directory or self.log_dir
        )
    
    def get_processing_config(self, cost_mode: str = None, gpu_rate: float = None) -> ProcessingConfig:
        """Get processing configuration with optional overrides"""
        return ProcessingConfig(
            cost_mode=cost_mode or self.cost_mode,
            gpu_hourly_rate=gpu_rate or self.gpu_hourly_rate,
            batch_size=self.batch_size,
            max_workers=self.max_workers
        )
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check LLM configuration
        if self.llm_provider != "ollama" and not self.llm_api_key:
            status["errors"].append(f"API key required for {self.llm_provider} provider")
            status["valid"] = False
        
        # Check database configuration if enabled
        if self.neo4j_enabled:
            if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
                status["warnings"].append("Neo4j enabled but configuration incomplete")
        
        # Check logging directory
        if self.logging_enabled and not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception as e:
                status["errors"].append(f"Cannot create log directory: {e}")
                status["valid"] = False
        
        return status
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("🔧 CrashTransformer Configuration Summary")
        print("=" * 50)
        print(f"LLM Provider: {self.llm_provider}")
        print(f"LLM Model: {self.llm_model}")
        print(f"API Key: {'✅ Set' if self.llm_api_key else '❌ Missing'}")
        print(f"Neo4j: {'✅ Enabled' if self.neo4j_enabled else '❌ Disabled'}")
        print(f"Logging: {'✅ Enabled' if self.logging_enabled else '❌ Disabled'}")
        print(f"Cost Mode: {self.cost_mode}")
        print(f"Output Directory: {self.output_dir}")

# Global config instance
config = ConfigManager()
