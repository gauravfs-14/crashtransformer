# llm_providers.py

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import os
from abc import ABC, abstractmethod

# LangChain imports for different providers
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.llms import Ollama
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.callbacks.manager import get_openai_callback
    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False

# XAI Grok support (if available)
try:
    from xai import Grok
    HAVE_GROK = True
except ImportError:
    HAVE_GROK = False

@dataclass
class LLMUsage:
    provider: str
    model: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    runtime_sec: float

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self._llm = None
    
    @abstractmethod
    def get_llm(self):
        """Get the LLM instance"""
        pass
    
    @abstractmethod
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        """Extract usage statistics from response"""
        pass
    
    def invoke(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Invoke the LLM with messages"""
        if self._llm is None:
            self._llm = self.get_llm()
        
        start_time = time.time()
        response = self._llm.invoke(messages, **kwargs)
        end_time = time.time()
        
        return response, self.get_usage_stats(response, start_time, end_time)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""
    
    def get_llm(self):
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0,
            **self.kwargs
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        # Try to get usage from response metadata
        try:
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                usage = response.response_metadata['token_usage']
                return LLMUsage(
                    provider="openai",
                    model=self.model_name,
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    completion_tokens=usage.get('completion_tokens', 0),
                    total_tokens=usage.get('total_tokens', 0),
                    total_cost_usd=0.0,  # Would need pricing info
                    runtime_sec=end_time - start_time
                )
        except:
            pass
        
        # Fallback
        return LLMUsage(
            provider="openai",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end_time - start_time
        )

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def get_llm(self):
        return ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0,
            **self.kwargs
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        try:
            if hasattr(response, 'response_metadata') and 'usage' in response.response_metadata:
                usage = response.response_metadata['usage']
                return LLMUsage(
                    provider="anthropic",
                    model=self.model_name,
                    prompt_tokens=usage.get('input_tokens', 0),
                    completion_tokens=usage.get('output_tokens', 0),
                    total_tokens=usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
                    total_cost_usd=0.0,
                    runtime_sec=end_time - start_time
                )
        except:
            pass
        
        return LLMUsage(
            provider="anthropic",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end_time - start_time
        )

class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""
    
    def get_llm(self):
        # Configure Google provider to use only API key authentication
        # and avoid GCP service authentication that causes ALTS warnings
        import os
        os.environ['GOOGLE_API_KEY'] = self.api_key
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0,
            convert_system_message_to_human=True,  # Avoid system message issues
            max_output_tokens=8192,  # Increase token limit to prevent truncation
            max_tokens=8192,  # Alternative parameter name
            **self.kwargs
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        try:
            # Debug: Print response structure for debugging
            import logging
            logging.info(f"Google Gemini response type: {type(response)}")
            logging.info(f"Google Gemini response attributes: {dir(response)}")
            
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                logging.info(f"Google Gemini response_metadata: {metadata}")
                
                # Check for usage_metadata structure
                if 'usage_metadata' in metadata:
                    usage = metadata['usage_metadata']
                    logging.info(f"Found usage_metadata: {usage}")
                    prompt_tokens = usage.get('prompt_token_count', 0)
                    completion_tokens = usage.get('candidates_token_count', 0)
                    total_tokens = usage.get('total_token_count', 0)
                # Check for direct usage structure
                elif 'usage' in metadata:
                    usage = metadata['usage']
                    logging.info(f"Found usage: {usage}")
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                # Check for token_usage structure (similar to OpenAI)
                elif 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    logging.info(f"Found token_usage: {usage}")
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                else:
                    # Fallback: try to extract from any available usage data
                    logging.info(f"No standard usage structure found, trying direct metadata access")
                    prompt_tokens = metadata.get('prompt_tokens', 0)
                    completion_tokens = metadata.get('completion_tokens', 0)
                    total_tokens = metadata.get('total_tokens', 0)
                
                # Calculate cost based on Google Gemini pricing
                # Gemini 2.5 Flash: $0.000075 per 1K input tokens, $0.0003 per 1K output tokens
                if "flash" in self.model_name.lower():
                    total_cost = (prompt_tokens * 0.000075 + completion_tokens * 0.0003) / 1000
                else:
                    # Gemini Pro pricing: $0.0005 per 1K input tokens, $0.0015 per 1K output tokens
                    total_cost = (prompt_tokens * 0.0005 + completion_tokens * 0.0015) / 1000
                
                logging.info(f"Extracted tokens - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}, cost: {total_cost}")
                
                return LLMUsage(
                    provider="google",
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    runtime_sec=end_time - start_time
                )
            else:
                logging.warning(f"No response_metadata found in Google Gemini response")
        except Exception as e:
            # Log the error for debugging
            import logging
            logging.warning(f"Failed to extract usage stats from Google Gemini response: {e}")
            pass
        
        # Fallback: try to estimate tokens using tiktoken if available
        try:
            import tiktoken
            # Use GPT-4 tokenizer as approximation for Gemini
            encoding = tiktoken.encoding_for_model("gpt-4")
            
            # Try to extract text from response for token estimation
            if hasattr(response, 'content'):
                content = str(response.content)
                estimated_tokens = len(encoding.encode(content))
                
                # Rough estimation: assume 70% input, 30% output
                prompt_tokens = int(estimated_tokens * 0.7)
                completion_tokens = int(estimated_tokens * 0.3)
                total_tokens = estimated_tokens
                
                # Calculate cost
                if "flash" in self.model_name.lower():
                    total_cost = (prompt_tokens * 0.000075 + completion_tokens * 0.0003) / 1000
                else:
                    total_cost = (prompt_tokens * 0.0005 + completion_tokens * 0.0015) / 1000
                
                import logging
                logging.info(f"Using tiktoken estimation - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}, cost: {total_cost}")
                
                return LLMUsage(
                    provider="google",
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    runtime_sec=end_time - start_time
                )
        except ImportError:
            import logging
            logging.warning("tiktoken not available for token estimation")
        except Exception as e:
            import logging
            logging.warning(f"Failed to estimate tokens for Google Gemini: {e}")
        
        # Final fallback - use reasonable estimates based on typical crash narratives
        import logging
        logging.warning("Using fallback token estimates for Google Gemini")
        
        # Typical crash narrative estimates
        estimated_prompt_tokens = 200  # Typical prompt size
        estimated_completion_tokens = 50  # Typical response size
        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
        
        # Calculate cost
        if "flash" in self.model_name.lower():
            total_cost = (estimated_prompt_tokens * 0.000075 + estimated_completion_tokens * 0.0003) / 1000
        else:
            total_cost = (estimated_prompt_tokens * 0.0005 + estimated_completion_tokens * 0.0015) / 1000
        
        return LLMUsage(
            provider="google",
            model=self.model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_total_tokens,
            total_cost_usd=total_cost,
            runtime_sec=end_time - start_time
        )

class OllamaProvider(BaseLLMProvider):
    """Ollama local provider implementation"""
    
    def get_llm(self):
        return Ollama(
            model=self.model_name,
            base_url=self.kwargs.get('base_url', 'http://localhost:11434'),
            **{k: v for k, v in self.kwargs.items() if k != 'base_url'}
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        return LLMUsage(
            provider="ollama",
            model=self.model_name,
            prompt_tokens=0,  # Ollama doesn't provide token counts
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end_time - start_time
        )

class GroqProvider(BaseLLMProvider):
    """Groq provider implementation"""
    
    def get_llm(self):
        return ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=0,
            **self.kwargs
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        try:
            if hasattr(response, 'response_metadata') and 'usage' in response.response_metadata:
                usage = response.response_metadata['usage']
                return LLMUsage(
                    provider="groq",
                    model=self.model_name,
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    completion_tokens=usage.get('completion_tokens', 0),
                    total_tokens=usage.get('total_tokens', 0),
                    total_cost_usd=0.0,
                    runtime_sec=end_time - start_time
                )
        except:
            pass
        
        return LLMUsage(
            provider="groq",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end_time - start_time
        )

class GrokProvider(BaseLLMProvider):
    """XAI Grok provider implementation"""
    
    def get_llm(self):
        if not HAVE_GROK:
            raise ImportError("XAI Grok SDK not available. Install with: pip install xai")
        
        return Grok(
            api_key=self.api_key,
            model=self.model_name,
            **self.kwargs
        )
    
    def get_usage_stats(self, response, start_time: float, end_time: float) -> LLMUsage:
        return LLMUsage(
            provider="grok",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            runtime_sec=end_time - start_time
        )

class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    
    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,  # Alias for google
        "ollama": OllamaProvider,
        "groq": GroqProvider,
        "grok": GrokProvider,
    }
    
    @classmethod
    def create_provider(cls, provider: str, model_name: str, api_key: str = None, **kwargs):
        """Create an LLM provider instance"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(cls.PROVIDERS.keys())}")
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = cls._get_api_key(provider)
        
        provider_class = cls.PROVIDERS[provider]
        return provider_class(model_name=model_name, api_key=api_key, **kwargs)
    
    @staticmethod
    def _get_api_key(provider: str) -> str:
        """Get API key from environment variables"""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "grok": "XAI_API_KEY",
            "ollama": None,  # No API key needed for local Ollama
        }
        
        env_var = key_mapping.get(provider)
        if env_var is None:
            return None
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found. Set {env_var} environment variable.")
        
        return api_key

def get_default_models() -> Dict[str, List[str]]:
    """Get default model names for each provider"""
    return {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        "google": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "ollama": ["llama3", "mistral", "codellama", "phi3"],
        "grok": ["grok-beta", "grok-2"],
    }

def validate_provider_config(provider: str, model_name: str, api_key: str = None) -> bool:
    """Validate provider configuration"""
    try:
        factory = LLMProviderFactory()
        provider_instance = factory.create_provider(provider, model_name, api_key)
        return True
    except Exception as e:
        print(f"Provider validation failed: {e}")
        return False
