#!/usr/bin/env python3
"""
Environment setup script for CrashTransformer
Helps users configure their environment variables and validate setup
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not example_file.exists():
        print("❌ env.example file not found")
        return False
    
    # Copy example to .env
    with open(example_file, 'r') as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    print("📝 Please edit .env file with your actual API keys and configuration")
    return True

def check_required_env_vars():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        return False, "No .env file found"
    
    # Check LLM provider
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        return False, "LLM_PROVIDER not set"
    
    # Check required API keys based on provider
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "grok": "XAI_API_KEY",
        "ollama": None  # No API key needed for local Ollama
    }
    
    required_key = required_keys.get(llm_provider)
    if required_key:
        api_key = os.getenv(required_key)
        if not api_key or api_key == f"your_{required_key.lower()}_here" or api_key.startswith("your_"):
            return False, f"{required_key} not set or using placeholder value"
    
    return True, "All required environment variables are set"

def validate_environment():
    """Validate environment configuration"""
    print("🔍 Validating Environment Configuration")
    print("=" * 50)
    
    is_valid, message = check_required_env_vars()
    
    if not is_valid:
        print(f"❌ {message}")
        print("💡 Run interactive setup to configure environment variables")
        return False
    
    print("✅ All required environment variables are configured")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check LLM provider and API key
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"✅ LLM Provider: {llm_provider}")
    
    # Check API key
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY", 
        "groq": "GROQ_API_KEY",
        "grok": "XAI_API_KEY",
        "ollama": None
    }
    
    required_key = required_keys.get(llm_provider)
    if required_key:
        api_key = os.getenv(required_key)
        if api_key and not api_key.startswith("your_"):
            print(f"✅ {required_key} is configured")
        else:
            print(f"❌ {required_key} not properly configured")
            return False
    else:
        print("✅ Ollama (local) - no API key needed")
    
    # Check optional configurations
    optional_configs = [
        ("NEO4J_URI", "Neo4j database URI"),
        ("NEO4J_USER", "Neo4j username"),
        ("NEO4J_PASSWORD", "Neo4j password"),
        ("LOG_DIR", "Log directory"),
        ("OUTPUT_DIR", "Output directory")
    ]
    
    for env_var, description in optional_configs:
        value = os.getenv(env_var)
        if value and value != f"your_{env_var.lower()}_here" and not value.startswith("your_"):
            print(f"✅ {env_var}: {description}")
        else:
            print(f"⚠️  {env_var}: {description} (using default)")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing Dependencies")
    print("=" * 30)
    
    dependencies = [
        "langchain-openai",
        "langchain-anthropic", 
        "langchain-google-genai",
        "langchain-groq",
        "langchain-community",
        "xai",
        "python-dotenv",
        "neo4j",
        "pandas",
        "transformers",
        "torch"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        os.system(f"pip install {dep}")

def interactive_env_setup():
    """Interactive environment variable setup through terminal"""
    print("🔧 Interactive Environment Setup")
    print("=" * 50)
    print("This will guide you through setting up your environment variables.")
    print("Press Enter to skip optional variables or use defaults.\n")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        create_env_file()
    
    # Load existing .env content
    env_content = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_content[key] = value
    
    # LLM Provider selection
    print("🤖 LLM Provider Configuration")
    print("-" * 30)
    
    providers = [
        ("openai", "OpenAI (GPT models)"),
        ("anthropic", "Anthropic Claude"),
        ("google", "Google Gemini"),
        ("groq", "Groq (Fast inference)"),
        ("ollama", "Ollama (Local)"),
        ("grok", "XAI Grok")
    ]
    
    print("Available LLM providers:")
    for i, (provider, description) in enumerate(providers, 1):
        print(f"  {i}. {provider} - {description}")
    
    while True:
        try:
            choice = input(f"\nSelect LLM provider (1-{len(providers)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                selected_provider = providers[int(choice) - 1][0]
                env_content["LLM_PROVIDER"] = selected_provider
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nSetup cancelled.")
            return False
    
    print(f"\n✅ Selected provider: {selected_provider}")
    
    # API Key configuration based on provider
    api_key_vars = {
        "openai": ("OPENAI_API_KEY", "OpenAI API key (sk-...)"),
        "anthropic": ("ANTHROPIC_API_KEY", "Anthropic API key (sk-ant-...)"),
        "google": ("GOOGLE_API_KEY", "Google API key (AI...)"),
        "groq": ("GROQ_API_KEY", "Groq API key (gsk_...)"),
        "grok": ("XAI_API_KEY", "XAI API key (xai-...)"),
        "ollama": (None, "No API key needed for local Ollama")
    }
    
    api_key_var, api_key_desc = api_key_vars[selected_provider]
    
    if api_key_var:
        print(f"\n🔑 {api_key_desc}")
        print("Enter your API key (input will be hidden):")
        
        import getpass
        api_key = getpass.getpass("API Key: ").strip()
        
        if api_key:
            env_content[api_key_var] = api_key
            print("✅ API key saved")
        else:
            print("⚠️  No API key provided. You'll need to set this later.")
    
    # Model selection
    model_vars = {
        "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": ("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        "google": ("GOOGLE_MODEL", "gemini-2.0-flash"),
        "groq": ("GROQ_MODEL", "llama-3.1-70b-versatile"),
        "grok": ("XAI_MODEL", "grok-beta"),
        "ollama": ("OLLAMA_MODEL", "llama3")
    }
    
    model_var, default_model = model_vars[selected_provider]
    
    print(f"\n🤖 Model Configuration")
    print(f"Default model: {default_model}")
    custom_model = input(f"Enter custom model name (or press Enter for default): ").strip()
    
    if custom_model:
        env_content[model_var] = custom_model
        print(f"✅ Model set to: {custom_model}")
    else:
        env_content[model_var] = default_model
        print(f"✅ Using default model: {default_model}")
    
    # Database configuration
    print(f"\n🗄️  Database Configuration")
    print("-" * 30)
    
    enable_neo4j = input("Enable Neo4j database storage? (y/N): ").strip().lower()
    if enable_neo4j in ['y', 'yes']:
        env_content["ENABLE_NEO4J"] = "true"
        
        neo4j_uri = input("Neo4j URI (default: bolt://localhost:7687): ").strip()
        if neo4j_uri:
            env_content["NEO4J_URI"] = neo4j_uri
        else:
            env_content["NEO4J_URI"] = "bolt://localhost:7687"
        
        neo4j_user = input("Neo4j username (default: neo4j): ").strip()
        if neo4j_user:
            env_content["NEO4J_USER"] = neo4j_user
        else:
            env_content["NEO4J_USER"] = "neo4j"
        
        neo4j_password = getpass.getpass("Neo4j password: ").strip()
        if neo4j_password:
            env_content["NEO4J_PASSWORD"] = neo4j_password
        else:
            env_content["NEO4J_PASSWORD"] = "password"
        
        print("✅ Neo4j configuration saved")
    else:
        env_content["ENABLE_NEO4J"] = "false"
        print("✅ Neo4j disabled")
    
    # Logging configuration
    print(f"\n📝 Logging Configuration")
    print("-" * 30)
    
    log_dir = input("Log directory (default: logs): ").strip()
    if log_dir:
        env_content["LOG_DIR"] = log_dir
    else:
        env_content["LOG_DIR"] = "logs"
    
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    print("Available log levels:")
    for i, level in enumerate(log_levels, 1):
        print(f"  {i}. {level}")
    
    while True:
        try:
            level_choice = input(f"Select log level (1-{len(log_levels)}, default: INFO): ").strip()
            if not level_choice:
                env_content["LOG_LEVEL"] = "INFO"
                break
            elif level_choice.isdigit() and 1 <= int(level_choice) <= len(log_levels):
                env_content["LOG_LEVEL"] = log_levels[int(level_choice) - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            env_content["LOG_LEVEL"] = "INFO"
            break
    
    print(f"✅ Log level set to: {env_content['LOG_LEVEL']}")
    
    # Output directory
    print(f"\n📁 Output Configuration")
    print("-" * 30)
    
    output_dir = input("Output directory (default: artifacts): ").strip()
    if output_dir:
        env_content["OUTPUT_DIR"] = output_dir
    else:
        env_content["OUTPUT_DIR"] = "artifacts"
    
    print(f"✅ Output directory set to: {env_content['OUTPUT_DIR']}")
    
    # Cost tracking configuration
    print(f"\n💰 Cost Tracking Configuration")
    print("-" * 30)
    
    enable_cost_tracking = input("Enable cost tracking? (Y/n): ").strip().lower()
    if enable_cost_tracking in ['n', 'no']:
        env_content["ENABLE_COST_TRACKING"] = "false"
        print("✅ Cost tracking disabled")
    else:
        env_content["ENABLE_COST_TRACKING"] = "true"
        
        # Provider-specific pricing configuration
        if selected_provider in ["anthropic", "google", "groq"]:
            print(f"\n💵 {selected_provider.title()} Pricing Configuration")
            print("Enter pricing in USD per 1,000,000 tokens (e.g., 0.1 for $0.10 per 1M tokens)")
            print("Press Enter to skip and configure later")
            
            # Show pricing examples
            show_examples = input("Show pricing examples? (Y/n): ").strip().lower()
            if show_examples not in ['n', 'no']:
                print_pricing_examples()
            
            # Input token pricing
            input_price = input(f"{selected_provider.title()} input token price per 1M (USD): ").strip()
            if input_price:
                try:
                    input_price_float = float(input_price)
                    env_content[f"{selected_provider.upper()}_INPUT_PRICE"] = str(input_price_float)
                    print(f"✅ Input price set to ${input_price_float} per 1M tokens")
                except ValueError:
                    print("⚠️  Invalid price format, skipping input pricing")
            
            # Output token pricing
            output_price = input(f"{selected_provider.title()} output token price per 1M (USD): ").strip()
            if output_price:
                try:
                    output_price_float = float(output_price)
                    env_content[f"{selected_provider.upper()}_OUTPUT_PRICE"] = str(output_price_float)
                    print(f"✅ Output price set to ${output_price_float} per 1M tokens")
                except ValueError:
                    print("⚠️  Invalid price format, skipping output pricing")
        
        elif selected_provider == "ollama":
            print(f"\n🖥️  Local GPU Cost Configuration")
            gpu_rate = input("GPU hourly rate in USD (default: 1.50): ").strip()
            if gpu_rate:
                try:
                    gpu_rate_float = float(gpu_rate)
                    env_content["GPU_HOURLY_RATE"] = str(gpu_rate_float)
                    print(f"✅ GPU rate set to ${gpu_rate_float} per hour")
                except ValueError:
                    print("⚠️  Invalid rate format, using default $1.50/hour")
                    env_content["GPU_HOURLY_RATE"] = "1.50"
            else:
                env_content["GPU_HOURLY_RATE"] = "1.50"
                print("✅ Using default GPU rate: $1.50/hour")
        
        elif selected_provider == "openai":
            print("✅ OpenAI cost tracking enabled automatically")
            print("💡 OpenAI pricing is calculated automatically via LangChain callbacks")
        
        elif selected_provider == "grok":
            print("⚠️  XAI Grok cost tracking is limited to runtime tracking only")
            print("💡 Token-based cost calculation not available for Grok")
    
    # Save configuration
    print(f"\n💾 Saving Configuration")
    print("-" * 30)
    
    # Write to .env file
    with open(env_file, 'w') as f:
        f.write("# CrashTransformer Environment Configuration\n")
        f.write("# Generated by interactive setup\n\n")
        
        # Write LLM configuration
        f.write("# LLM Provider Configuration\n")
        f.write(f"LLM_PROVIDER={env_content.get('LLM_PROVIDER', 'openai')}\n")
        if api_key_var and api_key_var in env_content:
            f.write(f"{api_key_var}={env_content[api_key_var]}\n")
        if model_var in env_content:
            f.write(f"{model_var}={env_content[model_var]}\n")
        f.write("\n")
        
        # Write database configuration
        f.write("# Database Configuration\n")
        f.write(f"NEO4J_URI={env_content.get('NEO4J_URI', 'bolt://localhost:7687')}\n")
        f.write(f"NEO4J_USER={env_content.get('NEO4J_USER', 'neo4j')}\n")
        f.write(f"NEO4J_PASSWORD={env_content.get('NEO4J_PASSWORD', 'password')}\n")
        f.write(f"ENABLE_NEO4J={env_content.get('ENABLE_NEO4J', 'false')}\n")
        f.write("\n")
        
        # Write logging configuration
        f.write("# Logging Configuration\n")
        f.write(f"LOG_DIR={env_content.get('LOG_DIR', 'logs')}\n")
        f.write(f"LOG_LEVEL={env_content.get('LOG_LEVEL', 'INFO')}\n")
        f.write("ENABLE_LOGGING=true\n")
        f.write("\n")
        
        # Write output configuration
        f.write("# Output Configuration\n")
        f.write(f"OUTPUT_DIR={env_content.get('OUTPUT_DIR', 'artifacts')}\n")
        f.write("\n")
        
        # Write cost tracking configuration
        f.write("# Cost Tracking Configuration\n")
        f.write(f"ENABLE_COST_TRACKING={env_content.get('ENABLE_COST_TRACKING', 'true')}\n")
        
        # Write provider-specific pricing
        if selected_provider in ["anthropic", "google", "groq"]:
            if f"{selected_provider.upper()}_INPUT_PRICE" in env_content:
                f.write(f"{selected_provider.upper()}_INPUT_PRICE={env_content[f'{selected_provider.upper()}_INPUT_PRICE']}\n")
            if f"{selected_provider.upper()}_OUTPUT_PRICE" in env_content:
                f.write(f"{selected_provider.upper()}_OUTPUT_PRICE={env_content[f'{selected_provider.upper()}_OUTPUT_PRICE']}\n")
        
        if selected_provider == "ollama" and "GPU_HOURLY_RATE" in env_content:
            f.write(f"GPU_HOURLY_RATE={env_content['GPU_HOURLY_RATE']}\n")
        
        f.write("\n")
        
        # Write other defaults
        f.write("# Other Configuration\n")
        f.write("COST_MODE=local\n")
        f.write("DEFAULT_SUMMARIZATION_MODEL=facebook/bart-base\n")
    
    print("✅ Configuration saved to .env file")
    
    # Validate the configuration
    print(f"\n🔍 Validating Configuration")
    print("-" * 30)
    
    # Load the new configuration
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if required API key is set
    if api_key_var and api_key_var in env_content:
        print(f"✅ {api_key_var} is configured")
    else:
        print(f"⚠️  {api_key_var} not set - you'll need to add this manually")
    
    print(f"✅ LLM Provider: {env_content.get('LLM_PROVIDER', 'openai')}")
    print(f"✅ Model: {env_content.get(model_var, default_model)}")
    print(f"✅ Neo4j: {'Enabled' if env_content.get('ENABLE_NEO4J') == 'true' else 'Disabled'}")
    print(f"✅ Log Level: {env_content.get('LOG_LEVEL', 'INFO')}")
    print(f"✅ Output Directory: {env_content.get('OUTPUT_DIR', 'artifacts')}")
    print(f"✅ Cost Tracking: {'Enabled' if env_content.get('ENABLE_COST_TRACKING', 'true') == 'true' else 'Disabled'}")
    
    # Show pricing configuration if enabled
    if env_content.get('ENABLE_COST_TRACKING', 'true') == 'true':
        if selected_provider in ["anthropic", "google", "groq"]:
            input_price = env_content.get(f"{selected_provider.upper()}_INPUT_PRICE")
            output_price = env_content.get(f"{selected_provider.upper()}_OUTPUT_PRICE")
            if input_price:
                print(f"✅ {selected_provider.title()} Input Price: ${input_price} per 1M tokens")
            if output_price:
                print(f"✅ {selected_provider.title()} Output Price: ${output_price} per 1M tokens")
        elif selected_provider == "ollama":
            gpu_rate = env_content.get('GPU_HOURLY_RATE', '1.50')
            print(f"✅ GPU Hourly Rate: ${gpu_rate}")
        elif selected_provider == "openai":
            print("✅ OpenAI: Automatic cost calculation enabled")
        elif selected_provider == "grok":
            print("⚠️  Grok: Runtime tracking only")
    
    print(f"\n🎉 Setup Complete!")
    print("You can now run: python main_pipeline.py --csv crashes.csv")
    
    return True

def print_usage_examples():
    """Print usage examples"""
    print("\n🚀 Usage Examples")
    print("=" * 30)
    
    examples = [
        {
            "title": "Basic usage with OpenAI",
            "command": "python main_pipeline.py --csv crashes.csv"
        },
        {
            "title": "Use Anthropic Claude",
            "command": "python main_pipeline.py --csv crashes.csv --llm_provider anthropic"
        },
        {
            "title": "Use Groq for fast processing",
            "command": "python main_pipeline.py --csv crashes.csv --llm_provider groq"
        },
        {
            "title": "Enable Neo4j storage",
            "command": "python main_pipeline.py --csv crashes.csv --neo4j_enabled"
        },
        {
            "title": "Debug mode with custom log level",
            "command": "python main_pipeline.py --csv crashes.csv --log_level DEBUG"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['command']}")

def print_pricing_examples():
    """Print pricing examples for different providers"""
    print("\n💡 Pricing Examples (as of 2024):")
    print("=" * 50)
    
    examples = [
        {
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "input": "$0.15",
            "output": "$0.60",
            "note": "Automatic calculation"
        },
        {
            "provider": "Anthropic",
            "model": "claude-3-haiku-20240307",
            "input": "$0.25",
            "output": "$1.25",
            "note": "Manual configuration"
        },
        {
            "provider": "Google",
            "model": "gemini-2.0-flash",
            "input": "$0.10",
            "output": "$0.40",
            "note": "Manual configuration"
        },
        {
            "provider": "Groq",
            "model": "llama-3.1-70b-versatile",
            "input": "$0.10",
            "output": "$0.10",
            "note": "Manual configuration"
        },
        {
            "provider": "Ollama",
            "model": "llama3 (local)",
            "input": "N/A",
            "output": "N/A",
            "note": "GPU hourly rate: $1.50"
        }
    ]
    
    for example in examples:
        print(f"\n{example['provider']} ({example['model']}):")
        if example['provider'] == 'Ollama':
            print(f"  GPU Rate: {example['input']}")
        else:
            print(f"  Input: {example['input']} per 1M tokens")
            print(f"  Output: {example['output']} per 1M tokens")
        print(f"  Note: {example['note']}")

def main():
    """Main setup function"""
    print("🔧 CrashTransformer Environment Setup")
    print("=" * 50)
    
    # First, check if environment is already properly configured
    is_valid, message = check_required_env_vars()
    
    if not is_valid:
        print(f"❌ Environment not configured: {message}")
        print("\n🔧 Let's set up your environment!")
        print("Choose an option:")
        print("1. Interactive environment setup (recommended)")
        print("2. Create .env file from template")
        print("3. Validate environment configuration")
        print("4. Install dependencies")
        print("5. Show usage examples")
        print("6. Full setup (all of the above)")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            interactive_env_setup()
        elif choice == "2":
            create_env_file()
        elif choice == "3":
            validate_environment()
        elif choice == "4":
            install_dependencies()
        elif choice == "5":
            print_usage_examples()
        elif choice == "6":
            interactive_env_setup()
            install_dependencies()
            validate_environment()
            print_usage_examples()
        else:
            print("Invalid choice")
    else:
        print("✅ Environment is properly configured!")
        print(f"✅ {message}")
        
        # Show current configuration
        from dotenv import load_dotenv
        load_dotenv()
        
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        print(f"✅ LLM Provider: {llm_provider}")
        
        # Show model
        model_vars = {
            "openai": "OPENAI_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "google": "GOOGLE_MODEL",
            "groq": "GROQ_MODEL",
            "grok": "XAI_MODEL",
            "ollama": "OLLAMA_MODEL"
        }
        
        model_var = model_vars.get(llm_provider)
        if model_var:
            model = os.getenv(model_var, "default")
            print(f"✅ Model: {model}")
        
        # Show Neo4j status
        neo4j_enabled = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
        print(f"✅ Neo4j: {'Enabled' if neo4j_enabled else 'Disabled'}")
        
        print("\n🎉 You're ready to run CrashTransformer!")
        print("Try: python crashtransformer.py run --csv data/test_data_5rows.csv")
        
        # Ask if user wants to reconfigure
        reconfigure = input("\nWould you like to reconfigure your environment? (y/N): ").strip().lower()
        if reconfigure in ['y', 'yes']:
            interactive_env_setup()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-env":
            create_env_file()
        elif sys.argv[1] == "--validate":
            validate_environment()
        elif sys.argv[1] == "--install":
            install_dependencies()
        elif sys.argv[1] == "--examples":
            print_usage_examples()
        elif sys.argv[1] == "--interactive":
            interactive_env_setup()
        else:
            print("Usage: python setup_env.py [--create-env|--validate|--install|--examples|--interactive]")

if __name__ == "__main__":
    main()
