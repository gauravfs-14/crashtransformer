#!/usr/bin/env python3
"""
CrashTransformer - AI-Powered Crash Analysis Pipeline
Main entry point for environment setup and pipeline execution
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_environment():
    """Run environment setup"""
    try:
        from src.setup_env import main as setup_main
        print("🔧 Running Environment Setup...")
        setup_main()
    except ImportError as e:
        print(f"❌ Error importing setup module: {e}")
        print("💡 Make sure you're in the correct directory with src/ folder")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    return True

def run_pipeline():
    """Run the main pipeline"""
    try:
        from src.main_pipeline import main as pipeline_main
        print("🚀 Running CrashTransformer Pipeline...")
        pipeline_main()
    except ImportError as e:
        print(f"❌ Error importing pipeline module: {e}")
        print("💡 Make sure you're in the correct directory with src/ folder")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    return True

def run_training():
    """Run model training"""
    try:
        from src.train_model import main as training_main
        print("🎯 Running Model Training...")
        training_main()
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        print("💡 Make sure you're in the correct directory with src/ folder")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    return True

def run_data_preparation():
    """Run training data preparation"""
    try:
        from src.prepare_training_data import main as data_prep_main
        print("📊 Preparing Training Data...")
        data_prep_main()
    except ImportError as e:
        print(f"❌ Error importing data preparation module: {e}")
        print("💡 Make sure you're in the correct directory with src/ folder")
        return False
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    return True

def show_help():
    """Show help information"""
    print("""
🔧 CrashTransformer - AI-Powered Crash Analysis Pipeline
=======================================================

MAIN COMMANDS:
  setup          Interactive environment configuration
  run            Execute the crash analysis pipeline  
  train         Fine-tune models for crash summarization
  prepare-data   Prepare training data from pipeline outputs
  docs           Open local HTML documentation
  help           Show this help message

QUICK START:
  1. python crashtransformer.py setup
  2. python crashtransformer.py run --csv crashes.csv

EXAMPLES:
  # Basic usage
  python crashtransformer.py run --csv crashes.csv
  
  # Use Anthropic Claude
  python crashtransformer.py run --csv crashes.csv --llm_provider anthropic
  
  # Enable Neo4j storage
  python crashtransformer.py run --csv crashes.csv --neo4j_enabled

For detailed documentation, run: python crashtransformer.py docs
""")

def run_docs():
    """Simple docs serving - just open the docs_html directory if it exists"""
    docs_html_dir = os.path.join(os.path.dirname(__file__), 'docs_html')
    
    if not os.path.isdir(docs_html_dir):
        print("❌ Docs HTML directory not found. Run 'make docs' first to build documentation.")
        return False
    
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(docs_html_dir)}/index.html")
        print(f"📚 Opening docs at: {docs_html_dir}/index.html")
        return True
    except Exception as e:
        print(f"❌ Failed to open docs: {e}")
        return False

def main():
    """Main entry point for CrashTransformer"""
    
    if len(sys.argv) < 2:
        print("🔧 CrashTransformer - AI-Powered Crash Analysis Pipeline")
        print("=" * 60)
        print("Choose an option:")
        print("1. Setup environment (recommended for first time)")
        print("2. Run pipeline")
        print("3. Train model")
        print("4. Prepare training data")
        print("5. Show help")
        print("6. Docs (open local documentation)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            setup_environment()
        elif choice == "2":
            print("💡 First run: python crashtransformer.py setup")
            print("Then: python crashtransformer.py run --csv crashes.csv")
        elif choice == "3":
            print("💡 Training requires prepared data first")
            print("Run: python crashtransformer.py prepare-data --help")
        elif choice == "4":
            print("💡 Prepare training data from pipeline outputs")
            print("Run: python crashtransformer.py prepare-data --help")
        elif choice == "5":
            show_help()
        elif choice == "6":
            run_docs()
        elif choice == "7":
            print("👋 Goodbye!")
            return
        else:
            print("Invalid choice")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        # Pass remaining arguments to setup
        sys.argv = ["setup_env.py"] + sys.argv[2:]
        setup_environment()
        
    elif command == "run":
        # Pass remaining arguments to pipeline
        sys.argv = ["main_pipeline.py"] + sys.argv[2:]
        run_pipeline()
        
    elif command == "validate":
        # Validate configuration
        sys.argv = ["setup_env.py", "--validate"]
        setup_environment()
        
    elif command == "install":
        # Install dependencies
        sys.argv = ["setup_env.py", "--install"]
        setup_environment()
        
    elif command == "examples":
        # Show examples
        sys.argv = ["setup_env.py", "--examples"]
        setup_environment()
        
    elif command == "train":
        # Run model training
        sys.argv = ["train_model.py"] + sys.argv[2:]
        run_training()
        
    elif command == "prepare-data":
        # Prepare training data
        sys.argv = ["prepare_training_data.py"] + sys.argv[2:]
        run_data_preparation()
        
    elif command == "docs":
        # Build and serve HTML docs from docs/*.md
        run_docs()
        
    elif command in ["help", "--help", "-h"]:
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Run 'python crashtransformer.py help' for usage information")

if __name__ == "__main__":
    main()
