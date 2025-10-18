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

def clean_database():
    """Clean Neo4j database"""
    try:
        from src.utils.config import config
        from src.utils.neo4j_io import Neo4jSink
        
        # Reload config to get latest environment variables
        config.reload()
        
        if not config.neo4j_enabled:
            print("⚠️ Neo4j is not enabled, skipping database cleanup")
            return True
            
        print("🗄️ Connecting to Neo4j database...")
        sink = Neo4jSink(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password
        )
        
        print("🧹 Clearing all data from Neo4j database...")
        with sink._driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ All nodes and relationships deleted")
            
        sink.close()
        print("✅ Neo4j database cleaned successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing Neo4j module: {e}")
        return False
    except Exception as e:
        print(f"❌ Database cleanup failed: {e}")
        return False

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
  clean-db       Clear Neo4j database (removes all data)
  docs           Convert docs to HTML and serve in browser
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
    """Convert docs folder to HTML and serve in browser"""
    import markdown
    import webbrowser
    from pathlib import Path
    
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    docs_html_dir = os.path.join(os.path.dirname(__file__), 'docs_html')
    
    # Create docs_html directory if it doesn't exist
    os.makedirs(docs_html_dir, exist_ok=True)
    
    print("📚 Converting documentation to HTML...")
    
    # Configure markdown with extensions
    md = markdown.Markdown(
        extensions=[
            'markdown.extensions.toc',
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'pymdownx.superfences',
            'pymdownx.tabbed'
        ],
        extension_configs={
            'markdown.extensions.toc': {
                'permalink': True,
                'permalink_title': 'Link to this section'
            }
        }
    )
    
    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - CrashTransformer Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        code {{
            background-color: #f1f2f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding-left: 20px;
            color: #7f8c8d;
        }}
        .nav {{
            background: #34495e;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .nav a {{
            color: #ecf0f1;
            text-decoration: none;
            margin-right: 20px;
        }}
        .nav a:hover {{
            color: #3498db;
        }}
        .toc {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }}
        .toc ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .toc a {{
            color: #2c3e50;
            text-decoration: none;
        }}
        .toc a:hover {{
            color: #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="index.html">🏠 Home</a>
            <a href="00_START_HERE.html">🚀 Start Here</a>
            <a href="INTRODUCTION.html">📖 Introduction</a>
            <a href="USAGE_GUIDE.html">📋 Usage Guide</a>
            <a href="CLI_REFERENCE.html">🧰 CLI Reference</a>
            <a href="PROVIDERS_GUIDE.html">🤖 Providers</a>
            <a href="NEO4J_GUIDE.html">🗄️ Neo4j</a>
            <a href="OUTPUTS.html">📤 Outputs</a>
            <a href="COST_PERFORMANCE.html">💰 Cost & Performance</a>
            <a href="FINE_TUNING_GUIDE.html">🎯 Fine-tuning</a>
            <a href="TROUBLESHOOTING_FAQ.html">🛠️ Troubleshooting</a>
        </div>
        {content}
    </div>
</body>
</html>"""
    
    # Get all markdown files in docs directory
    docs_path = Path(docs_dir)
    md_files = list(docs_path.glob('*.md'))
    
    if not md_files:
        print("❌ No markdown files found in docs directory")
        return False
    
    # Convert each markdown file to HTML
    converted_files = []
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML
            html_content = md.convert(md_content)
            
            # Create HTML file
            html_file = os.path.join(docs_html_dir, f"{md_file.stem}.html")
            title = md_file.stem.replace('_', ' ').replace('-', ' ').title()
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_template.format(title=title, content=html_content))
            
            converted_files.append(html_file)
            print(f"✅ Converted: {md_file.name} → {os.path.basename(html_file)}")
            
        except Exception as e:
            print(f"❌ Failed to convert {md_file.name}: {e}")
            continue
    
    if not converted_files:
        print("❌ No files were successfully converted")
        return False
    
    # Create index.html if it doesn't exist
    index_file = os.path.join(docs_html_dir, 'index.html')
    if not os.path.exists(index_file):
        index_content = html_template.format(
            title="CrashTransformer Documentation",
            content="""
            <h1>🔧 CrashTransformer Documentation</h1>
            <p>Welcome to the CrashTransformer documentation! This comprehensive guide will help you get started with AI-powered crash analysis.</p>
            
            <h2>📚 Quick Navigation</h2>
            <ul>
                <li><a href="00_START_HERE.html">🚀 Start Here</a> - Begin your journey with CrashTransformer</li>
                <li><a href="INTRODUCTION.html">📖 Introduction</a> - Learn what CrashTransformer does</li>
                <li><a href="USAGE_GUIDE.html">📋 Usage Guide</a> - Complete usage documentation</li>
                <li><a href="CLI_REFERENCE.html">🧰 CLI Reference</a> - All available commands</li>
                <li><a href="PROVIDERS_GUIDE.html">🤖 Providers Guide</a> - LLM provider setup</li>
                <li><a href="NEO4J_GUIDE.html">🗄️ Neo4j Guide</a> - Graph database integration</li>
                <li><a href="OUTPUTS.html">📤 Outputs</a> - Understanding results</li>
                <li><a href="COST_PERFORMANCE.html">💰 Cost & Performance</a> - Optimization guide</li>
                <li><a href="FINE_TUNING_GUIDE.html">🎯 Fine-tuning Guide</a> - Custom model training</li>
                <li><a href="TROUBLESHOOTING_FAQ.html">🛠️ Troubleshooting FAQ</a> - Common issues and solutions</li>
            </ul>
            
            <h2>🚀 Quick Start</h2>
            <ol>
                <li>Run <code>python crashtransformer.py setup</code> to configure your environment</li>
                <li>Prepare your crash data in CSV format with required columns</li>
                <li>Run <code>python crashtransformer.py run --csv your_data.csv</code> to analyze crashes</li>
                <li>Check the results in the <code>artifacts/</code> directory</li>
            </ol>
            
            <h2>💡 Need Help?</h2>
            <p>If you encounter any issues, check the <a href="TROUBLESHOOTING_FAQ.html">Troubleshooting FAQ</a> or run <code>python crashtransformer.py help</code> for command-line assistance.</p>
            """
        )
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        print(f"✅ Created: index.html")
    
    # Open in browser
    try:
        webbrowser.open(f"file://{os.path.abspath(index_file)}")
        print(f"📚 Documentation opened in browser: {index_file}")
        print(f"📁 HTML files generated in: {docs_html_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"📁 HTML files generated in: {docs_html_dir}")
        print(f"🌐 Open manually: file://{os.path.abspath(index_file)}")
        return True

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
        print("6. Docs (convert docs to HTML and serve in browser)")
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
        
    elif command == "clean-db":
        # Clean Neo4j database
        clean_database()
        
    elif command in ["help", "--help", "-h"]:
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Run 'python crashtransformer.py help' for usage information")

if __name__ == "__main__":
    main()
