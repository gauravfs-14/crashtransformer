#!/usr/bin/env python3
"""
CrashTransformer - AI-Powered Crash Analysis Pipeline
Main entry point for environment setup and pipeline execution
"""

import sys
import os
import glob
import webbrowser
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

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
    """Show comprehensive help"""
    print("""
🔧 CrashTransformer - AI-Powered Crash Analysis Pipeline
=======================================================

OVERVIEW:
CrashTransformer is an AI system that processes vehicle crash narratives 
into structured causal summaries using Large Language Models and 
transformer-based summarization.

MAIN COMMANDS:
  setup          Interactive environment configuration
  run            Execute the crash analysis pipeline
  train         Fine-tune models for crash summarization
  prepare-data   Prepare training data from pipeline outputs
  docs           Build and launch local HTML docs server
  validate       Validate current configuration
  install        Install required dependencies
  examples       Show usage examples
  help           Show this help message

QUICK START:
  1. python crashtransformer.py setup
  2. python crashtransformer.py run --csv crashes.csv

ENVIRONMENT SETUP:
  python crashtransformer.py setup
  python crashtransformer.py setup --interactive
  python crashtransformer.py setup --validate
  python crashtransformer.py setup --install

PIPELINE EXECUTION:
  python crashtransformer.py run --csv crashes.csv
  python crashtransformer.py run --csv crashes.csv --llm_provider anthropic
  python crashtransformer.py run --csv crashes.csv --neo4j_enabled

FEATURES:
  🤖 Multi-Provider LLM Support (OpenAI, Anthropic, Google, Groq, Ollama, XAI)
  📊 Graph Database Integration (Neo4j)
  📝 Comprehensive Logging
  💰 Cost Tracking and Optimization
  🔒 Secure Environment Configuration
  📈 Performance Analytics
  🎯 Quality Metrics

SUPPORTED PROVIDERS:
  • OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
  • Anthropic (Claude-3 Opus, Sonnet, Haiku)
  • Google (Gemini-1.5-Pro, Gemini-1.5-Flash)
  • Groq (Llama-3.1, Mixtral-8x7b)
  • Ollama (Local models: Llama3, Mistral, CodeLlama)
  • XAI (Grok-Beta, Grok-2)

OUTPUTS:
  • Structured crash graphs (JSON/JSONL)
  • Causal summaries with quality metrics
  • Neo4j graph database storage
  • Cost and performance reports
  • Detailed processing logs

EXAMPLES:
  # Basic usage
  python crashtransformer.py run --csv crashes.csv

  # Use Anthropic Claude
  python crashtransformer.py run --csv crashes.csv --llm_provider anthropic

  # Enable Neo4j storage
  python crashtransformer.py run --csv crashes.csv --neo4j_enabled

  # Debug mode
  python crashtransformer.py run --csv crashes.csv --log_level DEBUG

  # Multiple models
  python crashtransformer.py run --csv crashes.csv --batch_models facebook/bart-base t5-base

DOCS:
  # Build HTML from docs/*.md and serve locally
  python crashtransformer.py docs

CONFIGURATION:
  Environment variables are managed through .env file
  Run 'python crashtransformer.py setup' for interactive configuration

DOCUMENTATION:
  • INTERACTIVE_SETUP.md - Environment setup guide
  • ENVIRONMENT_SETUP.md - Configuration documentation
  • src/ - Source code and modules

SUPPORT:
  For issues and questions, check the documentation or run:
  python crashtransformer.py help
""")

def run_docs():
    """Convert docs/*.md to HTML and serve locally from docs_html/."""
    src_dir = os.path.join(os.path.dirname(__file__), 'docs')
    out_dir = os.path.join(os.path.dirname(__file__), 'docs_html')

    if not os.path.isdir(src_dir):
        print(f"❌ Docs directory not found: {src_dir}")
        return False

    os.makedirs(out_dir, exist_ok=True)

    # Try to use Python-Markdown (with GitHub-flavored extensions if available);
    # if not available, create a simple fallback wrapper
    md = None
    renderer_label = "plain"
    try:
        import markdown  # type: ignore
        # Preferred GitHub-flavored stack with pymdown-extensions
        try:
            extensions = [
                'extra',            # includes tables, fenced_code, def_list, footnotes
                'sane_lists',
                'admonition',
                'attr_list',
                'def_list',
                'footnotes',
                'toc',
                'pymdownx.github',
                'pymdownx.superfences',
                'pymdownx.highlight',
                'pymdownx.inlinehilite',
                'pymdownx.magiclink',
                'pymdownx.tasklist',
                'pymdownx.tilde',
                'pymdownx.emoji',
            ]
            extension_configs = {
                'toc': {'permalink': True, 'baselevel': 1},
                'pymdownx.highlight': {
                    'linenums': False,
                    'anchor_linenums': False,
                    'guess_lang': False,
                    'auto_title': False,
                    'use_pygments': True
                },
                'pymdownx.tasklist': {'custom_checkbox': True},
                'pymdownx.github': {
                    'repo_url_shorthand': True,
                    'mention_emails': True,
                },
            }
            # Import to ensure availability; config above suffices
            import pymdownx  # type: ignore  # noqa: F401
            md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
            renderer_label = "github-flavored"
        except Exception:
            # Safe fallback without nl2br/codehilite
            try:
                extensions = [
                    'extra', 'sane_lists', 'admonition', 'attr_list', 'def_list', 'footnotes', 'toc'
                ]
                extension_configs = {'toc': {'permalink': True, 'baselevel': 1}}
                md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
                renderer_label = "standard"
            except Exception:
                md = None
    except Exception:
        pass

    # Prepare Pygments CSS for syntax highlighting (escaped for str.format)
    pygments_css = ""
    try:
        from pygments.formatters import HtmlFormatter  # type: ignore
        _css = HtmlFormatter().get_style_defs('.highlight')
        # Escape curly braces so str.format doesn't try to interpolate
        pygments_css = _css.replace('{', '{{').replace('}', '}}')
        renderer_label = renderer_label or "github-flavored"
    except Exception:
        pygments_css = ""

    # HTML template with CSS; double curly braces to escape str.format
    template = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>{title}</title>"
        "<style>"
        "body{{max-width:860px;margin:2rem auto;padding:0 1rem;"
        "font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.6;color:#24292e;background:#fff}}"
        "h1,h2,h3,h4,h5,h6{{font-weight:600;line-height:1.25;margin-top:1.8rem}}"
        "pre,code,kbd,samp{{font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,Monaco,monospace}}"
        "pre{{background:#f6f8fa;padding:12px;overflow:auto;border-radius:6px;white-space:pre}}"
        "code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}"
        "a{{color:#0969da;text-decoration:none}}a:hover{{text-decoration:underline}}"
        "ul,ol{{padding-left:1.5rem}}"
        ".codehilite pre{{background:#f6f8fa}}.highlight pre{{background:#f6f8fa}}"
        "blockquote{{margin:1rem 0;padding:0.5rem 1rem;border-left:4px solid #d0d7de;background:#f6f8fa}}"
        "hr{{border:0;border-top:1px solid #d0d7de;margin:2rem 0}}"
        "table{{border-collapse:collapse;width:100%}}"
        "th,td{{border:1px solid #d0d7de;padding:6px 10px;text-align:left}}"
        "{pygments_css}"
        "</style></head><body><main>{content}</main></body></html>"
    )

    generated = []
    for md_path in glob.glob(os.path.join(src_dir, '*.md')):
        name = Path(md_path).stem
        title = name.replace('_', ' ').title()
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                src = f.read()
            if md:
                md.reset()
                html_body = md.convert(src)
            else:
                # Minimal fallback if markdown lib missing
                html_body = f"<pre>{src.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</pre>"
            html = template.format(title=title, content=html_body, pygments_css=pygments_css)
            out_file = os.path.join(out_dir, f"{name}.html")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(html)
            generated.append((name, out_file))
        except Exception as e:
            print(f"⚠️ Failed to convert {md_path}: {e}")

    # Preferred ordering for navigation
    preferred = [
        '00_START_HERE',
        'INTRODUCTION',
        'ENVIRONMENT_SETUP',
        'INTERACTIVE_SETUP',
        'DATA_SPEC',
        'USAGE_GUIDE',
        'CLI_REFERENCE',
        'PROVIDERS_GUIDE',
        'NEO4J_GUIDE',
        'OUTPUTS',
        'COST_PERFORMANCE',
        'FINE_TUNING_GUIDE',
        'TROUBLESHOOTING_FAQ',
        'SECURITY',
        'CONTRIBUTING',
        'CHANGELOG',
        'README',
    ]
    order_index = {name: i for i, name in enumerate(preferred)}

    def sort_key(item):
        name, _ = item
        idx = order_index.get(name, 10_000)
        return (idx, name)

    # Write an index.html linking all pages in preferred order, then alpha
    links = []
    for name, out_file in sorted(generated, key=sort_key):
        href = f"{Path(out_file).name}"
        label = name.replace('_', ' ').title()
        links.append(f"<li><a href=\"{href}\">{label}</a></li>")
    intro = (
        "<p>Welcome to CrashTransformer. Start with <strong>Start Here</strong> and follow the sequence." \
        " You can open individual pages from the list below.</p>"
    )
    index_html = template.format(
        title='CrashTransformer Docs',
        content=("<h1>CrashTransformer Documentation</h1>" + intro + ("<ul>" + "\n".join(links) + "</ul>" if links else "<p>No docs found.</p>")),
        pygments_css=pygments_css
    )
    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)

    # Serve directory on an available port
    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=out_dir, **kwargs)

    port = 8080
    httpd = None
    for p in [8080, 8000, 8888, 0]:
        try:
            httpd = HTTPServer(('127.0.0.1', p), _Handler)
            port = httpd.server_address[1]
            break
        except OSError:
            continue

    if httpd is None:
        print('❌ Failed to start HTTP server for docs.')
        print(f'📁 Docs built at: {out_dir}')
        return False

    url = f"http://127.0.0.1:{port}/index.html"
    print(f"📚 Docs built in: {out_dir}")
    print(f"🧩 Docs renderer: {renderer_label}")
    print(f"🌐 Serving docs at: {url}")

    def _serve():
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        print('Press Ctrl+C to stop the docs server.')
        while True:
            thread.join(timeout=1)
    except KeyboardInterrupt:
        print('\n🛑 Stopping docs server...')
        try:
            httpd.shutdown()
        except Exception:
            pass
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
        print("6. Docs (build and launch local docs)")
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
