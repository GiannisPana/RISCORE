"""
Example: Using RISCORE with API Models (OpenAI, Anthropic, etc.)

This example shows how to use the UnifiedModelInterface to seamlessly work with
both local Hugging Face models and API-based models via LiteLLM.
"""

import os
from pathlib import Path
import sys

# Add riscore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from riscore.core import UnifiedModelInterface
from riscore.data import DatasetLoader
from riscore.prompting import RISCOREPrompt
from riscore.evaluation import EvaluationMetrics
from riscore.utils import ConfigManager


def run_with_openai():
    """Run RISCORE with OpenAI GPT models."""
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("Running RISCORE with OpenAI GPT-4...")
    
    # Create model interface - automatically uses LiteLLM for API models
    model = UnifiedModelInterface.create(
        model_name="gpt-4-turbo-preview",
        provider="openai"
    )
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    loader = DatasetLoader(data_dir)
    test_data = loader.load_dataset("SP_new_test.npy", limit=10)
    
    # Create RISCORE with template loading and embedding caching
    riscore = RISCOREPrompt(
        model=model,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=data_dir / "cache",  # Embeddings will be saved here
        use_templates=True,  # Load prompts from YAML templates
    )
    
    # Evaluate
    results = riscore.evaluate(test_data)
    
    # Calculate metrics
    metrics = EvaluationMetrics.from_results(results)
    print(f"\nAccuracy: {metrics.accuracy:.2%}")
    print(f"Average confidence: {metrics.confidence_mean:.2f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    metrics.save(output_dir / "openai_results.json")


def run_with_anthropic():
    """Run RISCORE with Anthropic Claude models."""
    # Ensure API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    print("Running RISCORE with Claude 3...")
    
    # Create model interface
    model = UnifiedModelInterface.create(
        model_name="claude-3-opus-20240229",
        provider="anthropic"
    )
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    loader = DatasetLoader(data_dir)
    test_data = loader.load_dataset("WP_new_test.npy", limit=10)
    
    # Create RISCORE with caching
    riscore = RISCOREPrompt(
        model=model,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        cache_dir=data_dir / "cache",
        use_templates=True,
    )
    
    # Evaluate
    results = riscore.evaluate(test_data)
    
    # Calculate metrics
    metrics = EvaluationMetrics.from_results(results)
    print(f"\nAccuracy: {metrics.accuracy:.2%}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    metrics.save(output_dir / "anthropic_results.json")


def run_with_local_model():
    """Run RISCORE with local Hugging Face model for comparison."""
    print("Running RISCORE with local model...")
    
    # Create model interface - automatically uses HF pipeline
    model = UnifiedModelInterface.create(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="huggingface",
        device_map="auto",
        load_in_4bit=True,  # Quantization for efficiency
    )
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    loader = DatasetLoader(data_dir)
    test_data = loader.load_dataset("SP_new_test.npy", limit=10)
    
    # Create RISCORE - embeddings will be cached and reused
    riscore = RISCOREPrompt(
        model=model,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=data_dir / "cache",
        use_templates=True,
    )
    
    # Evaluate
    results = riscore.evaluate(test_data)
    
    # Calculate metrics
    metrics = EvaluationMetrics.from_results(results)
    print(f"\nAccuracy: {metrics.accuracy:.2%}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    metrics.save(output_dir / "local_results.json")


def compare_all_models():
    """Compare results across different model providers."""
    print("Comparing all model providers...\n")
    
    all_results = {}
    
    # Try each provider
    providers = [
        ("Local (Llama-3-8B)", run_with_local_model),
    ]
    
    # Only add API providers if keys are available
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI (GPT-4)", run_with_openai))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic (Claude-3)", run_with_anthropic))
    
    for name, func in providers:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print('='*50)
        try:
            func()
            all_results[name] = "✓ Success"
        except Exception as e:
            all_results[name] = f"✗ Failed: {str(e)}"
            print(f"Error: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print('='*50)
    for name, status in all_results.items():
        print(f"{name}: {status}")


def customize_templates():
    """Show how to customize prompt templates."""
    print("Customizing prompt templates...")
    
    from riscore.prompting import PromptTemplateManager
    
    # Load template manager
    template_dir = Path(__file__).parent.parent / "riscore" / "prompting" / "templates"
    manager = PromptTemplateManager(template_dir)
    
    # Load and view RISCORE template
    riscore_template = manager.get_template("riscore")
    print(f"\nRISCORE Template Name: {riscore_template.name}")
    print(f"Description: {riscore_template.description}")
    print(f"\nSystem Prompt:\n{riscore_template.system_prompt[:200]}...")
    
    # Create custom template
    custom_template = manager.get_template("riscore")
    custom_template.system_prompt = "You are an expert riddle solver with deep reasoning abilities."
    
    # Use custom template
    model = UnifiedModelInterface.create(
        model_name="gpt-3.5-turbo",
        provider="openai"
    )
    
    riscore = RISCOREPrompt(
        model=model,
        use_templates=False,  # Will use default, but can override
    )
    
    # Override system prompt
    riscore.system_prompt = custom_template.system_prompt
    
    print("\nCustom template applied successfully!")


def check_embedding_cache():
    """Check and manage embedding cache."""
    print("Checking embedding cache...")
    
    from riscore.utils import EmbeddingManager
    
    data_dir = Path(__file__).parent.parent / "data"
    cache_dir = data_dir / "cache"
    
    # Create embedding manager
    manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=cache_dir
    )
    
    # Check cache stats
    cache_file = cache_dir / "embeddings.pkl"
    if cache_file.exists():
        manager.load()
        print(f"\nCache loaded from: {cache_file}")
        print(f"Cached embeddings: {len(manager.cache.cache)}")
        
        # Show memory usage
        import sys
        cache_size = sys.getsizeof(manager.cache.cache)
        print(f"Memory usage: {cache_size / 1024:.2f} KB")
    else:
        print(f"\nNo cache found at: {cache_file}")
        print("Cache will be created on first run")
    
    # Example: pre-compute embeddings for dataset
    loader = DatasetLoader(data_dir)
    train_data = loader.load_dataset("SP-train.npy", limit=100)
    
    print(f"\nPre-computing embeddings for {len(train_data)} samples...")
    
    texts = [item["question"] for item in train_data]
    embeddings = manager.encode(texts, show_progress=True)
    
    print(f"Computed {len(embeddings)} embeddings")
    
    # Save cache
    manager.save()
    print(f"Cache saved to: {cache_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RISCORE API Models Example")
    parser.add_argument(
        "--mode",
        choices=["openai", "anthropic", "local", "compare", "templates", "cache"],
        default="compare",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.mode == "openai":
        run_with_openai()
    elif args.mode == "anthropic":
        run_with_anthropic()
    elif args.mode == "local":
        run_with_local_model()
    elif args.mode == "compare":
        compare_all_models()
    elif args.mode == "templates":
        customize_templates()
    elif args.mode == "cache":
        check_embedding_cache()
