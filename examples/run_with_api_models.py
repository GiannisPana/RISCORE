"""Run RISCORE with local or API-backed models.

Examples:
    python examples/run_with_api_models.py --provider openai --model gpt-4o-mini
    python examples/run_with_api_models.py --provider anthropic --model claude-3-5-sonnet-20241022
    python examples/run_with_api_models.py --provider huggingface --model meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import os

from riscore.core import UnifiedModelInterface
from riscore.data import DatasetLoader
from riscore.evaluation import Evaluator, ResultsManager
from riscore.prompting import RISCOREPrompt


def get_api_key(provider: str) -> str | None:
    """Return the conventional environment variable for a provider."""
    env_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
        "huggingface": "HF_TOKEN",
    }
    name = env_names.get(provider)
    return os.getenv(name) if name else None


def run(args: argparse.Namespace) -> None:
    """Run a small RISCORE evaluation with the requested provider."""
    model = UnifiedModelInterface.create(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key or get_api_key(args.provider),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        quantization=args.quantization,
        load_in_4bit=args.load_in_4bit,
    )

    train_dataset = DatasetLoader.load_train_dataset(args.dataset_type, args.data_dir)
    test_dataset = DatasetLoader.load_test_dataset(args.dataset_type, args.data_dir)
    if args.max_examples:
        test_dataset.examples = test_dataset.examples[: args.max_examples]

    prompt_strategy = RISCOREPrompt(
        use_cot=not args.no_cot,
        num_exemplars=args.num_exemplars,
        similarity_based_selection=not args.no_similarity,
    )

    results_manager = ResultsManager(output_dir=args.output_dir)
    results_manager.set_metadata(
        provider=args.provider,
        model_name=args.model,
        method="riscore",
        dataset_type=args.dataset_type,
        num_exemplars=args.num_exemplars,
    )

    evaluator = Evaluator(model=model, prompt_strategy=prompt_strategy, results_manager=results_manager)
    evaluator.evaluate(
        dataset=test_dataset,
        exemplars=train_dataset.examples,
        verbose=True,
        model_type=args.model_type,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    output_file = results_manager.save_results(format="json")
    results_manager.print_summary()
    print(f"Saved results to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RISCORE with local or API-backed models.")
    parser.add_argument("--provider", default="openai", choices=["huggingface", "openai", "anthropic", "cohere", "replicate", "google", "litellm"])
    parser.add_argument("--model", default="gpt-4o-mini", help="Provider-specific model name")
    parser.add_argument("--api-key", help="Optional API key; defaults to provider environment variable")
    parser.add_argument("--dataset-type", default="SP", choices=["SP", "WP"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output/api_model")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--num-exemplars", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--model-type", default="llama3", help="Chat template for local Hugging Face models")
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--quantization", action="store_true", default=True)
    parser.add_argument("--no-quantization", dest="quantization", action="store_false")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
