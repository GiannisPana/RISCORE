"""Command-line interface for RISCORE framework."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from riscore.core import BaseModelConfig, ChatModel, UnifiedModelInterface
from riscore.data import DatasetLoader
from riscore.prompting import (
    RISCOREPrompt,
    FewShotPrompt,
    FewShotCoTPrompt,
    ZeroShotPrompt,
    ZeroShotCoTPrompt,
    SelfConsistencyPrompt,
    ContextReconstructionGenerator,
)
from riscore.evaluation import Evaluator, ResultsManager
from riscore.utils import ConfigManager, Logger
from riscore.utils.validation import EmbeddingConfig


def create_model(args):
    """Create model from arguments."""
    # Use UnifiedModelInterface if provider is specified
    if hasattr(args, 'provider') and args.provider:
        return UnifiedModelInterface.create(
            model_name=args.model,
            provider=args.provider,
            api_key=getattr(args, 'api_key', None),
            temperature=getattr(args, 'temperature', 0.7),
            max_tokens=getattr(args, 'max_tokens', 2048),
        )
    
    # Default to local HuggingFace model
    config = BaseModelConfig(
        model_name=args.model,
        quantization=args.quantization,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    model = ChatModel(config, hf_token=args.hf_token)
    return model


def create_prompt_strategy(args, model=None):
    """Create prompting strategy from arguments."""
    if args.method == "riscore":
        embedding_config = EmbeddingConfig(model_name=args.embedding_model)
        return RISCOREPrompt(
            use_cot=args.use_cot,
            num_exemplars=args.num_exemplars,
            similarity_based_selection=args.similarity_selection,
            similarity_threshold=args.similarity_threshold,
            embedding_config=embedding_config,
        )
    elif args.method == "fewshot":
        return FewShotPrompt()
    elif args.method == "fewshot_cot":
        return FewShotCoTPrompt()
    elif args.method == "zeroshot":
        return ZeroShotPrompt()
    elif args.method == "zeroshot_cot":
        return ZeroShotCoTPrompt()
    elif args.method == "selfcon":
        base_strategy = FewShotCoTPrompt() if args.use_cot else FewShotPrompt()
        return SelfConsistencyPrompt(
            base_strategy=base_strategy,
            num_paths=args.num_paths,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")


def run_evaluation(args):
    """Run evaluation experiment."""
    logger = Logger(verbose=args.verbose)
    
    logger.info(f"Starting RISCORE evaluation: {args.method}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset_type}")
    
    # Load model
    logger.info("Loading model...")
    model = create_model(args)
    logger.info(f"Model loaded: {model.get_model_info()}")
    
    # Load dataset
    logger.info("Loading dataset...")
    if args.dataset_file:
        dataset = DatasetLoader.load_dataset(args.dataset_file, args.dataset_type)
    else:
        if args.split == "train":
            dataset = DatasetLoader.load_train_dataset(args.dataset_type, args.data_dir)
        elif args.split == "test":
            dataset = DatasetLoader.load_test_dataset(args.dataset_type, args.data_dir)
        elif args.split == "val":
            dataset = DatasetLoader.load_validation_dataset(args.dataset_type, args.data_dir)
        else:
            raise ValueError(f"Unknown split: {args.split}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Load exemplars for few-shot
    exemplars = None
    if args.method in ["riscore", "fewshot", "fewshot_cot", "selfcon"]:
        logger.info(f"Loading {args.num_exemplars} exemplars...")
        train_dataset = DatasetLoader.load_train_dataset(args.dataset_type, args.data_dir)
        
        # Select exemplars
        if args.method == "riscore":
            prompt_strategy = create_prompt_strategy(args)
            # For RISCORE, exemplars will be selected per-example based on similarity
            exemplars = train_dataset.examples[:args.num_exemplars * 10]  # Get a pool
        else:
            import random
            random.seed(args.seed)
            exemplars = random.sample(train_dataset.examples, min(args.num_exemplars, len(train_dataset)))
    
    # Create prompt strategy
    prompt_strategy = create_prompt_strategy(args, model)
    
    # Create results manager
    results_manager = ResultsManager(output_dir=args.output_dir)
    results_manager.set_metadata(
        model_name=args.model,
        method=args.method,
        dataset_type=args.dataset_type,
        num_exemplars=args.num_exemplars if exemplars else 0,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        prompt_strategy=prompt_strategy,
        results_manager=results_manager,
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    generation_kwargs = {
        "model_type": args.model_type,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences if args.method == "selfcon" else 1,
    }
    
    # Limit dataset if specified
    if args.max_examples:
        dataset.examples = dataset.examples[:args.max_examples]
    
    results = evaluator.evaluate(
        dataset=dataset,
        exemplars=exemplars,
        verbose=args.verbose,
        **generation_kwargs
    )
    
    # Save results
    logger.info("Saving results...")
    output_file = results_manager.save_results(format=args.output_format)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    if args.print_summary:
        results_manager.print_summary()
    
    logger.info("Evaluation complete!")


def run_reconstruction_generation(args):
    """Generate context reconstructions for dataset."""
    logger = Logger(verbose=args.verbose)
    
    logger.info("Starting context reconstruction generation")
    logger.info(f"Model: {args.model}")
    
    # Load model
    logger.info("Loading model...")
    model = create_model(args)
    
    # Load dataset
    logger.info("Loading dataset...")
    if args.dataset_file:
        dataset = DatasetLoader.load_dataset(args.dataset_file, args.dataset_type)
    else:
        dataset = DatasetLoader.load_train_dataset(args.dataset_type, args.data_dir)
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Limit if specified
    if args.max_examples:
        dataset.examples = dataset.examples[:args.max_examples]
    
    # Create generator
    generator = ContextReconstructionGenerator(model, temperature=args.temperature)
    
    # Generate reconstructions
    logger.info("Generating reconstructions...")
    augmented_examples = generator.augment_dataset(
        examples=dataset.examples,
        model_type=args.model_type,
    )
    
    # Save augmented dataset
    output_path = Path(args.output_file)
    logger.info(f"Saving augmented dataset to: {output_path}")
    
    dataset.examples = augmented_examples
    
    if output_path.suffix == ".npy":
        dataset.to_numpy(output_path)
    elif output_path.suffix == ".json":
        dataset.to_json(output_path)
    elif output_path.suffix == ".csv":
        dataset.to_csv(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    logger.info("Context reconstruction generation complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RISCORE: RIddle Solving with COntext REconstruction"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    
    # Model arguments
    eval_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    eval_parser.add_argument("--provider", type=str, choices=["huggingface", "openai", "anthropic", "cohere", "replicate"],
                            help="Model provider (default: huggingface if not specified)")
    eval_parser.add_argument("--api-key", type=str, help="API key for provider (or set via environment variable)")
    eval_parser.add_argument("--model-type", type=str, default="llama3", 
                            choices=["llama3", "llama2", "mistral", "phi3", "gemma", "zephyr"],
                            help="Model type (for local models)")
    eval_parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    eval_parser.add_argument("--quantization", action="store_true", default=True, help="Use quantization (local models)")
    eval_parser.add_argument("--no-quantization", dest="quantization", action="store_false")
    eval_parser.add_argument("--load-in-4bit", action="store_true", default=True)
    eval_parser.add_argument("--load-in-8bit", action="store_true", default=False)
    
    # Dataset arguments
    eval_parser.add_argument("--dataset-type", type=str, default="SP", choices=["SP", "WP"], 
                            help="Dataset type")
    eval_parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    eval_parser.add_argument("--dataset-file", type=str, help="Specific dataset file")
    eval_parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"],
                            help="Dataset split")
    eval_parser.add_argument("--max-examples", type=int, help="Maximum examples to evaluate")
    
    # Prompting arguments
    eval_parser.add_argument("--method", type=str, default="riscore",
                            choices=["riscore", "fewshot", "fewshot_cot", "zeroshot", "zeroshot_cot", "selfcon"],
                            help="Prompting method")
    eval_parser.add_argument("--num-exemplars", type=int, default=2, help="Number of exemplars")
    eval_parser.add_argument("--use-cot", action="store_true", default=True, help="Use chain-of-thought")
    eval_parser.add_argument("--no-cot", dest="use_cot", action="store_false")
    eval_parser.add_argument("--similarity-selection", action="store_true", default=True)
    eval_parser.add_argument("--similarity-threshold", type=float, default=0.4)
    eval_parser.add_argument("--embedding-model", type=str, 
                            default="sentence-transformers/all-MiniLM-L6-v2")
    eval_parser.add_argument("--num-paths", type=int, default=5, help="Paths for self-consistency")
    
    # Generation arguments
    eval_parser.add_argument("--temperature", type=float, default=0.5)
    eval_parser.add_argument("--top-k", type=int, default=50)
    eval_parser.add_argument("--top-p", type=float, default=0.9)
    eval_parser.add_argument("--repetition-penalty", type=float, default=1.15)
    eval_parser.add_argument("--max-new-tokens", type=int, default=700)
    eval_parser.add_argument("--num-return-sequences", type=int, default=5)
    
    # Output arguments
    eval_parser.add_argument("--output-dir", type=str, default="output")
    eval_parser.add_argument("--output-format", type=str, default="json", choices=["json", "csv", "txt"])
    eval_parser.add_argument("--print-summary", action="store_true", default=True)
    eval_parser.add_argument("--verbose", action="store_true", default=True)
    
    # Other
    eval_parser.add_argument("--seed", type=int, default=42)
    
    # Reconstruct command
    recon_parser = subparsers.add_parser("reconstruct", help="Generate context reconstructions")
    
    recon_parser.add_argument("--model", type=str, required=True)
    recon_parser.add_argument("--provider", type=str, choices=["huggingface", "openai", "anthropic", "cohere", "replicate"],
                            help="Model provider (default: huggingface if not specified)")
    recon_parser.add_argument("--api-key", type=str, help="API key for provider")
    recon_parser.add_argument("--model-type", type=str, default="llama3")
    recon_parser.add_argument("--hf-token", type=str)
    recon_parser.add_argument("--quantization", action="store_true", default=True)
    recon_parser.add_argument("--load-in-4bit", action="store_true", default=True)
    recon_parser.add_argument("--load-in-8bit", action="store_true", default=False)
    recon_parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for API models")
    recon_parser.add_argument("--dataset-type", type=str, default="SP")
    recon_parser.add_argument("--data-dir", type=str, default="data")
    recon_parser.add_argument("--dataset-file", type=str)
    recon_parser.add_argument("--output-file", type=str, required=True)
    recon_parser.add_argument("--temperature", type=float, default=0.7)
    recon_parser.add_argument("--max-examples", type=int)
    recon_parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "reconstruct":
        run_reconstruction_generation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
