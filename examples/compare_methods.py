"""Example script: Compare different prompting methods."""

from riscore import (
    ChatModel,
    BaseModelConfig,
    DatasetLoader,
    ZeroShotPrompt,
    ZeroShotCoTPrompt,
    FewShotPrompt,
    FewShotCoTPrompt,
    RISCOREPrompt,
    Evaluator,
    ResultsManager,
)
import random

def evaluate_method(model, method_name, prompt_strategy, test_dataset, exemplars=None):
    """Evaluate a single prompting method."""
    print(f"\nEvaluating {method_name}...")
    
    results_manager = ResultsManager(output_dir=f"output/{method_name}")
    results_manager.set_metadata(method=method_name)
    
    evaluator = Evaluator(model, prompt_strategy, results_manager)
    
    results = evaluator.evaluate(
        dataset=test_dataset,
        exemplars=exemplars,
        model_type="llama3",
        temperature=0.5,
        verbose=False,
    )
    
    metrics = results_manager.get_metrics()
    accuracy = metrics["accuracy"]
    
    print(f"{method_name} Accuracy: {accuracy:.2%}")
    
    return accuracy, results_manager


def main():
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DATASET_TYPE = "SP"
    NUM_EXEMPLARS = 2
    HF_TOKEN = "your_hf_token_here"
    NUM_TEST_EXAMPLES = 20  # Use small number for demo
    
    print("=" * 80)
    print("Comparing Prompting Methods")
    print("=" * 80)
    
    # Load Model
    print("\nLoading model...")
    config = BaseModelConfig(model_name=MODEL_NAME, quantization=True)
    model = ChatModel(config, hf_token=HF_TOKEN)
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = DatasetLoader.load_train_dataset(dataset_type=DATASET_TYPE)
    test_dataset = DatasetLoader.load_test_dataset(dataset_type=DATASET_TYPE)
    
    # Limit test set for demo
    test_dataset.examples = test_dataset.examples[:NUM_TEST_EXAMPLES]
    
    # Select exemplars
    random.seed(42)
    exemplars = random.sample(train_dataset.examples, NUM_EXEMPLARS)
    
    # Methods to compare
    methods = {
        "Zero-Shot": (ZeroShotPrompt(), None),
        "Zero-Shot CoT": (ZeroShotCoTPrompt(), None),
        "Few-Shot": (FewShotPrompt(), exemplars),
        "Few-Shot CoT": (FewShotCoTPrompt(), exemplars),
        "RISCORE": (RISCOREPrompt(use_cot=True), train_dataset.examples[:20]),
    }
    
    # Run evaluations
    print("\n" + "=" * 80)
    print("Running Evaluations")
    print("=" * 80)
    
    results = {}
    for method_name, (strategy, exs) in methods.items():
        accuracy, manager = evaluate_method(
            model, method_name, strategy, test_dataset, exs
        )
        results[method_name] = accuracy
    
    # Display comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, accuracy) in enumerate(sorted_results, 1):
        print(f"{i}. {method:20s}: {accuracy:.2%}")
    
    print("\n" + "=" * 80)
    print(f"Best Method: {sorted_results[0][0]} ({sorted_results[0][1]:.2%})")
    print("=" * 80)


if __name__ == "__main__":
    main()
