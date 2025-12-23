"""Example script: Run RISCORE evaluation."""

import os

from dotenv import load_dotenv

from riscore import (
    ChatModel,
    BaseModelConfig,
    DatasetLoader,
    RISCOREPrompt,
    Evaluator,
    ResultsManager,
)

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DATASET_TYPE = "SP"  # Sentence Puzzle
    NUM_EXEMPLARS = 2
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file before running this example")
    
    print("=" * 80)
    print("RISCORE Evaluation Example")
    print("=" * 80)
    
    # 1. Load Model
    print("\n1. Loading model...")
    config = BaseModelConfig(
        model_name=MODEL_NAME,
        quantization=True,
        load_in_4bit=True,
    )
    model = ChatModel(config, hf_token=HF_TOKEN)
    print(f"Model loaded: {model.get_model_info()}")
    
    # 2. Load Datasets
    print("\n2. Loading datasets...")
    train_dataset = DatasetLoader.load_train_dataset(dataset_type=DATASET_TYPE)
    test_dataset = DatasetLoader.load_test_dataset(dataset_type=DATASET_TYPE)
    print(f"Train: {len(train_dataset)} examples")
    print(f"Test: {len(test_dataset)} examples")
    
    # 3. Create RISCORE Prompt Strategy
    print("\n3. Creating RISCORE prompt strategy...")
    prompt_strategy = RISCOREPrompt(
        use_cot=True,
        similarity_based_selection=True,
        similarity_threshold=0.4,
    )
    
    # 4. Setup Evaluation
    print("\n4. Setting up evaluation...")
    results_manager = ResultsManager(output_dir="output")
    results_manager.set_metadata(
        model_name=MODEL_NAME,
        method="riscore",
        dataset_type=DATASET_TYPE,
        num_exemplars=NUM_EXEMPLARS,
    )
    
    evaluator = Evaluator(
        model=model,
        prompt_strategy=prompt_strategy,
        results_manager=results_manager,
    )
    
    # 5. Run Evaluation (on first 10 examples for demo)
    print("\n5. Running evaluation...")
    test_dataset.examples = test_dataset.examples[:10]  # Demo: first 10 only
    
    results = evaluator.evaluate(
        dataset=test_dataset,
        exemplars=train_dataset.examples[:NUM_EXEMPLARS * 10],  # Pool for selection
        model_type="llama3",
        temperature=0.5,
        top_k=50,
        repetition_penalty=1.15,
        verbose=True,
    )
    
    # 6. Save and Display Results
    print("\n6. Saving results...")
    output_file = results_manager.save_results(format="json")
    print(f"Results saved to: {output_file}")
    
    print("\n7. Results Summary:")
    results_manager.print_summary()
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
