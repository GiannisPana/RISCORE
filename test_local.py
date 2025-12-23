"""
Quick test script for running RISCORE locally with a small model.
No API keys required - just a free Hugging Face token.
"""

import os
from dotenv import load_dotenv
from riscore import (
    ChatModel,
    BaseModelConfig,
    DatasetLoader,
    RISCOREPrompt,
    Evaluator,
    ResultsManager
)

# Load environment variables from .env file
load_dotenv()

def main():
    """Run RISCORE with a small quantized model locally."""
    
    print("="*80)
    print("RISCORE Local Test - Small Quantized Model")
    print("="*80)
    
    # 1. Load small quantized model (Phi-2 - 2.7B parameters)
    print("\n[1/5] Loading model (Phi-2 - 2.7B, 4-bit quantized)...")
    print("      First run will download model (~2GB) - this may take a few minutes")
    
    config = BaseModelConfig(
        model_name="microsoft/phi-2",  # Small, fast model
        quantization=True,  # Enable 4-bit quantization (saves memory)
        device_map="auto"  # Auto-detect GPU/CPU
    )
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment!")
        print("   Please set it in .env file or export HF_TOKEN='your_token'")
        return
    
    model = ChatModel(config, hf_token=hf_token)
    print("Model loaded successfully!")
    
    # 2. Load small test dataset
    print("\n[2/5] Loading dataset (first 10 examples for quick test)...")
    test_dataset = DatasetLoader.load_test_dataset(dataset_type="SP")
    test_dataset.examples = test_dataset.examples[:10]  # Only 10 for testing
    
    train_dataset = DatasetLoader.load_train_dataset(dataset_type="SP")
    print(f"Loaded {len(test_dataset)} test examples")
    print(f"Loaded {len(train_dataset)} training examples")
    
    # 3. Create RISCORE strategy
    print("\n[3/5] Creating RISCORE strategy...")
    strategy = RISCOREPrompt(
        use_cot=True,  # Use Chain-of-Thought reasoning
        similarity_based_selection=True  # Select similar examples
    )
    print("Strategy created")
    
    # 4. Run evaluation
    print("\n[4/5] Running evaluation...")
    print("      This will take a few minutes (processing 10 examples)")
    
    evaluator = Evaluator(
        model=model,
        prompt_strategy=strategy,
        results_manager=ResultsManager(output_dir="output/")
    )
    
    results = evaluator.evaluate(
        dataset=test_dataset,
        verbose=True,  # Show progress
        exemplars=train_dataset.examples[:20],  # Use 20 training examples
        model_type="phi3",  # Phi-2 uses similar format to Phi-3
        temperature=0.5,
        max_new_tokens=300  # Reduced for faster generation
    )
    
    # 5. Show results
    print("\n[5/5] Results:")
    print("="*80)
    evaluator.results_manager.print_summary()
    
    # Save results
    evaluator.results_manager.save_results(format="json")
    print(f"\nDetailed results saved to: output/")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check results in output/ directory")
    print("2. Try running on full dataset (remove [:10] limit)")
    print("3. Experiment with different models (see README.md)")
    print("4. Compare with other methods (examples/compare_methods.py)")

if __name__ == "__main__":
    main()
