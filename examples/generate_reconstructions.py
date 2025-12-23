"""Example script: Generate context reconstructions."""

import os

from dotenv import load_dotenv

from riscore import (
    ChatModel,
    BaseModelConfig,
    DatasetLoader,
    ContextReconstructionGenerator,
)

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DATASET_TYPE = "SP"
    OUTPUT_FILE = "data/SP-train_with_reconstructions.npy"
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file before running this example")
    NUM_EXAMPLES = 10  # Number of examples to augment (use -1 for all)
    
    print("=" * 80)
    print("Context Reconstruction Generation")
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
    
    # 2. Load Training Dataset
    print("\n2. Loading training dataset...")
    dataset = DatasetLoader.load_train_dataset(dataset_type=DATASET_TYPE)
    print(f"Loaded {len(dataset)} examples")
    
    # Limit examples for demo
    if NUM_EXAMPLES > 0:
        dataset.examples = dataset.examples[:NUM_EXAMPLES]
        print(f"Processing first {NUM_EXAMPLES} examples (demo mode)")
    
    # 3. Create Generator
    print("\n3. Creating context reconstruction generator...")
    generator = ContextReconstructionGenerator(
        model=model,
        temperature=0.7,
    )
    
    # 4. Generate Reconstructions
    print("\n4. Generating context reconstructions...")
    print("This may take a while...\n")
    
    augmented_examples = generator.augment_dataset(
        examples=dataset.examples,
        model_type="llama3",
    )
    
    # 5. Save Augmented Dataset
    print("\n5. Saving augmented dataset...")
    dataset.examples = augmented_examples
    dataset.to_numpy(OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")
    
    # 6. Show Example
    print("\n6. Example reconstruction:")
    print("-" * 80)
    example = augmented_examples[0]
    print(f"Original Question: {example.question}")
    print(f"Original Answer: {example.answer}")
    if example.reconstructed_question:
        print(f"\nReconstructed Question: {example.reconstructed_question}")
        print(f"Reconstructed Answer: {example.reconstructed_answer}")
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Context Reconstruction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
