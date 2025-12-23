# Usage

This guide keeps setup and runnable examples out of the root README.

## Installation

```bash
git clone https://github.com/GiannisPana/RISCORE.git
cd RISCORE
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For API-backed models through LiteLLM:

```bash
pip install -e ".[api]"
```

Copy the example environment file if you need model credentials:

```bash
copy .env.example .env
```

Set `HF_TOKEN` for gated Hugging Face models, or provider keys such as
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, and
`REPLICATE_API_TOKEN` for API models.

## CLI Quickstart

Prepare data first using [DATA.md](DATA.md).

Run RISCORE evaluation with a local Hugging Face model:

```bash
riscore evaluate ^
  --model meta-llama/Meta-Llama-3-8B-Instruct ^
  --provider huggingface ^
  --dataset-type SP ^
  --data-dir data ^
  --method riscore ^
  --num-exemplars 2 ^
  --max-examples 20 ^
  --output-dir output
```

Generate context-reconstructed training examples:

```bash
riscore reconstruct ^
  --model meta-llama/Meta-Llama-3-8B-Instruct ^
  --provider huggingface ^
  --dataset-type SP ^
  --data-dir data ^
  --output-file data/SP-train_with_reconstructions.npy ^
  --max-examples 50
```

Run a small API-model experiment:

```bash
python examples/run_with_api_models.py ^
  --provider openai ^
  --model gpt-4o-mini ^
  --dataset-type SP ^
  --max-examples 10
```

## Python API

```python
from riscore.core import UnifiedModelInterface
from riscore.data import DatasetLoader
from riscore.evaluation import Evaluator, ResultsManager
from riscore.prompting import RISCOREPrompt

model = UnifiedModelInterface.create(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.5,
)

train = DatasetLoader.load_train_dataset("SP", "data")
test = DatasetLoader.load_test_dataset("SP", "data")

strategy = RISCOREPrompt(use_cot=True, num_exemplars=2)
manager = ResultsManager(output_dir="output")
evaluator = Evaluator(model=model, prompt_strategy=strategy, results_manager=manager)

evaluator.evaluate(test, exemplars=train.examples, max_new_tokens=700)
manager.print_summary()
```

## More Examples

See [examples/README.md](../examples/README.md) for the script-level guide.
