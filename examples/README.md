# Examples

These scripts are small entry points for trying RISCORE workflows after
installing the package and preparing data.

## Scripts

- `run_riscore_evaluation.py`: minimal RISCORE evaluation with a local Hugging
  Face chat model.
- `generate_reconstructions.py`: generate context-reconstructed training
  examples and save them back to `data/`.
- `compare_methods.py`: compare zero-shot, few-shot, CoT, and RISCORE prompting
  methods on a small subset.
- `run_with_api_models.py`: run RISCORE through `UnifiedModelInterface` with
  OpenAI, Anthropic, Hugging Face, or another LiteLLM-compatible provider.

## Before Running

1. Install the package from the repository root.
2. Prepare datasets using [docs/DATA.md](../docs/DATA.md).
3. Set credentials in `.env` or your shell environment.

Most examples default to a small subset so the workflow can be checked before
running full experiments.
