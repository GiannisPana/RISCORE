# Reproducing RISCORE Runs

This repository provides reusable code for RISCORE-style prompting and
evaluation. Exact paper reproduction also depends on model access, generated
context reconstructions, and the same benchmark splits and decoding settings.

## Recommended Order

1. Install the package and optional provider dependencies.
2. Prepare BrainTeaser data with `scripts/prepare_data.py`.
3. Run a small evaluation with `--max-examples 10`.
4. Generate or load context-reconstructed training examples.
5. Run full SP/WP evaluations for RISCORE and baselines.

## Baselines

The CLI exposes:

- `zeroshot`
- `zeroshot_cot`
- `fewshot`
- `fewshot_cot`
- `selfcon`
- `riscore`

Example:

```bash
riscore evaluate --model meta-llama/Meta-Llama-3-8B-Instruct --provider huggingface --dataset-type SP --method fewshot_cot --max-examples 50
```

## Outputs

`ResultsManager` writes JSON, CSV, or text summaries under `output/`. JSON files
include run metadata, aggregate metrics, and per-example predictions.

## Differences From The Paper

- The code defaults are meant for practical local runs, not as locked paper
  hyperparameters.
- Reconstructed exemplars must be generated or supplied locally.
- API-provider models may change over time; record model names and dates in
  result metadata.
- RiddleSense is optional and has five choices, while the current default
  evaluation setup targets the four-choice BrainTeaser SP/WP data.
