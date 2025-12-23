# Utils

Shared helpers, validation models, configuration objects, and embedding support
live here.

## Configuration

`config.py` contains dataclass-based experiment configuration:

- `ModelConfig`
- `DatasetConfig`
- `PromptingConfig`
- `EvaluationConfig`
- `ExperimentConfig`
- `ConfigManager`

## Validation

`validation.py` contains Pydantic models for:

- extracted answers
- reconstructed riddles
- provider configuration
- generation configuration
- embedding and similarity settings

## Embeddings

`embeddings.py` wraps `sentence-transformers` and caches embeddings on disk.
RISCORE uses this for semantic exemplar retrieval when
`similarity_based_selection=True`.

## Helpers

`helpers.py` includes small text, answer extraction, voting, filename, time, and
logging utilities used across the package.
