# Core

Model wrappers live here. They give evaluation code one common interface for
local Hugging Face models and API-backed providers.

## Main Classes

- `BaseModelConfig`: local model loading options such as quantization and device
  mapping.
- `ChatModel`: Hugging Face causal language model wrapper with chat-template
  formatting for Llama, Mistral, Phi, Gemma, and related models.
- `LiteLLMModel`: API-backed model wrapper powered by LiteLLM.
- `UnifiedModelInterface`: provider switch that exposes `generate`,
  `generate_chat`, `count_tokens`, and `get_model_info`.

## Typical Usage

```python
from riscore.core import UnifiedModelInterface

model = UnifiedModelInterface.create(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.5,
    max_tokens=700,
)
```

For local models:

```python
model = UnifiedModelInterface.create(
    provider="huggingface",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization=True,
    load_in_4bit=True,
)
```

Install the `api` extra for LiteLLM providers:

```bash
pip install -e ".[api]"
```
