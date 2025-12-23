# RISCORE Package

This package contains the reusable RISCORE implementation. Use the
[root README](../README.md) for the paper overview and [docs/USAGE.md](../docs/USAGE.md)
for end-to-end commands.

## Modules

- [core](core/README.md): model wrappers for local Hugging Face and API models.
- [data](data/README.md): normalized riddle examples and dataset loaders.
- [prompting](prompting/README.md): baselines, RISCORE prompts, and reconstruction generation.
- [evaluation](evaluation/README.md): metrics, result records, and evaluation loops.
- [utils](utils/README.md): config, validation, embeddings, and helper utilities.

## Public Entry Points

Most users start with:

```python
from riscore.core import UnifiedModelInterface
from riscore.data import DatasetLoader
from riscore.prompting import RISCOREPrompt
from riscore.evaluation import Evaluator, ResultsManager
```

The package also exposes these classes from `riscore.__init__` for shorter
imports.
