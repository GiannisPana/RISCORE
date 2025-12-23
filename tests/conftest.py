"""Lightweight dependency stubs for unit tests.

The project supports large optional ML dependencies. These tests exercise pure
Python behavior without requiring model runtimes to be installed.
"""

from __future__ import annotations

import sys
import types


def _install_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


torch = types.ModuleType("torch")
torch.float16 = "float16"
torch.dtype = object
_install_module("torch", torch)

transformers = types.ModuleType("transformers")
transformers.AutoModelForCausalLM = object
transformers.AutoTokenizer = object
transformers.BitsAndBytesConfig = object
_install_module("transformers", transformers)

datasets = types.ModuleType("datasets")


class _Dataset:
    @classmethod
    def from_list(cls, data):
        return list(data)


datasets.Dataset = _Dataset
_install_module("datasets", datasets)

pandas = types.ModuleType("pandas")
pandas.read_csv = lambda *args, **kwargs: None
pandas.DataFrame = lambda data: types.SimpleNamespace(to_csv=lambda *args, **kwargs: None)
pandas.notna = lambda value: value is not None
_install_module("pandas", pandas)

sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
_install_module("sentence_transformers", sentence_transformers)

sklearn = types.ModuleType("sklearn")
metrics = types.ModuleType("sklearn.metrics")
pairwise = types.ModuleType("sklearn.metrics.pairwise")
pairwise.cosine_similarity = lambda query, candidates: None
pairwise.euclidean_distances = lambda query, candidates: None
metrics.pairwise = pairwise
sklearn.metrics = metrics
_install_module("sklearn", sklearn)
_install_module("sklearn.metrics", metrics)
_install_module("sklearn.metrics.pairwise", pairwise)

nltk = types.ModuleType("nltk")
tokenize = types.ModuleType("nltk.tokenize")
tokenize.word_tokenize = lambda text: text.split()
nltk.tokenize = tokenize
nltk.download = lambda *args, **kwargs: True
_install_module("nltk", nltk)
_install_module("nltk.tokenize", tokenize)
