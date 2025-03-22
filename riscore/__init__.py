"""
RISCORE: RIddle Solving with COntext REconstruction

A framework for enhancing in-context riddle solving in language models through
context-reconstructed example augmentation.
"""

__version__ = "0.1.0"

from riscore.core import (
    BaseModel,
    BaseModelConfig,
    CausalLMModel,
    ChatModel,
)

from riscore.data import (
    RiddleExample,
    RiddleDataset,
    DatasetLoader,
)

from riscore.prompting import (
    BasePromptStrategy,
    ZeroShotPrompt,
    ZeroShotCoTPrompt,
    FewShotPrompt,
    FewShotCoTPrompt,
    SelfConsistencyPrompt,
    RISCOREPrompt,
    ContextReconstructionGenerator,
)

from riscore.evaluation import (
    PredictionResult,
    EvaluationMetrics,
    ResultsManager,
    Evaluator,
)

from riscore.utils import (
    ConfigManager,
    ExperimentConfig,
    Logger,
)

__all__ = [
    # Core
    "BaseModel",
    "BaseModelConfig",
    "CausalLMModel",
    "ChatModel",
    # Data
    "RiddleExample",
    "RiddleDataset",
    "DatasetLoader",
    # Prompting
    "BasePromptStrategy",
    "ZeroShotPrompt",
    "ZeroShotCoTPrompt",
    "FewShotPrompt",
    "FewShotCoTPrompt",
    "SelfConsistencyPrompt",
    "RISCOREPrompt",
    "ContextReconstructionGenerator",
    # Evaluation
    "PredictionResult",
    "EvaluationMetrics",
    "ResultsManager",
    "Evaluator",
    # Utils
    "ConfigManager",
    "ExperimentConfig",
    "Logger",
]
