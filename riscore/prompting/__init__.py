"""Prompting strategies module for RISCORE framework."""

from .base_prompts import (
    BasePromptStrategy,
    ZeroShotPrompt,
    ZeroShotCoTPrompt,
    FewShotPrompt,
    FewShotCoTPrompt,
    SelfConsistencyPrompt,
    ActivePrompt,
)

from .riscore import (
    RISCOREPrompt,
    ContextReconstructionGenerator,
)

from .templates import (
    PromptTemplate,
    PromptTemplateManager,
    get_template_manager,
    get_template,
)

__all__ = [
    "BasePromptStrategy",
    "ZeroShotPrompt",
    "ZeroShotCoTPrompt",
    "FewShotPrompt",
    "FewShotCoTPrompt",
    "SelfConsistencyPrompt",
    "ActivePrompt",
    "RISCOREPrompt",
    "ContextReconstructionGenerator",
    "PromptTemplate",
    "PromptTemplateManager",
    "get_template_manager",
    "get_template",
]
