"""Base prompting strategies for RISCORE framework."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..data.dataset import RiddleExample
import numpy as np

class BasePromptStrategy(ABC):
    """Abstract base class for prompting strategies."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize prompt strategy.
        
        Args:
            system_prompt: System-level instruction
        """
        self.system_prompt = system_prompt or self.get_default_system_prompt()
    
    @abstractmethod
    def get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        pass
    
    @abstractmethod
    def format_prompt(self, example: RiddleExample, **kwargs) -> str:
        """
        Format prompt for a single example.
        
        Args:
            example: Riddle example to format
            **kwargs: Additional formatting parameters
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def format_question_with_choices(self, question: str, choices: List[str]) -> str:
        """
        Format question with multiple choice options.
        
        Args:
            question: Question text
            choices: List of answer choices
            
        Returns:
            Formatted string
        """
        formatted_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{question}\n{formatted_choices}"
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Extracted answer
        """
        import re
        
        # Try to find answer patterns like "Answer: A" or "The answer is A"
        patterns = [
            r"(?:answer|Answer|ANSWER)\s*(?:is|:)?\s*([A-D])",
            r"(?:correct answer|option)\s*(?:is|:)?\s*([A-D])",
            r"\b([A-D])\s*(?:is correct|is the answer)",
            r"^([A-D])\s*$",  # Single letter answer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, try to find the first occurrence of A, B, C, or D
        match = re.search(r"\b([A-D])\b", response)
        if match:
            return match.group(1).upper()
        
        return ""


class ZeroShotPrompt(BasePromptStrategy):
    """Zero-shot prompting strategy."""
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for zero-shot."""
        return (
            "You are an expert at solving riddles. "
            "Analyze the question carefully and select the best answer from the given choices. "
            "Provide your answer as a single letter (A, B, C, or D)."
        )
    
    def format_prompt(self, example: RiddleExample, **kwargs) -> str:
        """Format zero-shot prompt."""
        question_text = self.format_question_with_choices(example.question, example.choices)
        return f"{question_text}\n\nAnswer:"


class ZeroShotCoTPrompt(BasePromptStrategy):
    """Zero-shot Chain-of-Thought prompting strategy."""
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for zero-shot CoT."""
        return (
            "You are an expert at solving riddles. "
            "Think step by step to analyze the question and arrive at the correct answer. "
            "Explain your reasoning process, then provide your final answer as a single letter (A, B, C, or D)."
        )
    
    def format_prompt(self, example: RiddleExample, **kwargs) -> str:
        """Format zero-shot CoT prompt."""
        question_text = self.format_question_with_choices(example.question, example.choices)
        return f"{question_text}\n\nLet's think step by step:\n"


class FewShotPrompt(BasePromptStrategy):
    """Few-shot prompting strategy."""
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for few-shot."""
        return (
            "You are an expert at solving riddles. "
            "Study the examples below and use similar reasoning to solve new riddles. "
            "Provide your answer as a single letter (A, B, C, or D)."
        )
    
    def format_example(self, example: RiddleExample, include_answer: bool = True) -> str:
        """
        Format a single example for few-shot learning.
        
        Args:
            example: Riddle example
            include_answer: Whether to include the answer
            
        Returns:
            Formatted example string
        """
        question_text = self.format_question_with_choices(example.question, example.choices)
        
        if include_answer:
            answer_letter = chr(65 + example.answer_idx)  # Convert 0->A, 1->B, etc.
            return f"{question_text}\nAnswer: {answer_letter}"
        else:
            return f"{question_text}\nAnswer:"
    
    def format_prompt(
        self,
        example: RiddleExample,
        exemplars: Optional[List[RiddleExample]] = None,
        **kwargs
    ) -> str:
        """
        Format few-shot prompt.
        
        Args:
            example: Target example to solve
            exemplars: List of example riddles for few-shot learning
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        if not exemplars:
            # Fall back to zero-shot if no exemplars provided
            return ZeroShotPrompt().format_prompt(example)
        
        # Format exemplars
        exemplar_texts = [self.format_example(ex, include_answer=True) for ex in exemplars]
        exemplars_str = "\n\n".join(exemplar_texts)
        
        # Format target question
        target_text = self.format_example(example, include_answer=False)
        
        return f"{exemplars_str}\n\n{target_text}"


class FewShotCoTPrompt(FewShotPrompt):
    """Few-shot Chain-of-Thought prompting strategy."""
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for few-shot CoT."""
        return (
            "You are an expert at solving riddles. "
            "Study the examples below, paying attention to the reasoning process. "
            "Use similar step-by-step thinking to solve new riddles. "
            "Explain your reasoning, then provide your final answer as a single letter (A, B, C, or D)."
        )
    
    def format_example(self, example: RiddleExample, include_answer: bool = True) -> str:
        """Format example with chain-of-thought reasoning."""
        question_text = self.format_question_with_choices(example.question, example.choices)
        
        if include_answer:
            answer_letter = chr(65 + example.answer_idx)
            
            # Include CoT if available
            if example.cot:
                return f"{question_text}\n\nReasoning: {example.cot}\nAnswer: {answer_letter}"
            else:
                return f"{question_text}\nAnswer: {answer_letter}"
        else:
            return f"{question_text}\n\nReasoning:"


class SelfConsistencyPrompt(BasePromptStrategy):
    """Self-consistency prompting strategy (multiple reasoning paths)."""
    
    def __init__(
        self,
        base_strategy: BasePromptStrategy,
        num_paths: int = 5,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize self-consistency strategy.
        
        Args:
            base_strategy: Base prompting strategy to use
            num_paths: Number of reasoning paths to generate
            system_prompt: System prompt (uses base strategy's if None)
        """
        self.base_strategy = base_strategy
        self.num_paths = num_paths
        super().__init__(system_prompt or base_strategy.system_prompt)
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return self.base_strategy.get_default_system_prompt()
    
    def format_prompt(self, example: RiddleExample, **kwargs) -> str:
        """Format prompt using base strategy."""
        return self.base_strategy.format_prompt(example, **kwargs)
    
    def aggregate_answers(self, responses: List[str]) -> str:
        """
        Aggregate multiple responses using majority voting.
        
        Args:
            responses: List of model responses
            
        Returns:
            Most common answer
        """
        from collections import Counter
        
        # Extract answers from all responses
        answers = [self.extract_answer_from_response(resp) for resp in responses]
        
        # Filter out empty answers
        valid_answers = [ans for ans in answers if ans]
        
        if not valid_answers:
            return ""
        
        # Return most common answer
        counter = Counter(valid_answers)
        return counter.most_common(1)[0][0]


class ActivePrompt(BasePromptStrategy):
    """Active prompting strategy with uncertainty-based selection."""
    
    def __init__(
        self,
        base_strategy: BasePromptStrategy,
        uncertainty_threshold: float = 0.5,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize active prompting strategy.
        
        Args:
            base_strategy: Base prompting strategy
            uncertainty_threshold: Threshold for selecting uncertain examples
            system_prompt: System prompt
        """
        self.base_strategy = base_strategy
        self.uncertainty_threshold = uncertainty_threshold
        super().__init__(system_prompt or base_strategy.system_prompt)
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return self.base_strategy.get_default_system_prompt()
    
    def format_prompt(self, example: RiddleExample, **kwargs) -> str:
        """Format prompt using base strategy."""
        return self.base_strategy.format_prompt(example, **kwargs)
    
    def calculate_uncertainty(self, responses: List[str]) -> float:
        """
        Calculate uncertainty score for multiple responses.
        
        Args:
            responses: List of model responses
            
        Returns:
            Uncertainty score (0-1, higher means more uncertain)
        """
        from collections import Counter
        
        answers = [self.extract_answer_from_response(resp) for resp in responses]
        valid_answers = [ans for ans in answers if ans]
        
        if not valid_answers:
            return 1.0  # Maximum uncertainty
        
        # Calculate entropy-based uncertainty
        counter = Counter(valid_answers)
        total = len(valid_answers)
        
        # If all answers are the same, uncertainty is 0
        if len(counter) == 1:
            return 0.0
        
        # Calculate normalized entropy
        entropy = -sum((count/total) * np.log(count/total) for count in counter.values())
        max_entropy = np.log(len(counter))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
