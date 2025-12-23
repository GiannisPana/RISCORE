"""RISCORE: RIddle Solving with COntext REconstruction framework."""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from .base_prompts import FewShotPrompt, FewShotCoTPrompt
from .templates import get_template_manager, PromptTemplate
from ..data.dataset import RiddleExample
from ..utils.embeddings import EmbeddingManager, EmbeddingCache
from ..utils.validation import EmbeddingConfig, SimilarityConfig, ReconstructedRiddle


class RISCOREPrompt(FewShotPrompt):
    """
    RISCORE prompting strategy with context reconstruction.
    
    This strategy uses pairs of riddles: original and context-reconstructed versions
    that share the same reasoning logic but different surface-level contexts.
    """
    
    def __init__(
        self,
        use_cot: bool = True,
        similarity_based_selection: bool = True,
        num_exemplars: int = 2,
        similarity_threshold: float = 0.4,
        embedding_config: Optional[EmbeddingConfig] = None,
        similarity_config: Optional[SimilarityConfig] = None,
        template_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize RISCORE prompting strategy.
        
        Args:
            use_cot: Whether to use chain-of-thought reasoning
            similarity_based_selection: Use similarity-based exemplar selection
            similarity_threshold: Minimum similarity threshold for exemplar selection
            embedding_model: Name of sentence transformer model for embeddings
            system_prompt: Custom system prompt
        """
        self.use_cot = use_cot
        self.similarity_based_selection = similarity_based_selection
        self.num_exemplars = num_exemplars
        
        # Setup configurations
        if embedding_config is None:
            embedding_config = EmbeddingConfig(cache_dir=str(cache_dir) if cache_dir else None)
        self.embedding_config = embedding_config
        
        if similarity_config is None:
            similarity_config = SimilarityConfig(threshold=similarity_threshold)
        self.similarity_config = similarity_config
        
        # Initialize embedding manager with cache
        self.embedding_manager = None
        if similarity_based_selection:
            cache = EmbeddingCache(cache_dir or Path("cache/embeddings"))
            self.embedding_manager = EmbeddingManager(embedding_config, cache)
        
        # Load template
        self.template_name = template_name
        self.template = self._load_template()
        
        super().__init__(system_prompt)
    
    def _load_template(self) -> Optional[PromptTemplate]:
        """Load prompt template."""
        if self.template_name:
            return get_template_manager().get_template(self.template_name)
        
        # Use default template based on CoT setting
        template_name = "riscore_system" if self.use_cot else "riscore_no_cot"
        return get_template_manager().get_template(template_name)
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for RISCORE."""
        if self.template:
            return self.template.system
        
        # Fallback if template not found
        return (
            "You are an expert at solving riddles by understanding the underlying reasoning logic. "
            "Study the examples below carefully. Each example shows an original riddle and a "
            "context-reconstructed version that uses the same reasoning logic in a different context. "
            "Focus on the reasoning pattern that connects them, not the surface-level words. "
            "Use this abstract reasoning to solve new riddles. "
            "Provide your final answer as a single letter (A, B, C, or D)."
        )
    
    def format_reconstruction_pair(
        self,
        example: RiddleExample,
        include_answer: bool = True
    ) -> str:
        """
        Format an example with its context reconstruction.
        
        Args:
            example: Riddle example with reconstruction
            include_answer: Whether to include the answer
            
        Returns:
            Formatted string with both original and reconstructed versions
        """
        # Original riddle
        original_text = self.format_question_with_choices(
            example.question,
            example.choices
        )
        
        result = f"Original Riddle:\n{original_text}\n"
        
        # Reconstructed riddle (if available)
        if example.reconstructed_question and example.reconstructed_choices:
            reconstructed_text = self.format_question_with_choices(
                example.reconstructed_question,
                example.reconstructed_choices
            )
            result += f"\nContext-Reconstructed Version:\n{reconstructed_text}\n"
        
        # Add CoT reasoning if available and requested
        if self.use_cot and example.cot:
            result += f"\nReasoning Logic:\n{example.cot}\n"
        
        # Add answer
        if include_answer:
            answer_letter = chr(65 + example.answer_idx)
            result += f"\nAnswer: {answer_letter}"
        else:
            result += "\nAnswer:"
        
        return result
    
    def format_prompt(
        self,
        example: RiddleExample,
        exemplars: Optional[List[RiddleExample]] = None,
        **kwargs
    ) -> str:
        """
        Format RISCORE prompt with context-reconstructed pairs.
        
        Args:
            example: Target example to solve
            exemplars: List of example riddles with reconstructions
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        if not exemplars:
            # Fall back to standard few-shot if no exemplars
            return super().format_prompt(example)

        num_exemplars = kwargs.get("num_exemplars", self.num_exemplars)
        if num_exemplars and len(exemplars) > num_exemplars:
            if self.similarity_based_selection:
                exemplars = self.find_similar_exemplars(
                    target_example=example,
                    candidate_examples=exemplars,
                    k=num_exemplars,
                    threshold=kwargs.get("similarity_threshold"),
                )
            else:
                exemplars = exemplars[:num_exemplars]
        
        # Format exemplar pairs
        exemplar_texts = [
            self.format_reconstruction_pair(ex, include_answer=True)
            for ex in exemplars
        ]
        exemplars_str = "\n\n---\n\n".join(exemplar_texts)
        
        # Format target question (original only, no reconstruction)
        target_text = self.format_question_with_choices(example.question, example.choices)
        target_str = f"Now solve this riddle:\n{target_text}\n"
        
        if self.use_cot:
            target_str += "\nReasoning Logic:\n"
        else:
            target_str += "\nAnswer:"
        
        return f"{exemplars_str}\n\n---\n\n{target_str}"
    
    def compute_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Compute embeddings for texts with caching.
        
        Args:
            texts: List of text strings
            use_cache: Whether to use cache
            
        Returns:
            Embeddings array
        """
        if self.embedding_manager is None:
            raise ValueError("Embedding manager not initialized")
        
        return self.embedding_manager.encode(texts, use_cache=use_cache)
    
    def find_similar_exemplars(
        self,
        target_example: RiddleExample,
        candidate_examples: List[RiddleExample],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[RiddleExample]:
        """
        Find top-k similar exemplars based on semantic similarity.
        
        Args:
            target_example: Target riddle to find exemplars for
            candidate_examples: Pool of candidate exemplars
            k: Number of exemplars to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of top-k similar exemplars
        """
        if not self.similarity_based_selection or self.embedding_manager is None:
            # Return random exemplars if similarity not enabled
            import random
            k = k or self.similarity_config.top_k
            return random.sample(candidate_examples, min(k, len(candidate_examples)))
        
        # Use config values if not provided
        k = k or self.similarity_config.top_k
        threshold = threshold or self.similarity_config.threshold
        
        # Use embedding manager to find similar
        target_text = target_example.question
        candidate_texts = [ex.question for ex in candidate_examples]
        
        similar_indices = self.embedding_manager.find_similar(
            query_text=target_text,
            candidate_texts=candidate_texts,
            top_k=k,
            threshold=threshold,
            metric=self.similarity_config.metric,
            filter_duplicates=self.similarity_config.filter_duplicates,
        )
        
        # If no exemplars found, fall back to random selection
        if not similar_indices:
            import random
            return random.sample(candidate_examples, min(k, len(candidate_examples)))
        
        # Return selected exemplars
        return [candidate_examples[idx] for idx, _ in similar_indices]


class ContextReconstructionGenerator:
    """
    Generator for creating context-reconstructed riddle versions.
    
    This class uses an LLM to generate alternative versions of riddles
    that maintain the same reasoning logic but change the surface context.
    """
    
    def __init__(self, model, temperature: float = 0.7):
        """
        Initialize context reconstruction generator.
        
        Args:
            model: Language model instance (from core.base_model)
            temperature: Sampling temperature for generation
        """
        self.model = model
        self.temperature = temperature
    
    def get_reconstruction_prompt(self, example: RiddleExample) -> str:
        """
        Create prompt for generating context reconstruction.
        
        Args:
            example: Original riddle example
            
        Returns:
            Prompt for reconstruction
        """
        question_text = "\n".join([
            f"{chr(65+i)}. {choice}" 
            for i, choice in enumerate(example.choices)
        ])
        
        prompt = f"""You are an expert at creating alternative versions of riddles that preserve the reasoning logic but change the context.

Original Riddle:
{example.question}
{question_text}
Answer: {example.answer}

Task: Create a NEW riddle that:
1. Uses the SAME reasoning logic/pattern as the original
2. Changes the context completely (different domain, scenario, objects)
3. Has the same difficulty level
4. Maintains the multiple-choice format with 4 options

Generate the reconstructed riddle in this format:
Question: [new riddle question]
A. [choice A]
B. [choice B]
C. [choice C]
D. [choice D]
Answer: [correct letter]

Reconstructed Riddle:"""
        
        return prompt
    
    def generate_reconstruction(
        self,
        example: RiddleExample,
        model_type: str = "llama3",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate context-reconstructed version of a riddle.
        
        Args:
            example: Original riddle example
            model_type: Type of model being used
            
        Returns:
            Dictionary with reconstructed question, choices, and answer
        """
        import re
        
        # Load reconstruction template
        template = get_template_manager().get_template("context_reconstruction")
        
        if template:
            # Use template
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example.choices)])
            system_prompt, user_prompt = template.format(
                question=example.question,
                choices=choices_text,
                answer=example.answer,
            )
        else:
            # Fallback to method
            system_prompt = "You are an expert at creating riddles and understanding reasoning patterns."
            user_prompt = self.get_reconstruction_prompt(example)
        
        # Generate reconstruction
        if hasattr(self.model, 'generate_chat'):
            response = self.model.generate_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type=model_type,
                temperature=self.temperature,
                max_new_tokens=500
            )
        else:
            response = self.model.generate(
                user_prompt,
                temperature=self.temperature,
                max_new_tokens=500
            )
        
        # Parse response
        question_match = re.search(r"Question:\s*(.+?)(?=\n[A-D]\.)", response, re.DOTALL)
        choices = re.findall(r"([A-D])\.\s*(.+?)(?=\n[A-D]\.|Answer:|$)", response, re.DOTALL)
        answer_match = re.search(r"Answer:\s*([A-D])", response)
        
        if not question_match or len(choices) < 4 or not answer_match:
            if validate:
                raise ValueError(f"Failed to parse reconstruction: {response}")
            # Return None values if not validating
            return None
        
        reconstructed_question = question_match.group(1).strip()
        reconstructed_choices = [choice[1].strip() for choice in choices[:4]]
        reconstructed_answer_letter = answer_match.group(1)
        reconstructed_answer_idx = ord(reconstructed_answer_letter) - 65
        
        # Validate with Pydantic if requested
        if validate:
            try:
                validated = ReconstructedRiddle(
                    question=reconstructed_question,
                    choices=reconstructed_choices,
                    answer=reconstructed_answer_letter,
                    answer_idx=reconstructed_answer_idx,
                    raw_response=response,
                )
                return {
                    "reconstructed_question": validated.question,
                    "reconstructed_choices": validated.choices,
                    "reconstructed_answer": validated.choices[validated.answer_idx],
                    "reconstructed_answer_idx": validated.answer_idx,
                }
            except Exception as e:
                raise ValueError(f"Validation failed: {e}")
        
        return {
            "reconstructed_question": reconstructed_question,
            "reconstructed_choices": reconstructed_choices,
            "reconstructed_answer": reconstructed_choices[reconstructed_answer_idx],
            "reconstructed_answer_idx": reconstructed_answer_idx,
        }
    
    def augment_dataset(
        self,
        examples: List[RiddleExample],
        model_type: str = "llama3"
    ) -> List[RiddleExample]:
        """
        Generate context reconstructions for a dataset.
        
        Args:
            examples: List of riddle examples
            model_type: Type of model
            
        Returns:
            List of examples with reconstructions added
        """
        from tqdm import tqdm
        
        augmented_examples = []
        
        for example in tqdm(examples, desc="Generating reconstructions"):
            try:
                reconstruction = self.generate_reconstruction(example, model_type)
                
                # Create new example with reconstruction
                augmented_example = RiddleExample(
                    question=example.question,
                    choices=example.choices,
                    answer=example.answer,
                    answer_idx=example.answer_idx,
                    riddle_id=example.riddle_id,
                    category=example.category,
                    reconstructed_question=reconstruction["reconstructed_question"],
                    reconstructed_choices=reconstruction["reconstructed_choices"],
                    reconstructed_answer=reconstruction["reconstructed_answer"],
                    reconstructed_answer_idx=reconstruction["reconstructed_answer_idx"],
                    cot=example.cot,
                    hint=example.hint,
                    explanation=example.explanation,
                    metadata=example.metadata,
                )
                
                augmented_examples.append(augmented_example)
                
            except Exception as e:
                print(f"Failed to generate reconstruction for example {example.riddle_id}: {e}")
                # Keep original example without reconstruction
                augmented_examples.append(example)
        
        return augmented_examples
