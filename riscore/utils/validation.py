"""Pydantic models for validated outputs."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class AnswerExtraction(BaseModel):
    """Validated answer extraction from model output."""
    
    answer: str = Field(..., description="Extracted answer letter (A, B, C, or D)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Extracted reasoning text")
    raw_response: str = Field(..., description="Full model response")
    
    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Ensure answer is a valid letter."""
        v = v.upper().strip()
        if v not in ["A", "B", "C", "D"]:
            # Try to extract from string
            for char in v:
                if char in ["A", "B", "C", "D"]:
                    return char
            raise ValueError(f"Answer must be A, B, C, or D, got: {v}")
        return v
    
    def to_index(self) -> int:
        """Convert answer letter to index (A=0, B=1, C=2, D=3)."""
        return ord(self.answer) - ord("A")


class ReconstructedRiddle(BaseModel):
    """Validated reconstructed riddle."""
    
    question: str = Field(..., min_length=10, description="Reconstructed question")
    choices: List[str] = Field(..., min_length=4, max_length=4, description="Four answer choices")
    answer: str = Field(..., description="Correct answer letter")
    answer_idx: int = Field(..., ge=0, le=3, description="Index of correct answer")
    raw_response: str = Field(..., description="Full generation response")
    
    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate answer letter."""
        v = v.upper().strip()
        if v not in ["A", "B", "C", "D"]:
            raise ValueError(f"Answer must be A, B, C, or D, got: {v}")
        return v
    
    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: List[str]) -> List[str]:
        """Ensure all choices are non-empty."""
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 choices, got {len(v)}")
        for i, choice in enumerate(v):
            if not choice or not choice.strip():
                raise ValueError(f"Choice {i} is empty")
        return [c.strip() for c in v]
    
    @model_validator(mode='after')
    def validate_answer_idx(self):
        """Ensure answer_idx matches answer letter."""
        expected_idx = ord(self.answer) - ord("A")
        if self.answer_idx != expected_idx:
            self.answer_idx = expected_idx
        return self


class GenerationConfig(BaseModel):
    """Validated generation configuration."""
    
    temperature: float = Field(0.5, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    max_new_tokens: int = Field(700, ge=1, le=4096, description="Maximum tokens to generate")
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0, description="Repetition penalty")
    num_return_sequences: int = Field(1, ge=1, description="Number of sequences to return")
    do_sample: bool = Field(True, description="Whether to use sampling")
    
    class Config:
        frozen = False  # Allow updates


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    
    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    cache_dir: Optional[str] = Field(None, description="Directory to cache embeddings")
    device: Optional[str] = Field(None, description="Device for embedding model")
    normalize: bool = Field(True, description="Normalize embeddings")
    batch_size: int = Field(32, ge=1, description="Batch size for encoding")
    
    class Config:
        frozen = False


class SimilarityConfig(BaseModel):
    """Configuration for similarity-based selection."""
    
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Minimum similarity threshold")
    top_k: int = Field(5, ge=1, description="Number of similar examples to retrieve")
    metric: Literal["cosine", "euclidean", "dot"] = Field("cosine", description="Similarity metric")
    filter_duplicates: bool = Field(True, description="Filter duplicate examples")
    
    class Config:
        frozen = False


class PromptConfig(BaseModel):
    """Configuration for prompting strategy."""
    
    method: Literal["riscore", "fewshot", "fewshot_cot", "zeroshot", "zeroshot_cot", "selfcon"] = Field(
        "riscore",
        description="Prompting method"
    )
    num_exemplars: int = Field(2, ge=0, description="Number of exemplars")
    use_cot: bool = Field(True, description="Use chain-of-thought reasoning")
    use_similarity: bool = Field(True, description="Use similarity-based selection")
    similarity_config: Optional[SimilarityConfig] = Field(None, description="Similarity configuration")
    template_name: Optional[str] = Field(None, description="Custom template name")
    
    @model_validator(mode='after')
    def set_defaults(self):
        """Set default similarity config if needed."""
        if self.use_similarity and self.similarity_config is None:
            self.similarity_config = SimilarityConfig()
        return self
    
    class Config:
        frozen = False


class ModelProviderConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: Literal["huggingface", "openai", "anthropic", "cohere", "replicate", "google", "litellm"] = Field(
        "huggingface",
        description="Model provider"
    )
    model_name: str = Field(..., description="Model identifier")
    api_key: Optional[str] = Field(None, description="API key for provider")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    organization: Optional[str] = Field(None, description="Organization ID")
    
    # Hugging Face specific
    quantization: bool = Field(True, description="Use quantization (HF only)")
    load_in_4bit: bool = Field(True, description="4-bit quantization (HF only)")
    load_in_8bit: bool = Field(False, description="8-bit quantization (HF only)")
    
    class Config:
        frozen = False
