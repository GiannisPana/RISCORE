"""LiteLLM integration for API-based model providers."""

from typing import List, Optional, Union, Dict, Any
import os

try:
    import litellm
    from litellm import completion, embedding
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not installed. API providers will not be available.")
    print("Install with: pip install litellm")

from ..utils.validation import ModelProviderConfig, GenerationConfig, AnswerExtraction
import re


class LiteLLMModel:
    """
    Unified API model using LiteLLM.
    
    Supports:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - Anthropic (Claude)
    - Google (PaLM, Gemini)
    - Cohere
    - Replicate
    - Hugging Face Inference API
    - And many more via LiteLLM
    """
    
    def __init__(
        self,
        config: ModelProviderConfig,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize LiteLLM model.
        
        Args:
            config: Model provider configuration
            generation_config: Generation parameters
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for API models. Install with: pip install litellm")
        
        self.config = config
        self.generation_config = generation_config or GenerationConfig()
        
        # Set API keys from config or environment
        self._setup_api_keys()
    
    def _setup_api_keys(self):
        """Setup API keys for providers."""
        if self.config.api_key:
            # Set provider-specific API key
            if "openai" in self.config.model_name.lower() or self.config.provider == "openai":
                os.environ["OPENAI_API_KEY"] = self.config.api_key
            elif "claude" in self.config.model_name.lower() or self.config.provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
            elif "palm" in self.config.model_name.lower() or "gemini" in self.config.model_name.lower():
                os.environ["GOOGLE_API_KEY"] = self.config.api_key
            elif "cohere" in self.config.model_name.lower():
                os.environ["COHERE_API_KEY"] = self.config.api_key
        
        if self.config.api_base:
            os.environ["OPENAI_API_BASE"] = self.config.api_base
        
        if self.config.organization:
            os.environ["OPENAI_ORGANIZATION"] = self.config.organization
    
    def _format_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """Format prompts as messages for chat models."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using LiteLLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text or list of texts
        """
        # Merge generation configs
        gen_config = self.generation_config.model_copy()
        for key, value in kwargs.items():
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)
        
        # Format messages
        messages = self._format_messages(
            system_prompt or "",
            prompt
        )
        
        # Prepare litellm kwargs
        litellm_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "max_tokens": gen_config.max_new_tokens,
            "n": gen_config.num_return_sequences,
        }
        
        # Call LiteLLM
        try:
            response = completion(**litellm_kwargs)
            
            # Extract generated text(s)
            if gen_config.num_return_sequences == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation failed: {e}")
    
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate response for chat interaction.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            **kwargs: Generation parameters
            
        Returns:
            Generated response(s)
        """
        return self.generate(user_prompt, system_prompt=system_prompt, **kwargs)
    
    def extract_answer(
        self,
        response: str,
    ) -> AnswerExtraction:
        """
        Extract and validate answer from response.
        
        Args:
            response: Model response
            
        Returns:
            Validated AnswerExtraction
        """
        # Try to extract answer letter
        patterns = [
            r"(?:answer|Answer|ANSWER)\s*(?:is|:)?\s*([A-D])",
            r"(?:correct answer|option)\s*(?:is|:)?\s*([A-D])",
            r"\b([A-D])\s*(?:is correct|is the answer)",
            r"^([A-D])\s*$",
            r"\*\*([A-D])\*\*",
        ]
        
        answer = None
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                break
        
        # If no pattern matches, try first A-D occurrence
        if answer is None:
            match = re.search(r"\b([A-D])\b", response)
            if match:
                answer = match.group(1).upper()
        
        # Default to "A" if nothing found (will be marked as failed validation)
        if answer is None:
            answer = "A"
        
        # Try to extract reasoning
        reasoning = None
        reasoning_patterns = [
            r"(?:Reasoning|Explanation|Let's think step by step):\s*(.+?)(?:Answer:|$)",
            r"(.+?)(?:Therefore|Thus|So),?\s*(?:the answer is|answer:)",
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                break
        
        return AnswerExtraction(
            answer=answer,
            reasoning=reasoning,
            raw_response=response,
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        try:
            # Use litellm's token counter
            return litellm.token_counter(model=self.config.model_name, text=text)
        except:
            # Fallback to rough estimate (4 chars per token)
            return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "api_configured": bool(self.config.api_key or os.getenv("OPENAI_API_KEY")),
        }


class UnifiedModelInterface:
    """
    Unified interface supporting both local (Hugging Face) and API models.
    
    Automatically selects the appropriate backend based on configuration.
    """
    
    def __init__(
        self,
        config: ModelProviderConfig,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize unified model interface.
        
        Args:
            config: Model provider configuration
            generation_config: Generation parameters
        """
        self.config = config
        self.generation_config = generation_config or GenerationConfig()
        self.model = None
        
        self._initialize_model()

    @classmethod
    def create(
        cls,
        model_name: str,
        provider: str = "huggingface",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ) -> "UnifiedModelInterface":
        """Create a unified model interface from simple provider arguments."""
        model_config_keys = {"quantization", "load_in_4bit", "load_in_8bit"}
        generation_config_keys = set(GenerationConfig.model_fields)

        model_config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in model_config_keys}
        generation_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in generation_config_keys}

        if "max_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        config = ModelProviderConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            organization=organization,
            **model_config_kwargs,
        )
        generation_config = GenerationConfig(**generation_kwargs)
        return cls(config=config, generation_config=generation_config)
    
    def _initialize_model(self):
        """Initialize the appropriate model backend."""
        if self.config.provider == "huggingface":
            # Use local Hugging Face model
            from ..core.base_model import BaseModelConfig, ChatModel
            
            hf_config = BaseModelConfig(
                model_name=self.config.model_name,
                quantization=self.config.quantization,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )
            
            self.model = ChatModel(hf_config, hf_token=self.config.api_key)
        
        elif self.config.provider in ["openai", "anthropic", "cohere", "replicate", "google", "litellm"]:
            # Use LiteLLM for API providers
            self.model = LiteLLMModel(self.config, self.generation_config)
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(self, prompt: str, **kwargs) -> Union[str, List[str]]:
        """Generate text."""
        return self.model.generate(prompt, **kwargs)
    
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate chat response."""
        if hasattr(self.model, 'generate_chat'):
            # Handle model_type for HF models
            if self.config.provider == "huggingface" and "model_type" not in kwargs:
                # Auto-detect model type from name
                model_name_lower = self.config.model_name.lower()
                if "llama-3" in model_name_lower or "llama3" in model_name_lower:
                    kwargs["model_type"] = "llama3"
                elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
                    kwargs["model_type"] = "llama2"
                elif "mistral" in model_name_lower:
                    kwargs["model_type"] = "mistral"
                elif "phi" in model_name_lower:
                    kwargs["model_type"] = "phi3"
                elif "gemma" in model_name_lower:
                    kwargs["model_type"] = "gemma"
            
            return self.model.generate_chat(system_prompt, user_prompt, **kwargs)
        else:
            return self.model.generate(user_prompt, system_prompt=system_prompt, **kwargs)
    
    def extract_answer(self, response: str) -> AnswerExtraction:
        """Extract and validate answer from response."""
        if hasattr(self.model, 'extract_answer'):
            return self.model.extract_answer(response)
        else:
            # Use standalone extraction
            from ..prompting.base_prompts import BasePromptStrategy
            answer_str = BasePromptStrategy().extract_answer_from_response(response)
            return AnswerExtraction(answer=answer_str or "A", raw_response=response)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.model.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model.get_model_info()
