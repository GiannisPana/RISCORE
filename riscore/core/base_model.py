"""Base model handler for RISCORE framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

warnings.simplefilter("ignore")


class BaseModelConfig:
    """Configuration for model initialization."""
    
    def __init__(
        self,
        model_name: str,
        quantization: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: Hugging Face model identifier
            quantization: Whether to use quantization
            load_in_4bit: Load model in 4-bit precision
            load_in_8bit: Load model in 8-bit precision
            bnb_4bit_compute_dtype: Computation dtype for 4-bit models
            bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
            bnb_4bit_use_double_quant: Use nested quantization
            trust_remote_code: Trust remote code from model
            device_map: Device mapping strategy
            dtype: Default torch dtype
        """
        self.model_name = model_name
        self.quantization = quantization
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.dtype = dtype


class BaseModel(ABC):
    """Abstract base class for model handlers in RISCORE."""
    
    def __init__(self, config: BaseModelConfig, hf_token: Optional[str] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            hf_token: Hugging Face authentication token
        """
        self.config = config
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        
        if hf_token:
            self._authenticate()
        
        self._load_model()
        self._load_tokenizer()
    
    def _authenticate(self):
        """Authenticate with Hugging Face."""
        from huggingface_hub import login
        login(token=self.hf_token)
    
    def _create_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Create BitsAndBytes configuration for quantization."""
        if not self.config.quantization:
            return None
        
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None
    
    def _load_model(self):
        """Load the model with specified configuration."""
        bnb_config = self._create_bnb_config()
        
        model_kwargs = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self.config.dtype,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 700,
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling
            
        Returns:
            Generated text or list of texts
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.model_name,
            "quantization": self.config.quantization,
            "device": str(self.model.device) if self.model else "unknown",
            "dtype": str(self.config.dtype),
        }


class CausalLMModel(BaseModel):
    """Causal language model implementation."""
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 700,
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        """Generate text using causal language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove input prompt from output
            generated_ids = output[inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts[0] if num_return_sequences == 1 else generated_texts


class ChatModel(CausalLMModel):
    """Chat-based model with conversation templates."""
    
    def format_chat_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: str = "llama3"
    ) -> str:
        """
        Format prompt for chat models.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            model_type: Type of model ('llama3', 'llama2', 'mistral', etc.)
            
        Returns:
            Formatted prompt string
        """
        if model_type.lower() in ["llama3", "llama-3"]:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        elif model_type.lower() in ["llama2", "llama-2"]:
            return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
        elif model_type.lower() in ["mistral", "zephyr"]:
            return f"""<s>[INST] {system_prompt}

{user_prompt} [/INST]"""
        elif model_type.lower() in ["phi", "phi3", "phi-3"]:
            return f"""<|system|>
{system_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""
        elif model_type.lower() in ["gemma"]:
            return f"""<start_of_turn>user
{system_prompt}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        else:
            # Generic format
            return f"{system_prompt}\n\n{user_prompt}"
    
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: str = "llama3",
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate response for chat interaction.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            model_type: Type of model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response(s)
        """
        formatted_prompt = self.format_chat_prompt(system_prompt, user_prompt, model_type)
        return self.generate(formatted_prompt, **kwargs)
