"""Configuration management for RISCORE framework."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str
    model_type: str = "llama3"  # llama3, llama2, mistral, phi3, gemma, etc.
    quantization: bool = True
    load_in_4bit: bool = True
    temperature: float = 0.5
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_new_tokens: int = 700
    hf_token: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_type: str = "SP"  # SP or WP
    data_dir: str = "data"
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    val_file: Optional[str] = None


@dataclass
class PromptingConfig:
    """Prompting strategy configuration."""
    method: str = "riscore"  # riscore, fewshot, fewshot_cot, zeroshot, zeroshot_cot
    num_exemplars: int = 2
    use_cot: bool = True
    similarity_based_selection: bool = True
    similarity_threshold: float = 0.4
    use_self_consistency: bool = False
    num_consistency_paths: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    output_dir: str = "output"
    save_format: str = "json"  # json, csv, txt
    verbose: bool = True
    print_summary: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    model: ModelConfig
    dataset: DatasetConfig
    prompting: PromptingConfig
    evaluation: EvaluationConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "model": asdict(self.model),
            "dataset": asdict(self.dataset),
            "prompting": asdict(self.prompting),
            "evaluation": asdict(self.evaluation),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            experiment_name=data["experiment_name"],
            model=ModelConfig(**data["model"]),
            dataset=DatasetConfig(**data["dataset"]),
            prompting=PromptingConfig(**data["prompting"]),
            evaluation=EvaluationConfig(**data["evaluation"]),
        )
    
    def save(self, filepath: Path):
        """Save configuration to file."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(data)


class ConfigManager:
    """Manage experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory to store configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
    
    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None):
        """
        Save configuration.
        
        Args:
            config: Experiment configuration
            filename: Filename (auto-generated if None)
        """
        if filename is None:
            filename = f"{config.experiment_name}.json"
        
        filepath = self.config_dir / filename
        config.save(filepath)
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """
        Load configuration.
        
        Args:
            filename: Configuration filename
            
        Returns:
            Experiment configuration
        """
        filepath = self.config_dir / filename
        return ExperimentConfig.load(filepath)
    
    def list_configs(self) -> list:
        """List all available configurations."""
        return list(self.config_dir.glob("*.json")) + list(self.config_dir.glob("*.yaml"))
    
    @staticmethod
    def create_default_config(experiment_name: str = "default") -> ExperimentConfig:
        """
        Create default configuration.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Default experiment configuration
        """
        return ExperimentConfig(
            experiment_name=experiment_name,
            model=ModelConfig(
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                model_type="llama3",
            ),
            dataset=DatasetConfig(
                dataset_type="SP",
                data_dir="data",
            ),
            prompting=PromptingConfig(
                method="riscore",
                num_exemplars=2,
            ),
            evaluation=EvaluationConfig(
                output_dir="output",
            ),
        )
    
    @staticmethod
    def create_riscore_config(
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        dataset_type: str = "SP",
        num_exemplars: int = 2,
        experiment_name: str = "riscore_experiment"
    ) -> ExperimentConfig:
        """
        Create RISCORE-specific configuration.
        
        Args:
            model_name: Model identifier
            dataset_type: Dataset type (SP or WP)
            num_exemplars: Number of exemplars for few-shot
            experiment_name: Experiment name
            
        Returns:
            RISCORE experiment configuration
        """
        model_type = "llama3"
        if "llama-2" in model_name.lower():
            model_type = "llama2"
        elif "mistral" in model_name.lower():
            model_type = "mistral"
        elif "phi" in model_name.lower():
            model_type = "phi3"
        elif "gemma" in model_name.lower():
            model_type = "gemma"
        
        return ExperimentConfig(
            experiment_name=experiment_name,
            model=ModelConfig(
                model_name=model_name,
                model_type=model_type,
            ),
            dataset=DatasetConfig(
                dataset_type=dataset_type,
            ),
            prompting=PromptingConfig(
                method="riscore",
                num_exemplars=num_exemplars,
                use_cot=True,
                similarity_based_selection=True,
            ),
            evaluation=EvaluationConfig(
                output_dir="output",
                save_format="json",
            ),
        )
