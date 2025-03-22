"""Dataset handling for RISCORE framework."""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset
import json


@dataclass
class RiddleExample:
    """Single riddle example."""
    question: str
    choices: List[str]
    answer: str
    answer_idx: int
    riddle_id: Optional[str] = None
    category: Optional[str] = None
    reconstructed_question: Optional[str] = None
    reconstructed_choices: Optional[List[str]] = None
    reconstructed_answer: Optional[str] = None
    cot: Optional[str] = None
    hint: Optional[str] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "choices": self.choices,
            "answer": self.answer,
            "answer_idx": self.answer_idx,
            "riddle_id": self.riddle_id,
            "category": self.category,
            "reconstructed_question": self.reconstructed_question,
            "reconstructed_choices": self.reconstructed_choices,
            "reconstructed_answer": self.reconstructed_answer,
            "cot": self.cot,
            "hint": self.hint,
            "explanation": self.explanation,
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiddleExample":
        """Create from dictionary."""
        return cls(**data)


class RiddleDataset:
    """Dataset handler for riddles."""
    
    def __init__(
        self,
        examples: Optional[List[RiddleExample]] = None,
        dataset_type: str = "SP",
    ):
        """
        Initialize riddle dataset.
        
        Args:
            examples: List of riddle examples
            dataset_type: Type of dataset ('SP' or 'WP')
        """
        self.examples = examples or []
        self.dataset_type = dataset_type
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> RiddleExample:
        """Get example by index."""
        return self.examples[idx]
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)
    
    @classmethod
    def from_numpy(cls, file_path: Union[str, Path], dataset_type: str = "SP") -> "RiddleDataset":
        """
        Load dataset from numpy file.
        
        Args:
            file_path: Path to .npy file
            dataset_type: Type of dataset
            
        Returns:
            RiddleDataset instance
        """
        data = np.load(file_path, allow_pickle=True)
        examples = []
        
        for item in data:
            # Handle different numpy array structures
            if isinstance(item, dict):
                example_dict = item
            else:
                # Try to convert numpy array to dict
                example_dict = {
                    "question": item[0] if len(item) > 0 else "",
                    "choices": item[1] if len(item) > 1 else [],
                    "answer": item[2] if len(item) > 2 else "",
                    "answer_idx": item[3] if len(item) > 3 else 0,
                }
            
            # Create RiddleExample with available fields
            example = RiddleExample(
                question=example_dict.get("question", ""),
                choices=example_dict.get("choices", []),
                answer=example_dict.get("answer", ""),
                answer_idx=example_dict.get("answer_idx", 0),
                riddle_id=example_dict.get("riddle_id"),
                category=example_dict.get("category"),
                reconstructed_question=example_dict.get("reconstructed_question"),
                reconstructed_choices=example_dict.get("reconstructed_choices"),
                reconstructed_answer=example_dict.get("reconstructed_answer"),
                cot=example_dict.get("cot"),
                hint=example_dict.get("hint"),
                explanation=example_dict.get("explanation"),
                metadata=example_dict.get("metadata"),
            )
            examples.append(example)
        
        return cls(examples=examples, dataset_type=dataset_type)
    
    @classmethod
    def from_csv(cls, file_path: Union[str, Path], dataset_type: str = "SP") -> "RiddleDataset":
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            dataset_type: Type of dataset
            
        Returns:
            RiddleDataset instance
        """
        df = pd.read_csv(file_path)
        examples = []
        
        for _, row in df.iterrows():
            # Extract choices (assuming format: choice_1, choice_2, etc.)
            choices = []
            for i in range(1, 5):  # Assuming max 4 choices
                choice_key = f"choice_{i}"
                if choice_key in row and pd.notna(row[choice_key]):
                    choices.append(row[choice_key])
            
            example = RiddleExample(
                question=row.get("question", ""),
                choices=choices,
                answer=row.get("answer", ""),
                answer_idx=int(row.get("answer_idx", 0)) if "answer_idx" in row else 0,
                riddle_id=row.get("riddle_id"),
                category=row.get("category"),
                reconstructed_question=row.get("reconstructed_question"),
                cot=row.get("cot"),
                hint=row.get("hint"),
                explanation=row.get("explanation"),
            )
            examples.append(example)
        
        return cls(examples=examples, dataset_type=dataset_type)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path], dataset_type: str = "SP") -> "RiddleDataset":
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            dataset_type: Type of dataset
            
        Returns:
            RiddleDataset instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = [RiddleExample.from_dict(item) for item in data]
        return cls(examples=examples, dataset_type=dataset_type)
    
    def to_numpy(self, file_path: Union[str, Path]):
        """Save dataset to numpy file."""
        data = [example.to_dict() for example in self.examples]
        np.save(file_path, data, allow_pickle=True)
    
    def to_csv(self, file_path: Union[str, Path]):
        """Save dataset to CSV file."""
        data = []
        for example in self.examples:
            row = {
                "question": example.question,
                "answer": example.answer,
                "answer_idx": example.answer_idx,
                "riddle_id": example.riddle_id,
                "category": example.category,
                "reconstructed_question": example.reconstructed_question,
                "cot": example.cot,
                "hint": example.hint,
                "explanation": example.explanation,
            }
            # Add choices
            for i, choice in enumerate(example.choices, 1):
                row[f"choice_{i}"] = choice
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def to_json(self, file_path: Union[str, Path]):
        """Save dataset to JSON file."""
        data = [example.to_dict() for example in self.examples]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def to_hf_dataset(self) -> Dataset:
        """Convert to Hugging Face Dataset."""
        data = [example.to_dict() for example in self.examples]
        return Dataset.from_list(data)
    
    def filter(self, condition) -> "RiddleDataset":
        """
        Filter examples by condition.
        
        Args:
            condition: Function that takes RiddleExample and returns bool
            
        Returns:
            New filtered RiddleDataset
        """
        filtered_examples = [ex for ex in self.examples if condition(ex)]
        return RiddleDataset(examples=filtered_examples, dataset_type=self.dataset_type)
    
    def split(self, train_size: float = 0.8, shuffle: bool = True, seed: int = 42) -> Tuple["RiddleDataset", "RiddleDataset"]:
        """
        Split dataset into train and test.
        
        Args:
            train_size: Proportion for training
            shuffle: Whether to shuffle before splitting
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random
        
        examples = self.examples.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(examples)
        
        split_idx = int(len(examples) * train_size)
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]
        
        return (
            RiddleDataset(examples=train_examples, dataset_type=self.dataset_type),
            RiddleDataset(examples=test_examples, dataset_type=self.dataset_type),
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_examples": len(self.examples),
            "dataset_type": self.dataset_type,
            "has_reconstructed": sum(1 for ex in self.examples if ex.reconstructed_question is not None),
            "has_cot": sum(1 for ex in self.examples if ex.cot is not None),
            "has_hint": sum(1 for ex in self.examples if ex.hint is not None),
            "has_explanation": sum(1 for ex in self.examples if ex.explanation is not None),
            "has_category": sum(1 for ex in self.examples if ex.category is not None),
        }
        
        # Category distribution
        if any(ex.category for ex in self.examples):
            from collections import Counter
            categories = [ex.category for ex in self.examples if ex.category]
            stats["category_distribution"] = dict(Counter(categories))
        
        return stats


class DatasetLoader:
    """Utility class for loading various datasets."""
    
    @staticmethod
    def load_dataset(
        file_path: Union[str, Path],
        dataset_type: str = "SP",
        file_format: Optional[str] = None,
    ) -> RiddleDataset:
        """
        Load dataset from file (auto-detect format).
        
        Args:
            file_path: Path to dataset file
            dataset_type: Type of dataset ('SP' or 'WP')
            file_format: File format ('numpy', 'csv', 'json') or None for auto-detect
            
        Returns:
            RiddleDataset instance
        """
        file_path = Path(file_path)
        
        if file_format is None:
            # Auto-detect format
            if file_path.suffix == ".npy":
                file_format = "numpy"
            elif file_path.suffix == ".csv":
                file_format = "csv"
            elif file_path.suffix == ".json":
                file_format = "json"
            else:
                raise ValueError(f"Unknown file format: {file_path.suffix}")
        
        if file_format == "numpy":
            return RiddleDataset.from_numpy(file_path, dataset_type)
        elif file_format == "csv":
            return RiddleDataset.from_csv(file_path, dataset_type)
        elif file_format == "json":
            return RiddleDataset.from_json(file_path, dataset_type)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    @staticmethod
    def load_train_dataset(dataset_type: str = "SP", data_dir: Union[str, Path] = "data") -> RiddleDataset:
        """Load training dataset."""
        data_dir = Path(data_dir)
        file_name = f"{dataset_type}-train.npy"
        return DatasetLoader.load_dataset(data_dir / file_name, dataset_type)
    
    @staticmethod
    def load_test_dataset(dataset_type: str = "SP", data_dir: Union[str, Path] = "data") -> RiddleDataset:
        """Load test dataset."""
        data_dir = Path(data_dir)
        file_name = f"{dataset_type}_new_test.npy"
        return DatasetLoader.load_dataset(data_dir / file_name, dataset_type)
    
    @staticmethod
    def load_validation_dataset(dataset_type: str = "SP", data_dir: Union[str, Path] = "data") -> RiddleDataset:
        """Load validation dataset."""
        data_dir = Path(data_dir)
        file_name = f"{dataset_type}_val_question_random.npy"
        return DatasetLoader.load_dataset(data_dir / file_name, dataset_type)
