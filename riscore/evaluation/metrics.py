"""Evaluation metrics and result tracking for RISCORE framework."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict


@dataclass
class PredictionResult:
    """Single prediction result."""
    riddle_id: str
    question: str
    true_answer: str
    true_answer_idx: int
    predicted_answer: str
    predicted_answer_idx: int
    is_correct: bool
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "riddle_id": self.riddle_id,
            "question": self.question,
            "true_answer": self.true_answer,
            "true_answer_idx": self.true_answer_idx,
            "predicted_answer": self.predicted_answer,
            "predicted_answer_idx": self.predicted_answer_idx,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "response_time": self.response_time,
            "metadata": self.metadata,
        }


class EvaluationMetrics:
    """Calculate evaluation metrics for riddle-solving."""
    
    @staticmethod
    def accuracy(results: List[PredictionResult]) -> float:
        """Calculate accuracy."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.is_correct)
        return correct / len(results)
    
    @staticmethod
    def accuracy_by_category(
        results: List[PredictionResult],
        categories: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate accuracy per category.
        
        Args:
            results: List of prediction results
            categories: Mapping from riddle_id to category
            
        Returns:
            Dictionary of category -> accuracy
        """
        category_results = defaultdict(list)
        
        for result in results:
            category = categories.get(result.riddle_id, "unknown")
            category_results[category].append(result.is_correct)
        
        return {
            cat: sum(correct) / len(correct) if correct else 0.0
            for cat, correct in category_results.items()
        }
    
    @staticmethod
    def confusion_matrix(results: List[PredictionResult], num_classes: int = 4) -> List[List[int]]:
        """
        Calculate confusion matrix.
        
        Args:
            results: List of prediction results
            num_classes: Number of answer choices (default 4 for A, B, C, D)
            
        Returns:
            Confusion matrix as 2D list
        """
        matrix = [[0] * num_classes for _ in range(num_classes)]
        
        for result in results:
            true_idx = result.true_answer_idx
            pred_idx = result.predicted_answer_idx
            
            if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
                matrix[true_idx][pred_idx] += 1
        
        return matrix
    
    @staticmethod
    def answer_distribution(results: List[PredictionResult]) -> Dict[str, int]:
        """Get distribution of predicted answers."""
        predictions = [r.predicted_answer for r in results]
        return dict(Counter(predictions))
    
    @staticmethod
    def average_response_time(results: List[PredictionResult]) -> Optional[float]:
        """Calculate average response time."""
        times = [r.response_time for r in results if r.response_time is not None]
        return sum(times) / len(times) if times else None
    
    @staticmethod
    def calculate_all_metrics(results: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate all available metrics."""
        metrics = {
            "total_examples": len(results),
            "accuracy": EvaluationMetrics.accuracy(results),
            "correct_predictions": sum(1 for r in results if r.is_correct),
            "incorrect_predictions": sum(1 for r in results if not r.is_correct),
            "answer_distribution": EvaluationMetrics.answer_distribution(results),
            "confusion_matrix": EvaluationMetrics.confusion_matrix(results),
        }
        
        avg_time = EvaluationMetrics.average_response_time(results)
        if avg_time is not None:
            metrics["average_response_time"] = avg_time
        
        return metrics


class ResultsManager:
    """Manage evaluation results and outputs."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize results manager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.results: List[PredictionResult] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, result: PredictionResult):
        """Add a prediction result."""
        self.results.append(result)
    
    def add_results(self, results: List[PredictionResult]):
        """Add multiple prediction results."""
        self.results.extend(results)
    
    def set_metadata(self, **kwargs):
        """Set metadata for the evaluation run."""
        self.metadata.update(kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for current results."""
        return EvaluationMetrics.calculate_all_metrics(self.results)
    
    def save_results(
        self,
        filename: Optional[str] = None,
        format: str = "json"
    ) -> Path:
        """
        Save results to file.
        
        Args:
            filename: Output filename (auto-generated if None)
            format: Output format ('json', 'csv', or 'txt')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.metadata.get("model_name", "unknown").split("/")[-1]
            method = self.metadata.get("method", "unknown")
            dataset = self.metadata.get("dataset_type", "unknown")
            filename = f"{method}_{dataset}_{model_name}_{timestamp}"
        
        if format == "json":
            return self._save_json(filename)
        elif format == "csv":
            return self._save_csv(filename)
        elif format == "txt":
            return self._save_txt(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, filename: str) -> Path:
        """Save results as JSON."""
        filepath = self.output_dir / f"{filename}.json"
        
        output = {
            "metadata": self.metadata,
            "metrics": self.get_metrics(),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_csv(self, filename: str) -> Path:
        """Save results as CSV."""
        filepath = self.output_dir / f"{filename}.csv"
        
        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def _save_txt(self, filename: str) -> Path:
        """Save results as text."""
        filepath = self.output_dir / f"{filename}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write("=" * 80 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("METADATA:\n")
            for key, value in self.metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Write metrics
            f.write("METRICS:\n")
            metrics = self.get_metrics()
            for key, value in metrics.items():
                if key not in ["confusion_matrix", "answer_distribution"]:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Write answer distribution
            if "answer_distribution" in metrics:
                f.write("ANSWER DISTRIBUTION:\n")
                for answer, count in metrics["answer_distribution"].items():
                    f.write(f"  {answer}: {count}\n")
                f.write("\n")
            
            # Write individual results
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("-" * 80 + "\n")
            for i, result in enumerate(self.results, 1):
                f.write(f"\n[{i}] {result.riddle_id}\n")
                f.write(f"Question: {result.question}\n")
                f.write(f"True Answer: {result.true_answer}\n")
                f.write(f"Predicted: {result.predicted_answer}\n")
                f.write(f"Correct: {result.is_correct}\n")
                if result.reasoning:
                    f.write(f"Reasoning: {result.reasoning}\n")
                f.write("-" * 80 + "\n")
        
        return filepath
    
    def print_summary(self):
        """Print summary of results to console."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Correct: {metrics['correct_predictions']}")
        print(f"Incorrect: {metrics['incorrect_predictions']}")
        
        if "average_response_time" in metrics:
            print(f"Avg Response Time: {metrics['average_response_time']:.2f}s")
        
        print("\nAnswer Distribution:")
        for answer, count in metrics["answer_distribution"].items():
            print(f"  {answer}: {count}")
        print("=" * 80 + "\n")


class Evaluator:
    """Main evaluator class for running evaluations."""
    
    def __init__(
        self,
        model,
        prompt_strategy,
        results_manager: Optional[ResultsManager] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Language model instance
            prompt_strategy: Prompting strategy to use
            results_manager: Results manager (creates new if None)
        """
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.results_manager = results_manager or ResultsManager()
    
    def evaluate(
        self,
        dataset,
        exemplars: Optional[List] = None,
        verbose: bool = True,
        **generation_kwargs
    ) -> List[PredictionResult]:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: RiddleDataset to evaluate on
            exemplars: Exemplars for few-shot learning
            verbose: Whether to show progress
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of prediction results
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(dataset, desc="Evaluating") if verbose else dataset
        
        for example in iterator:
            # Format prompt
            if exemplars:
                prompt = self.prompt_strategy.format_prompt(example, exemplars=exemplars)
            else:
                prompt = self.prompt_strategy.format_prompt(example)
            
            # Generate response
            import time
            start_time = time.time()
            
            if hasattr(self.model, 'generate_chat'):
                response = self.model.generate_chat(
                    system_prompt=self.prompt_strategy.system_prompt,
                    user_prompt=prompt,
                    **generation_kwargs
                )
            else:
                response = self.model.generate(prompt, **generation_kwargs)
            
            response_time = time.time() - start_time
            
            # Extract answer
            predicted_answer = self.prompt_strategy.extract_answer_from_response(response)
            predicted_idx = ord(predicted_answer) - 65 if predicted_answer else -1
            
            # Create result
            result = PredictionResult(
                riddle_id=example.riddle_id or f"example_{len(results)}",
                question=example.question,
                true_answer=chr(65 + example.answer_idx),
                true_answer_idx=example.answer_idx,
                predicted_answer=predicted_answer,
                predicted_answer_idx=predicted_idx,
                is_correct=(predicted_idx == example.answer_idx),
                reasoning=response,
                response_time=response_time,
            )
            
            results.append(result)
            self.results_manager.add_result(result)
        
        return results
