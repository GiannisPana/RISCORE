# Evaluation

The evaluation module runs model predictions, extracts answer letters, computes
metrics, and saves result artifacts.

## Main Classes

- `PredictionResult`: per-riddle prediction record.
- `EvaluationMetrics`: accuracy, answer distribution, confusion matrix, category
  accuracy, and response-time helpers.
- `ResultsManager`: stores results, metadata, and JSON/CSV/TXT outputs.
- `Evaluator`: loops over a dataset, formats prompts, calls the model, and
  records predictions.

## Example

```python
from riscore.evaluation import Evaluator, ResultsManager
from riscore.prompting import RISCOREPrompt

strategy = RISCOREPrompt(num_exemplars=2)
manager = ResultsManager(output_dir="output")
evaluator = Evaluator(model=model, prompt_strategy=strategy, results_manager=manager)

evaluator.evaluate(test_dataset, exemplars=train_dataset.examples)
manager.print_summary()
manager.save_results(format="json")
```

The default metrics assume four answer choices, matching BrainTeaser SP/WP.
RiddleSense has five choices and should be treated as an extension unless the
answer extraction and metrics are adapted.
