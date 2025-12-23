# Data

The data module defines RISCORE's normalized riddle schema and loaders for local
dataset files.

## Main Classes

- `RiddleExample`: one multiple-choice riddle with optional reconstruction,
  reasoning, hints, and metadata.
- `RiddleDataset`: iterable collection with conversion helpers for NumPy, JSON,
  CSV, and Hugging Face Dataset objects.
- `DatasetLoader`: convenience methods for standard SP/WP train, test, and
  validation filenames.

## Expected Local Files

`DatasetLoader` looks under `data/` by default:

- `SP-train.npy`
- `SP_new_test.npy`
- `WP-train.npy`
- `WP_new_test.npy`
- `SP_val_question_random.npy` / `WP_val_question_random.npy` when available

Use `scripts/prepare_data.py` to normalize upstream BrainTeaser filenames into
this layout. See `docs/DATA.md`.

## Example

```python
from riscore.data import DatasetLoader

train = DatasetLoader.load_train_dataset("SP", "data")
test = DatasetLoader.load_test_dataset("SP", "data")

print(len(train), len(test))
print(train[0].question)
```
