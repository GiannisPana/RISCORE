# Data Setup

RISCORE expects local NumPy files under `data/`. Full benchmark files are not
committed to this repository.

## BrainTeaser

BrainTeaser is the primary source for the SP and WP riddle splits used by the
current CLI and examples.

- Upstream data directory: https://github.com/1171-jpg/BrainTeaser/tree/main/data
- Archive: `BTDATA.zip`
- Password: `brainteaser`

Prepare the local files:

```bash
python scripts/prepare_data.py --brainteaser-zip path\to\BTDATA.zip --output-dir data
```

The script normalizes the upstream filenames to the names expected by
`DatasetLoader`:

| Upstream file | RISCORE file |
| --- | --- |
| `SP_train.npy` | `SP-train.npy` |
| `SP_test.npy` + `SP_test_answer.npy` | `SP_new_test.npy` |
| `WP_train.npy` | `WP-train.npy` |
| `WP_test.npy` + `WP_test_answer.npy` | `WP_new_test.npy` |

The resulting examples use this schema:

```json
{
  "question": "riddle text",
  "choices": ["choice A", "choice B", "choice C", "choice D"],
  "answer": "choice text",
  "answer_idx": 0,
  "riddle_id": "SP_train_0",
  "category": "SP",
  "metadata": {"source": "BrainTeaser", "split": "train"}
}
```

## RiddleSense

RiddleSense is available through Hugging Face:

- Dataset: https://huggingface.co/datasets/INK-USC/riddle_sense
- Identifier: `INK-USC/riddle_sense`

Prepare normalized optional splits:

```bash
python scripts/prepare_data.py --riddlesense-hf --output-dir data
```

RiddleSense examples use five answer choices. The current RISCORE CLI and paper
workflow are centered on four-choice BrainTeaser SP/WP data, so treat
RiddleSense as an extension unless you also adapt answer extraction and
five-choice evaluation.

## What Not To Commit

Do not commit:

- `data/*.npy`
- downloaded archives such as `BTDATA.zip`
- generated reconstructions
- `output/` result files
- model or embedding caches

These paths are ignored by `.gitignore`.
