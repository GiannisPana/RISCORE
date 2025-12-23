"""Prepare upstream riddle datasets for RISCORE.

This script normalizes external data into the local ``data/`` layout expected by
``DatasetLoader``. Full benchmark data is intentionally not committed to this
repository; download it from the upstream sources listed in ``docs/DATA.md``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


BRAINTEASER_FILE_MAP = {
    "SP_train.npy": ("SP", "train", "SP-train.npy", None),
    "WP_train.npy": ("WP", "train", "WP-train.npy", None),
    "SP_test.npy": ("SP", "test", "SP_new_test.npy", "SP_test_answer.npy"),
    "WP_test.npy": ("WP", "test", "WP_new_test.npy", "WP_test_answer.npy"),
}


def _to_python(value: Any) -> Any:
    """Convert common NumPy scalar/object wrappers to plain Python values."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return _to_python(value.item())
    if hasattr(value, "item") and not isinstance(value, (dict, list, tuple, str, bytes)):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _lookup(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _choice_texts(raw_choices: Any) -> List[str]:
    raw_choices = _to_python(raw_choices)
    if raw_choices is None:
        return []
    if isinstance(raw_choices, Mapping):
        if "text" in raw_choices:
            return [str(choice) for choice in raw_choices["text"]]
        return [str(raw_choices[key]) for key in sorted(raw_choices)]
    if isinstance(raw_choices, np.ndarray):
        raw_choices = raw_choices.tolist()
    if isinstance(raw_choices, (list, tuple)):
        return [str(_to_python(choice)) for choice in raw_choices]
    return [str(raw_choices)]


def _answer_to_index(answer: Any, choices: List[str]) -> int:
    answer = _to_python(answer)
    if answer is None:
        raise ValueError("Missing answer for riddle record")

    if isinstance(answer, str):
        stripped = answer.strip()
        upper = stripped.upper()
        if len(upper) == 1 and "A" <= upper <= "Z":
            idx = ord(upper) - ord("A")
            if 0 <= idx < len(choices):
                return idx
        if stripped.isdigit():
            numeric = int(stripped)
            if 0 <= numeric < len(choices):
                return numeric
            if 1 <= numeric <= len(choices):
                return numeric - 1
        for idx, choice in enumerate(choices):
            if stripped.casefold() == choice.casefold():
                return idx

    if isinstance(answer, (int, np.integer)):
        numeric = int(answer)
        if 0 <= numeric < len(choices):
            return numeric
        if 1 <= numeric <= len(choices):
            return numeric - 1

    raise ValueError(f"Could not map answer {answer!r} to choices {choices!r}")


def normalize_brainteaser_record(
    record: Any,
    answer_override: Any = None,
    dataset_type: str = "SP",
    split: str = "train",
    index: int = 0,
) -> Dict[str, Any]:
    """Normalize one BrainTeaser record to the RISCORE schema."""
    record = _to_python(record)
    if not isinstance(record, Mapping):
        if isinstance(record, (list, tuple)) and len(record) >= 2:
            record = {
                "question": record[0],
                "choices": record[1],
                "answer": record[2] if len(record) > 2 else answer_override,
            }
        else:
            raise ValueError(f"Unsupported BrainTeaser record: {record!r}")

    question = str(_lookup(record, "question", "Question", "sent", "sentence") or "").strip()
    choices = _choice_texts(_lookup(record, "choices", "choice_list", "options", "answers"))
    answer_source = answer_override
    if answer_source is None:
        answer_source = _lookup(record, "answer_idx", "label", "answerKey", "answer")
    answer_idx = _answer_to_index(answer_source, choices)

    return {
        "question": question,
        "choices": choices,
        "answer": choices[answer_idx],
        "answer_idx": answer_idx,
        "riddle_id": str(_lookup(record, "riddle_id", "id") or f"{dataset_type}_{split}_{index}"),
        "category": dataset_type,
        "metadata": {"source": "BrainTeaser", "split": split},
    }


def normalize_riddlesense_record(record: Mapping[str, Any], split: str, index: int) -> Dict[str, Any]:
    """Normalize one RiddleSense record to the RISCORE schema."""
    choices = _choice_texts(record.get("choices"))
    answer_idx = _answer_to_index(record.get("answerKey"), choices)
    return {
        "question": str(record.get("question", "")).strip(),
        "choices": choices,
        "answer": choices[answer_idx],
        "answer_idx": answer_idx,
        "riddle_id": f"RS_{split}_{index}",
        "category": "RS",
        "metadata": {"source": "RiddleSense", "split": split},
    }


def _load_npy(path: Path) -> List[Any]:
    data = np.load(path, allow_pickle=True)
    return [_to_python(item) for item in data]


def _find_file(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def _write_examples(path: Path, examples: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(list(examples), dtype=object), allow_pickle=True)


def prepare_brainteaser_zip(
    zip_path: Path | str,
    output_dir: Path | str = "data",
    password: str = "brainteaser",
) -> Dict[str, str]:
    """Extract and normalize BrainTeaser ``BTDATA.zip`` into RISCORE filenames."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    summary: Dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir, pwd=password.encode("utf-8") if password else None)

        for source_name, (dataset_type, split, target_name, answer_name) in BRAINTEASER_FILE_MAP.items():
            source_path = _find_file(extract_dir, source_name)
            if source_path is None:
                continue

            answers = None
            if answer_name:
                answer_path = _find_file(extract_dir, answer_name)
                if answer_path is not None:
                    answers = _load_npy(answer_path)

            examples = []
            for index, record in enumerate(_load_npy(source_path)):
                answer_override = answers[index] if answers is not None and index < len(answers) else None
                examples.append(
                    normalize_brainteaser_record(
                        record,
                        answer_override=answer_override,
                        dataset_type=dataset_type,
                        split=split,
                        index=index,
                    )
                )

            _write_examples(output_dir / target_name, examples)
            summary[source_name] = target_name

    return summary


def prepare_riddlesense_jsonl(jsonl_path: Path | str, output_file: Path | str, split: str) -> int:
    """Normalize a local RiddleSense JSONL split into a RISCORE ``.npy`` file."""
    jsonl_path = Path(jsonl_path)
    output_file = Path(output_file)
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                examples.append(normalize_riddlesense_record(json.loads(line), split=split, index=index))
    _write_examples(output_file, examples)
    return len(examples)


def prepare_riddlesense_hf(output_dir: Path | str = "data") -> Dict[str, str]:
    """Download RiddleSense from Hugging Face and save normalized splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to download INK-USC/riddle_sense from Hugging Face") from exc

    output_dir = Path(output_dir)
    dataset = load_dataset("INK-USC/riddle_sense", trust_remote_code=True)
    names = {"train": "RS-train.npy", "validation": "RS_val.npy", "test": "RS_new_test.npy"}
    summary: Dict[str, str] = {}
    for split, target_name in names.items():
        if split not in dataset:
            continue
        examples = [
            normalize_riddlesense_record(record, split=split, index=index)
            for index, record in enumerate(dataset[split])
        ]
        _write_examples(output_dir / target_name, examples)
        summary[split] = target_name
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RISCORE datasets from upstream sources.")
    parser.add_argument("--brainteaser-zip", type=Path, help="Path to BrainTeaser BTDATA.zip")
    parser.add_argument("--brainteaser-password", default="brainteaser", help="Password for BTDATA.zip")
    parser.add_argument("--riddlesense-jsonl", type=Path, help="Path to a local RiddleSense JSONL split")
    parser.add_argument("--riddlesense-split", default="train", help="Split name for --riddlesense-jsonl")
    parser.add_argument("--riddlesense-hf", action="store_true", help="Download INK-USC/riddle_sense")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory")
    args = parser.parse_args()

    if args.brainteaser_zip:
        summary = prepare_brainteaser_zip(args.brainteaser_zip, args.output_dir, args.brainteaser_password)
        for source, target in summary.items():
            print(f"{source} -> {args.output_dir / target}")

    if args.riddlesense_jsonl:
        target = args.output_dir / f"RS_{args.riddlesense_split}.npy"
        count = prepare_riddlesense_jsonl(args.riddlesense_jsonl, target, args.riddlesense_split)
        print(f"{args.riddlesense_jsonl} -> {target} ({count} examples)")

    if args.riddlesense_hf:
        summary = prepare_riddlesense_hf(args.output_dir)
        for source, target in summary.items():
            print(f"INK-USC/riddle_sense:{source} -> {args.output_dir / target}")

    if not (args.brainteaser_zip or args.riddlesense_jsonl or args.riddlesense_hf):
        parser.print_help()


if __name__ == "__main__":
    main()
