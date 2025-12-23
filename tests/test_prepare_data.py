from __future__ import annotations

import zipfile

import numpy as np


def test_prepare_brainteaser_zip_normalizes_expected_filenames(tmp_path):
    from scripts.prepare_data import prepare_brainteaser_zip

    source = tmp_path / "source"
    source.mkdir()
    np.save(
        source / "SP_train.npy",
        np.array(
            [
                {
                    "question": "What gets wetter as it dries?",
                    "choice_list": ["A towel", "A candle", "A shadow", "A clock"],
                    "label": 0,
                }
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )
    np.save(
        source / "SP_test.npy",
        np.array(
            [
                {
                    "question": "What has keys but no locks?",
                    "choices": ["A piano", "A map", "A book", "A tree"],
                }
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )
    np.save(source / "SP_test_answer.npy", np.array([0]), allow_pickle=True)

    archive = tmp_path / "BTDATA.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in source.iterdir():
            zf.write(path, path.name)

    output_dir = tmp_path / "data"
    summary = prepare_brainteaser_zip(archive, output_dir)

    assert (output_dir / "SP-train.npy").exists()
    assert (output_dir / "SP_new_test.npy").exists()
    assert summary["SP_train.npy"] == "SP-train.npy"
    assert summary["SP_test.npy"] == "SP_new_test.npy"

    train = np.load(output_dir / "SP-train.npy", allow_pickle=True)
    first = train[0].item() if hasattr(train[0], "item") else train[0]
    assert first["answer"] == "A towel"
    assert first["answer_idx"] == 0


def test_normalize_riddlesense_record_preserves_five_choices_and_answer():
    from scripts.prepare_data import normalize_riddlesense_record

    record = {
        "answerKey": "E",
        "question": "What can be cracked but never held?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["stone", "egg", "voice", "code", "joke"],
        },
    }

    normalized = normalize_riddlesense_record(record, split="train", index=7)

    assert normalized["riddle_id"] == "RS_train_7"
    assert normalized["answer"] == "joke"
    assert normalized["answer_idx"] == 4
    assert normalized["choices"][-1] == "joke"
