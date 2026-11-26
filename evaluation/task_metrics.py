"""Task-level evaluation metrics (MMAU / classification / captioning)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

MMAU_TASK_LABELS = ("speech", "sound", "music")


def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\(([A-Z])\)", text.strip().upper())
    if match:
        return match.group(1)
    stripped = text.strip().upper()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped
    return None


def resolve_csv_fields(fieldnames: list[str] | None) -> tuple[str, str, str]:
    fields = set(fieldnames or [])
    if "task" in fields:
        return "task", "ground_truth", "generated_text"
    return "audio_path", "K", "answer_token_count"


def evaluate_choice_csv(csv_path: Path | str) -> Dict[str, Any]:
    """Accuracy on multiple-choice CSV (MMAU / similar letter-choice formats)."""
    csv_path = Path(csv_path)
    total = 0
    matched = 0
    correct = 0
    unknown_prediction = 0
    missing_gt = 0
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        task_field, gt_field, prediction_field = resolve_csv_fields(reader.fieldnames)
        raw_rows = list(reader)

    seen: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        key = row.get("audio_key") or row.get("sample_id") or str(len(seen))
        seen[key] = row
    rows = list(seen.values())

    for row in rows:
        total += 1
        gt_raw = row.get(gt_field, "")
        generated_text = row.get(prediction_field, "")
        task_name = (row.get(task_field, "") or "").strip().lower()
        gt = extract_choice_letter(gt_raw)
        if gt is None:
            missing_gt += 1
            continue
        matched += 1
        pred = extract_choice_letter(generated_text)
        if pred is None:
            unknown_prediction += 1
            per_class_total[gt] += 1
            confusion[gt]["<unknown>"] += 1
            if task_name in MMAU_TASK_LABELS:
                per_task_total[task_name] += 1
            continue
        per_class_total[gt] += 1
        confusion[gt][pred] += 1
        if task_name in MMAU_TASK_LABELS:
            per_task_total[task_name] += 1
        if pred == gt:
            correct += 1
            per_class_correct[gt] += 1
            if task_name in MMAU_TASK_LABELS:
                per_task_correct[task_name] += 1

    labels = sorted(per_class_total.keys())
    return {
        "csv_rows": len(raw_rows),
        "deduped_rows": len(rows),
        "matched": matched,
        "missing_gt": missing_gt,
        "unknown_prediction": unknown_prediction,
        "correct": correct,
        "accuracy": (correct / matched) if matched else 0.0,
        "per_task_accuracy": {
            task: {
                "accuracy": (per_task_correct[task] / per_task_total[task])
                if per_task_total[task]
                else 0.0,
                "correct": per_task_correct[task],
                "total": per_task_total[task],
            }
            for task in MMAU_TASK_LABELS
        },
        "per_class_accuracy": {
            label: {
                "accuracy": (per_class_correct[label] / per_class_total[label])
                if per_class_total[label]
                else 0.0,
                "correct": per_class_correct[label],
                "total": per_class_total[label],
            }
            for label in labels
        },
        "confusion": {gt: dict(confusion[gt]) for gt in labels},
    }


def save_json(results: Dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")


# Back-compat alias
evaluate_mmau_csv = evaluate_choice_csv
