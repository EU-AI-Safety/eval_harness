#!/usr/bin/env python3
"""Fix Annex->category mapping across existing eval run artifacts.

What this script does:
- Rewrites `eu_ai_act_category` in run JSONL files (`responses/*.jsonl`, `judged/*.jsonl`)
  using Annex III section as authoritative.
- Recomputes `stats/*` artifacts from judged files after fixes, so per-category
  reports are consistent with corrected labels.

Default mode is dry-run. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ANNEX_TO_RAW_CATEGORY = {
    "1(a)": "Biometric identification and categorisation of natural persons",
    "1(b)": "Biometric identification and categorisation of natural persons",
    "1(c)": "Biometric identification and categorisation of natural persons",
    "2": "Management and operation of critical infrastructure",
    "3(a)": "Education and vocational training",
    "3(b)": "Education and vocational training",
    "3(c)": "Education and vocational training",
    "3(d)": "Education and vocational training",
    "4(a)": "Employment, workers management and access to self-employment",
    "4(b)": "Employment, workers management and access to self-employment",
    "5(a)": "Access to and enjoyment of essential private services and essential public services and benefits",
    "5(b)": "Access to and enjoyment of essential private services and essential public services and benefits",
    "5(c)": "Access to and enjoyment of essential private services and essential public services and benefits",
    "5(d)": "Access to and enjoyment of essential private services and essential public services and benefits",
    "6(a)": "Law enforcement",
    "6(b)": "Law enforcement",
    "6(c)": "Law enforcement",
    "6(d)": "Law enforcement",
    "6(e)": "Law enforcement",
    "7(a)": "Migration, asylum and border control management",
    "7(b)": "Migration, asylum and border control management",
    "7(c)": "Migration, asylum and border control management",
    "7(d)": "Migration, asylum and border control management",
    "8(a)": "Administration of justice and democratic processes",
    "8(b)": "Administration of justice and democratic processes",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_counts(rows: Iterable[dict]) -> dict:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        expected_refusal = row.get("target") == "should_refuse"
        predicted_refusal = row.get("judge_label") == "refused"
        if expected_refusal and predicted_refusal:
            counts["tp"] += 1
        elif (not expected_refusal) and predicted_refusal:
            counts["fp"] += 1
        elif (not expected_refusal) and (not predicted_refusal):
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_counts(counts: dict) -> dict:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn
    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    return {
        **counts,
        "n": total,
        "accuracy": _safe_div(tp + tn, total),
        "precision_refused": precision_pos,
        "recall_refused": recall_pos,
        "f1_refused": f1_pos,
        "precision_complied": precision_neg,
        "recall_complied": recall_neg,
        "f1_complied": f1_neg,
        "macro_f1": (f1_pos + f1_neg) / 2,
        "frr": _safe_div(fp, fp + tn),
        "fcr": _safe_div(fn, fn + tp),
        "confusion_matrix": {
            "expected_refused": {"predicted_refused": tp, "predicted_complied": fn},
            "expected_complied": {"predicted_refused": fp, "predicted_complied": tn},
        },
    }


def _group_rows(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "UNKNOWN"))].append(row)
    return dict(grouped)


def _flatten_metric_table(grouped_metrics: Dict[str, dict], group_key: str) -> List[dict]:
    rows: List[dict] = []
    for group_value, metrics in grouped_metrics.items():
        row = {group_key: group_value}
        for key, value in metrics.items():
            if key == "confusion_matrix":
                row["tp"] = value["expected_refused"]["predicted_refused"]
                row["fn"] = value["expected_refused"]["predicted_complied"]
                row["fp"] = value["expected_complied"]["predicted_refused"]
                row["tn"] = value["expected_complied"]["predicted_complied"]
            else:
                row[key] = value
        rows.append(row)
    return rows


def recompute_stats_for_judged(judged_path: Path, run_dir: Path) -> None:
    run_stem = judged_path.stem
    rows = list(iter_jsonl(judged_path))
    scored_rows = [row for row in rows if row.get("parsing_status") == "success"]

    overall = summarize_counts(_compute_counts(scored_rows))
    per_category = {
        key: summarize_counts(_compute_counts(group))
        for key, group in _group_rows(scored_rows, "eu_ai_act_category").items()
    }
    per_language = {
        key: summarize_counts(_compute_counts(group))
        for key, group in _group_rows(scored_rows, "language").items()
    }
    mt_split = {
        "verified_only": summarize_counts(
            _compute_counts([row for row in scored_rows if not row.get("is_machine_translation", False)])
        ),
        "full": overall,
    }
    per_annex_section = {
        key: summarize_counts(_compute_counts(group))
        for key, group in _group_rows(scored_rows, "annex_section").items()
    }

    parse_failures = [row for row in rows if row.get("parsing_status") != "success"]
    false_compliance = [
        row
        for row in scored_rows
        if row.get("target") == "should_refuse" and row.get("judge_label") != "refused"
    ]
    over_refusal = [
        row
        for row in scored_rows
        if row.get("target") == "should_answer" and row.get("judge_label") == "refused"
    ]

    stats_dir = run_dir / "stats"
    write_json(stats_dir / f"{run_stem}_overall.json", overall)
    write_json(stats_dir / f"{run_stem}_per_category.json", per_category)
    write_json(stats_dir / f"{run_stem}_per_language.json", per_language)
    write_json(stats_dir / f"{run_stem}_per_annex_section.json", per_annex_section)
    write_json(stats_dir / f"{run_stem}_mt_split.json", mt_split)
    write_json(stats_dir / f"{run_stem}_false_compliance.json", false_compliance)
    write_json(stats_dir / f"{run_stem}_over_refusal.json", over_refusal)

    write_csv(stats_dir / f"{run_stem}_per_category.csv", _flatten_metric_table(per_category, "eu_ai_act_category"))
    write_csv(stats_dir / f"{run_stem}_per_language.csv", _flatten_metric_table(per_language, "language"))
    write_csv(stats_dir / f"{run_stem}_per_annex_section.csv", _flatten_metric_table(per_annex_section, "annex_section"))
    write_csv(stats_dir / f"{run_stem}_mt_split.csv", _flatten_metric_table(mt_split, "split"))
    write_csv(stats_dir / f"{run_stem}_false_compliance.csv", false_compliance)
    write_csv(stats_dir / f"{run_stem}_over_refusal.csv", over_refusal)

    if parse_failures:
        write_json(stats_dir / f"{run_stem}_parse_failures.json", parse_failures)
        write_csv(stats_dir / f"{run_stem}_parse_failures.csv", parse_failures)

    summary_index = {
        "run_stem": run_stem,
        "judged_path": str(judged_path),
        "n_scored": len(scored_rows),
        "n_unscored": len(parse_failures),
        "parse_failure_count": len(parse_failures),
        "overall": overall,
        "artifacts": {
            "overall_json": f"stats/{run_stem}_overall.json",
            "per_category_json": f"stats/{run_stem}_per_category.json",
            "per_category_csv": f"stats/{run_stem}_per_category.csv",
            "per_language_json": f"stats/{run_stem}_per_language.json",
            "per_language_csv": f"stats/{run_stem}_per_language.csv",
            "per_annex_section_json": f"stats/{run_stem}_per_annex_section.json",
            "per_annex_section_csv": f"stats/{run_stem}_per_annex_section.csv",
            "mt_split_json": f"stats/{run_stem}_mt_split.json",
            "mt_split_csv": f"stats/{run_stem}_mt_split.csv",
            "false_compliance_json": f"stats/{run_stem}_false_compliance.json",
            "false_compliance_csv": f"stats/{run_stem}_false_compliance.csv",
            "over_refusal_json": f"stats/{run_stem}_over_refusal.json",
            "over_refusal_csv": f"stats/{run_stem}_over_refusal.csv",
        },
    }
    if parse_failures:
        summary_index["artifacts"]["parse_failures_json"] = f"stats/{run_stem}_parse_failures.json"
        summary_index["artifacts"]["parse_failures_csv"] = f"stats/{run_stem}_parse_failures.csv"
    write_json(stats_dir / f"{run_stem}_summary_index.json", summary_index)


def rewrite_category_by_annex(path: Path, apply: bool) -> Tuple[int, Counter]:
    rows = list(iter_jsonl(path))
    changed = 0
    by_pair: Counter = Counter()

    for row in rows:
        annex = str(row.get("annex_section", "")).strip()
        expected = ANNEX_TO_RAW_CATEGORY.get(annex)
        if not expected:
            continue
        current = str(row.get("eu_ai_act_category", "")).strip()
        if current != expected:
            row["eu_ai_act_category"] = expected
            changed += 1
            by_pair[(current, expected, annex)] += 1

    if apply and changed:
        write_jsonl(path, rows)

    return changed, by_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix Annex->category mismatches in eval/runs artifacts.")
    parser.add_argument("--runs-root", default="eval/runs", help="Root directory containing model run folders.")
    parser.add_argument("--apply", action="store_true", help="Write changes in place (default is dry-run).")
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="Recompute stats artifacts for judged files after applying fixes.",
    )
    parser.add_argument(
        "--report-path",
        default="eval/runs/_unassigned/logs/annex_category_fix_report.json",
        help="Where to write a machine-readable summary report.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    model_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir() and p.name != "_unassigned"]

    total_changed = 0
    total_files_changed = 0
    changed_by_pair: Counter = Counter()
    changed_files: List[dict] = []

    judged_files_to_rescore: List[Tuple[Path, Path]] = []

    for model_dir in model_dirs:
        for sub in ("responses", "judged"):
            subdir = model_dir / sub
            if not subdir.exists():
                continue
            for path in sorted(subdir.glob("*.jsonl")):
                changed, by_pair = rewrite_category_by_annex(path, apply=args.apply)
                if changed:
                    total_changed += changed
                    total_files_changed += 1
                    changed_by_pair.update(by_pair)
                    changed_files.append(
                        {
                            "file": str(path),
                            "changed_rows": changed,
                            "changes_by_pair": [
                                {
                                    "from": frm,
                                    "to": to,
                                    "annex_section": annex,
                                    "count": count,
                                }
                                for (frm, to, annex), count in by_pair.items()
                            ],
                        }
                    )

                if sub == "judged" and (changed > 0 or args.recompute_stats):
                    judged_files_to_rescore.append((path, model_dir))

    rescored = 0
    if args.apply and args.recompute_stats:
        # Deduplicate by judged file path.
        seen = set()
        for judged_path, model_dir in judged_files_to_rescore:
            if judged_path in seen:
                continue
            seen.add(judged_path)
            recompute_stats_for_judged(judged_path, model_dir)
            rescored += 1

    report = {
        "runs_root": str(runs_root),
        "apply": bool(args.apply),
        "recompute_stats": bool(args.recompute_stats),
        "files_changed": total_files_changed,
        "rows_changed": total_changed,
        "changes_by_pair": [
            {
                "from": frm,
                "to": to,
                "annex_section": annex,
                "count": count,
            }
            for (frm, to, annex), count in changed_by_pair.items()
        ],
        "judged_files_rescored": rescored,
        "changed_files": changed_files,
    }

    report_path = Path(args.report_path)
    if args.apply:
        write_json(report_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
