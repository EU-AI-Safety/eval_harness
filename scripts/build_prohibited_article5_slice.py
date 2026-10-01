#!/usr/bin/env python3
"""Build the Article 5 prohibited eval slice from the handover CSV.

Output matches eval_harness dataset rows:
  prompt, target, language, eu_ai_act_category, annex_section,
  reasoning, reasoning2, is_machine_translation

Language order follows the EU-Alert test set. This handover has one
translation column per language (no separate verified German/Chinese/
Polish/Italian text), so only English is marked is_machine_translation=false.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    ROOT.parent
    / "EU-Alert_prohibited categories_to_handover_to_eval_team - prohibited.csv"
)
DEFAULT_JSONL = ROOT / "eval" / "slices" / "prohibited_article5.jsonl"
DEFAULT_MANIFEST = ROOT / "eval" / "slices" / "prohibited_article5.manifest.json"

# Same order as EU_alert test.jsonl parallel blocks, with a single Chinese
# row (the handover has translation_chinese only, not a verified Chinese).
LANGUAGES: list[tuple[str, str | None]] = [
    ("english", None),
    ("german", "translation_german"),
    ("chinese", "translation_chinese"),
    ("polish", "translation_polish"),
    ("italian", "translation_italian"),
    ("bulgarian", "translation_bulgarian"),
    ("croatian", "translation_croatian"),
    ("czech", "translation_czech"),
    ("danish", "translation_danish"),
    ("dutch", "translation_dutch"),
    ("estonian", "translation_estonian"),
    ("finnish", "translation_finnish"),
    ("french", "translation_french"),
    ("greek", "translation_greek"),
    ("hungarian", "translation_hungarian"),
    ("irish", "translation_irish"),
    ("latvian", "translation_latvian"),
    ("lithuanian", "translation_lithuanian"),
    ("maltese", "translation_maltese"),
    ("portuguese", "translation_portuguese"),
    ("romanian", "translation_romanian"),
    ("slovak", "translation_slovak"),
    ("slovenian", "translation_slovenian"),
    ("spanish", "translation_spanish"),
    ("swedish", "translation_swedish"),
    ("arabic", "translation_arabic"),
    ("catalan", "translation_catalan"),
    ("galician", "translation_galician"),
    ("hindi", "translation_hindi"),
    ("japanese", "translation_japanese"),
    ("korean", "translation_korean"),
    ("norwegian", "translation_norwegian"),
    ("russian", "translation_russian"),
    ("turkish", "translation_turkish"),
    ("ukrainian", "translation_ukrainian"),
]


def _cell(value: str | None) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build(csv_path: Path, jsonl_path: Path, manifest_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    missing_cols = [
        column
        for _language, column in LANGUAGES
        if column is not None and column not in rows[0]
    ]
    if missing_cols:
        raise SystemExit(f"CSV missing translation columns: {missing_cols}")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    targets: Counter[str] = Counter()
    sections: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    n_written = 0
    empty_translations: list[str] = []

    with jsonl_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            target = _cell(row.get("Annotator 1"))
            if target not in {"should_refuse", "should_answer"}:
                raise SystemExit(
                    f"Row {index} Annotator 1 target {target!r} is not should_refuse/should_answer"
                )
            category = _cell(row.get("Prohibition_category"))
            section = _cell(row.get("section_number"))
            reasoning = _cell(row.get("Reasoning Annotator 1"))
            reasoning2 = _cell(row.get("Reasoning Annotator 6"))
            if not category or not section:
                raise SystemExit(f"Row {index} missing category or section")

            targets[target] += 1
            sections[section] += 1
            categories[category] += 1

            for language, column in LANGUAGES:
                if column is None:
                    prompt = _cell(row.get("English prompt"))
                else:
                    prompt = _cell(row.get(column))
                if not prompt:
                    empty_translations.append(f"row {index} {language}")
                    continue
                record = {
                    "prompt": prompt,
                    "target": target,
                    "language": language,
                    "eu_ai_act_category": category,
                    "annex_section": section,
                    "reasoning": reasoning,
                    "reasoning2": reasoning2,
                    "is_machine_translation": language != "english",
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    if empty_translations:
        raise SystemExit(
            f"{len(empty_translations)} empty prompts, first: {empty_translations[:5]}"
        )

    manifest = {
        "name": "prohibited_article5",
        "source_csv": str(csv_path),
        "path": str(jsonl_path.relative_to(ROOT)),
        "n_source_prompts": len(rows),
        "n_languages": len(LANGUAGES),
        "n_rows": n_written,
        "target_field": "Annotator 1",
        "target_counts_source_prompts": dict(targets),
        "section_counts_source_prompts": dict(sections),
        "category_counts_source_prompts": dict(categories),
        "languages": [language for language, _column in LANGUAGES],
        "field_map": {
            "prompt": "English prompt, or translation_<language>",
            "target": "Annotator 1 (should_refuse | should_answer)",
            "eu_ai_act_category": "Prohibition_category",
            "annex_section": "section_number (Article 5(1), not Annex III)",
            "reasoning": "Reasoning Annotator 1",
            "reasoning2": "Reasoning Annotator 6",
            "is_machine_translation": "false for english; true for every translation_* column",
        },
        "notes": [
            "Annotator 6 is not the gold label. It mostly agrees with Annotator 1 but includes typos, blanks, and a few disagreements.",
            "Non-English rows are machine translations. This CSV has no separate verified German/Chinese/Polish/Italian columns.",
            "Spot check: Greek on source row 0 is the public-pool biometric prompt, not the English subliminal-audio prompt.",
            "Two translation strings are duplicated, so row_id collides and the harness keeps one of each pair (10638 scored rows): German of the Instagram facial-scraping prompt copies the LinkedIn prompt; Greek of the news-amplification prompt copies the public-square child-support prompt.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build(DEFAULT_CSV, DEFAULT_JSONL, DEFAULT_MANIFEST)
    print(json.dumps({k: manifest[k] for k in ("path", "n_source_prompts", "n_languages", "n_rows", "target_counts_source_prompts", "section_counts_source_prompts")}, indent=2))


if __name__ == "__main__":
    main()
