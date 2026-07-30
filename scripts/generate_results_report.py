from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_optional(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.is_file():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _interval(values: dict[str, Any] | None) -> str:
    if not values:
        return "Pending"
    return f"{_percent(values['lower_95'])} to {_percent(values['upper_95'])}"


def generate_report(
    metadata: dict[str, Any] | None,
    clean: dict[str, Any] | None,
    field: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    latency: dict[str, Any] | None,
    user_study: dict[str, Any] | None,
    evidence_hashes: dict[str, str],
) -> str:
    sections = [
        "# TomatoGuard evidence report",
        "",
        f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}",
        "",
        "This report prints only supplied evidence. Missing results remain `Pending`; they are never replaced by zero, estimated, or tutorial metrics.",
        "",
        "## Evidence status",
        "",
        "| Evidence | Status |",
        "| --- | --- |",
    ]
    items = {
        "Model metadata": metadata,
        "Locked clean and OOD evaluation": clean,
        "Expert field evaluation": field,
        "Comparable before/after evaluation": comparison if comparison and comparison.get("comparable") is True else None,
        "CPU latency benchmark": latency,
        "Aggregate user study": user_study if user_study and user_study.get("publishable") is True else None,
        "User-study impact comparison": user_study if user_study and user_study.get("impact_publishable") is True else None,
    }
    sections.extend(
        f"| {name} | {'Present' if value is not None else 'Pending'} |"
        for name, value in items.items()
    )

    if metadata:
        sections.extend(
            [
                "",
                "## Candidate identity",
                "",
                f"- Version: `{metadata.get('version', 'not recorded')}`",
                f"- Model SHA-256: `{metadata.get('model_sha256', 'not recorded')}`",
                f"- Calibration fitted: `{metadata.get('calibration', {}).get('fitted', False)}`",
                f"- OOD rejection validated: `{metadata.get('rejection', {}).get('validated', False)}`",
            ]
        )
    if clean:
        intervals = clean.get("classification_bootstrap_95_ci", {})
        sections.extend(
            [
                "",
                "## Locked clean and OOD results",
                "",
                f"- Macro F1: {_percent(clean['macro_f1'])}",
                f"- Macro F1 stratified bootstrap 95% CI: {_interval(intervals.get('macro_f1'))}",
                f"- ECE (15 bins): {_percent(clean['ece_15_bin'])}",
                f"- Accepted coverage: {_percent(clean['selective']['acceptance_coverage'])}",
                f"- Locked OOD false acceptance: {_percent(clean['ood_test']['false_acceptance_rate'])}",
                "",
                "| Class | Precision | Recall | F1 | F1 95% CI |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for class_id, values in sorted(clean.get("per_class", {}).items()):
            class_interval = (
                intervals.get("per_class", {}).get(class_id, {}).get("f1")
            )
            sections.append(
                f"| `{class_id}` | {_percent(values['precision'])} | "
                f"{_percent(values['recall'])} | {_percent(values['f1'])} | {_interval(class_interval)} |"
            )
    if field:
        interval = field.get("macro_f1_bootstrap_95_ci", {})
        sections.extend(
            [
                "",
                "## Expert field results",
                "",
                f"- Samples: {int(field['sample_count'])}",
                f"- Macro F1: {_percent(field['macro_f1'])}",
                f"- Macro F1 bootstrap 95% CI: {_percent(interval['lower_95'])} to {_percent(interval['upper_95'])}",
                f"- ECE (15 bins): {_percent(field['ece_15_bin'])}",
                f"- Acceptance coverage: {_percent(field['acceptance_coverage'])}",
                f"- Not externally validated: {', '.join(field.get('not_externally_validated_classes', [])) or 'None'}",
            ]
        )
    if comparison:
        sections.extend(["", "## Before/after comparison", ""])
        if comparison.get("comparable") is not True:
            sections.append(
                "No numeric comparison is published because the baseline and candidate were not evaluated on the same locked samples and metric definitions."
            )
        else:
            sections.extend(
                [
                    f"- Baseline {comparison['metric']}: {_percent(comparison['baseline'])}",
                    f"- Candidate {comparison['metric']}: {_percent(comparison['candidate'])}",
                    f"- Absolute change: {float(comparison['candidate']) - float(comparison['baseline']):+.4f}",
                    f"- Locked evaluation ID: `{comparison['evaluation_id']}`",
                ]
            )
    if latency:
        sections.extend(
            [
                "",
                "## CPU serving benchmark",
                "",
                f"- p50: {float(latency['p50_ms']):.1f} ms",
                f"- p95: {float(latency['p95_ms']):.1f} ms",
                f"- Model size: {float(latency['model_size_mb']):.2f} MB",
                f"- Device: {latency.get('device', 'not recorded')}",
            ]
        )
    if user_study and user_study.get("publishable") is True:
        sections.extend(
            [
                "",
                "## Aggregate user study",
                "",
                f"- Consented responses: {int(user_study['participant_count'])}",
                f"- Task completion: {_percent(user_study['task_completion_rate'])}",
                f"- Unassisted interpretation: {_percent(user_study['unassisted_interpretation_rate'])}",
                f"- Uncertainty understanding: {_percent(user_study['uncertainty_understanding_rate'])}",
            ]
        )
    impact = user_study.get("impact_comparison") if user_study else None
    if impact and impact.get("publishable") is True:
        primary_metric = impact["primary_metric"]
        primary_interval = impact["candidate_minus_baseline_95_ci"][primary_metric]
        sections.extend(
            [
                "",
                "## User-study version impact",
                "",
                f"- Unit: {impact['unit']}",
                f"- Baseline: `{impact['baseline_version']}`",
                f"- Candidate: `{impact['candidate_version']}`",
                f"- Preregistered primary metric: `{primary_metric}`",
                f"- Candidate minus baseline: {float(primary_interval['candidate_minus_baseline']):+.4f}",
                f"- Bootstrap 95% CI: {float(primary_interval['lower_95']):+.4f} to {float(primary_interval['upper_95']):+.4f}",
                f"- Demonstrated improvement: {impact['demonstrated_improvement']}",
            ]
        )
    sections.extend(["", "## Evidence file hashes", ""])
    if evidence_hashes:
        sections.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(evidence_hashes.items()))
    else:
        sections.append("No evidence files supplied.")
    sections.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These metrics describe the declared locked datasets and study sample only. They do not establish definitive diagnosis, universal farm performance, treatment efficacy, or population representativeness.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-only TomatoGuard report")
    for name in ("metadata", "clean_metrics", "field_metrics", "comparison", "latency", "user_study"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    hashes = {}
    for name in ("metadata", "clean_metrics", "field_metrics", "comparison", "latency", "user_study"):
        value, digest = load_optional(getattr(args, name))
        loaded[name] = value
        if digest:
            hashes[name] = digest
    report = generate_report(
        loaded["metadata"], loaded["clean_metrics"], loaded["field_metrics"],
        loaded["comparison"], loaded["latency"], loaded["user_study"], hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote evidence report: {args.output}")


if __name__ == "__main__":
    main()
