"""Tests for brain_mapping.report.ClinicalReportGenerator.

These tests do not require torch - report.py only depends on numpy and
brain_mapping.utils (which is itself torch-free at import time).

They exist specifically to lock in the honesty fix described in the repo
README: the report must only ever contain numeric ROI ids (e.g.
"ROI  45"), never fabricated anatomical names (e.g. "Left Temporal
Lobe") that this codebase has no way to produce (there is no atlas
label -> name lookup implemented anywhere).
"""

import re

from brain_mapping.report import ClinicalReportGenerator


def _sample_report(**overrides):
    kwargs = dict(
        case_name="test_case_001",
        tumor_volume=1234.5,
        affected_rois=[(45, 0.873), (46, 0.821), (47, 0.654)],
        uncertainty_stats={"mean": 0.12, "max": 0.45, "p95": 0.38},
        connectivity_disruption={45: 0.82, 46: 0.76},
    )
    kwargs.update(overrides)
    return ClinicalReportGenerator.generate_report(**kwargs)


def test_report_contains_case_id_and_volume():
    report = _sample_report()
    assert "test_case_001" in report
    assert "1234.50" in report


def test_report_lists_numeric_roi_ids_only():
    report = _sample_report()
    # Actual output format from the code: "  1. ROI  45: 87.3% overlap"
    assert re.search(r"ROI\s+45\D*87\.3%", report)
    assert re.search(r"ROI\s+46\D*82\.1%", report)


def test_report_does_not_contain_fabricated_region_names():
    """No anatomical-name lookup exists in this codebase, so the report
    must never contain named regions - guards against reintroducing the
    kind of fabricated content that was in the old README example
    (e.g. "(Left Temporal Lobe)")."""
    report = _sample_report()
    for fabricated_name in ["Temporal Lobe", "Hippocampus", "Parahippocampal", "Frontal Lobe"]:
        assert fabricated_name not in report


def test_report_includes_connectivity_section_when_provided():
    report = _sample_report()
    assert "FUNCTIONAL CONNECTIVITY IMPACT" in report
    assert "disruption score" in report


def test_report_omits_connectivity_section_when_absent():
    report = _sample_report(connectivity_disruption=None)
    assert "FUNCTIONAL CONNECTIVITY IMPACT" not in report


def test_report_limits_to_top_10_rois():
    many_rois = [(i, 1.0 - i * 0.01) for i in range(1, 21)]
    report = _sample_report(affected_rois=many_rois)
    roi_lines = [line for line in report.splitlines() if re.match(r"\s*\d+\.\s+ROI", line)]
    assert len(roi_lines) == 10


def test_report_saved_to_file(tmp_path):
    save_path = tmp_path / "report.txt"
    report = _sample_report(save_path=str(save_path))
    assert save_path.exists()
    assert save_path.read_text(encoding="utf-8") == report
