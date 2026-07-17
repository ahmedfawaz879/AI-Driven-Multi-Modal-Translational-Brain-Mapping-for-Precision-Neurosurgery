"""Clinical text report generation.

Extracted verbatim from the original script (ClinicalReportGenerator) -
no output text was changed. This is called out explicitly because it
matters for an honesty issue in the *README*, not this file: the report
below lists affected ROIs by their **numeric atlas label only** (e.g.
"ROI  45: 87.3% overlap"). There is no atlas-label-to-anatomical-name
lookup anywhere in this codebase, so this generator cannot and never
could produce named regions (e.g. "Left Temporal Lobe"). The original
project's README showed a fabricated-looking example with invented named
regions and disruption commentary that this code has never been able to
produce. That example has been corrected in README.md to match this
function's real, verbatim output (see the "Example output - illustrative,
not real results" section there).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import logger

# ======================== Clinical Report Generator ========================


class ClinicalReportGenerator:
    """Generate clinical reports from analysis results"""

    @staticmethod
    def generate_report(case_name: str,
                       tumor_volume: float,
                       affected_rois: List[Tuple[int, float]],
                       uncertainty_stats: Dict[str, float],
                       connectivity_disruption: Optional[Dict] = None,
                       save_path: Optional[str] = None) -> str:
        """Generate structured clinical report"""

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          AI-Driven Brain Tumor Analysis Report               ║
╚══════════════════════════════════════════════════════════════╝

Case ID: {case_name}
Analysis Date: {np.datetime64('today')}

─────────────────────────────────────────────────────────────

TUMOR CHARACTERISTICS:
  • Estimated Volume: {tumor_volume:.2f} mm³
  • Mean Uncertainty: {uncertainty_stats.get('mean', 0):.3f}
  • Max Uncertainty: {uncertainty_stats.get('max', 0):.3f}
  • 95th Percentile Uncertainty: {uncertainty_stats.get('p95', 0):.3f}

─────────────────────────────────────────────────────────────

AFFECTED BRAIN REGIONS (Top 10):
"""
        for i, (roi, score) in enumerate(affected_rois[:10], 1):
            report += f"  {i:2d}. ROI {roi:3d}: {score*100:5.1f}% overlap\n"

        if connectivity_disruption:
            report += """
─────────────────────────────────────────────────────────────

FUNCTIONAL CONNECTIVITY IMPACT:
"""
            for roi, impact in list(connectivity_disruption.items())[:5]:
                report += f"  • ROI {roi}: {impact:.2f} disruption score\n"

        report += """
─────────────────────────────────────────────────────────────

INTERPRETATION NOTES:
  • High uncertainty regions require additional clinical review
  • Affected ROIs indicate potential functional impact zones
  • Connectivity analysis shows network-level implications


─────────────────────────────────────────────────────────────
"""

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f'Saved clinical report to {save_path}')

        return report
