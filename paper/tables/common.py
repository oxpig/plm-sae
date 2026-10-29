"""Helpers shared by the table builders."""
import numpy as np
from scipy import stats

MN = {"piggen": "p-IgGen", "IgLM": "IgLM", "progen2_oas": "ProGen2-OAS"}


def ok(s):
    """Criterion column (pass / fail, or booleans) as a boolean Series."""
    return s.astype(str).isin(["pass", "True"])


def pfmt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def mcnemar(a, b):
    """Exact two-sided McNemar test; a and b are booleans over the same latents or pairs, in the same order."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
