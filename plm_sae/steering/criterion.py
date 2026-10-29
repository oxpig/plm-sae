"""The correlation criterion of steering (Methods): in one seed, the target proportion rises with alpha if Spearman's
rho over the alpha ladder is at least 0.7 with p < 0.05 in the expected direction; a latent passes if this holds in at
least 2 of its 3 seeds. The magnitude criterion is applied on top of it by the analysis scripts
(paper.benchmark.analyze_final)."""
import math

MIN_ABS_RHO = 0.7
P_THRESHOLD = 0.05
SEEDS_REQUIRED = 2


def judge_success(rho, p, sign=0):
    """Verdict for one seed: pass, insufficient, fail_weak_rho, fail_p or fail_direction. sign: +1 if the proportion
    should rise with alpha, -1 if it should fall, 0 for either. The callers require at least three strengths."""
    if rho is None or math.isnan(rho):
        return "insufficient"
    if abs(rho) < MIN_ABS_RHO:
        return "fail_weak_rho"
    if not 0 <= p < P_THRESHOLD:
        return "fail_p"
    if sign != 0 and math.copysign(1, rho) != sign:
        return "fail_direction"
    return "pass"


def judge_pair(results, hypothesised_sign=0):
    """(verdict, seeds passing, seeds) over per-seed dicts with spearman_rho and spearman_p; verdict pass if at least
    SEEDS_REQUIRED seeds pass, insufficient if there are no seeds."""
    n_pass = sum(judge_success(r["spearman_rho"], r["spearman_p"], hypothesised_sign) == "pass" for r in results)
    if not results:
        return "insufficient", 0, 0
    return ("pass" if n_pass >= SEEDS_REQUIRED else "fail"), n_pass, len(results)
