"""Ablation tasks (Supplementary Section A.9): one per single latent (18: the highest-F1 latent of each J gene in each
p-IgGen matched SAE, ablation/single_latents.txt) and one per co-activation group (22: TopK and Batch TopK groups at
phi >= 0.7 and phi >= 0.5, redundancy/coactivation_groups.txt). Both lists are deposited results (a copy under
<work_dir>/ takes precedence). Writes <work_dir>/ablation/tasks.txt, one task per line: "kind sae gene latents"
(latents joined by '+'). Run: python -m paper.analyses.ablation.make_tasks; then paper/scripts/ablation/ablate.sh
<line index> per task."""
from plm_sae import paths as P
from paper.benchmark._inputs import coactivation_groups, inp


def single_latents():
    """(sae, gene, [latent]) of ablation/single_latents.txt (lines "sae gene latent")."""
    for line in open(inp("ablation", "single_latents.txt")):
        if line.strip():
            sae, gene, lat = line.split()
            yield sae, gene, [int(lat)]


def tasks():
    """(kind, phi, sae, gene, latents) in the scorer's order."""
    for sae, gene, lats in single_latents():
        yield "single", "", sae, gene, lats
    for phi, sae, gene, lats in coactivation_groups():
        yield "group", phi, sae, gene, lats


def main():
    rows = [f"{k} {sae} {g} {'+'.join(map(str, lats))}" for k, _, sae, g, lats in tasks()]
    out = P.work("ablation", "tasks.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{r}\n" for r in rows))
    print(f"{out}: {sum(r.startswith('single') for r in rows)} single, {sum(r.startswith('group') for r in rows)} group tasks")


if __name__ == "__main__":
    main()
