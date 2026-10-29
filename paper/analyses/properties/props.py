"""Sequence properties for the graded steering tasks. Heavy chain = first 140 residues of a p-IgGen output
(VH + VL concatenated) or a VH. CDR3 = residues after the last C that lies 6-30 residues before the FR4 W[GA]xG
(first such motif after residue 90) up to that W, exclusive (IMGT 105-117). Light chain = residues after the heavy
chain's FR4 ...VTVSS (p-IgGen paired output). kappa/lambda from the light chain's FR4 end."""
import re
WG = re.compile(r"W[GA].G")
def cdr3(s):
    h = s[:140]; m = [x for x in WG.finditer(h) if x.start() >= 90]
    if not m: return None
    w = m[0].start(); c = h.rfind("C", max(0, w - 31), w - 5)
    return h[c + 1:w] if c >= 0 else None
def light(s):
    k = s.find("VSS", 95)
    return s[k + 3:] if k > 0 else None
def lc_type(s):   # light-chain FR4 ends ...EIK / ...KVEIK (kappa) or ...TVL (lambda); require the FR4 G-x-G-T 6-10 residues from the end
    l = light(s)
    if not l or len(l) < 80 or not re.search(r"G.GT", l[-12:]): return None
    return "kappa" if l[-1] in "KR" else "lambda" if l[-1] == "L" else None
def props(s):
    c = cdr3(s)
    if c is None or not 3 <= len(c) <= 30: return None
    return dict(len=len(c), Y=c.count("Y") / len(c), charge=c.count("K") + c.count("R") - c.count("D") - c.count("E"))
