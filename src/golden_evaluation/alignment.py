"""Bounded monotonic 1:1, 1:N and N:1 alignment. No timestamp shifts."""
from dataclasses import dataclass, asdict
import math
from src.golden_evaluation.text_metrics import tokens, edit_counts


@dataclass(frozen=True)
class EvaluationConfig:
    alignment_max_time_delta_seconds: float = 15.0
    alignment_min_text_similarity: float = 0.35
    alignment_max_group: int = 3
    golden_timestamp_tolerance_seconds: float = 1.0
    fillers: tuple = ("ờ", "ừ", "ừm", "ờm", "à")

    def __post_init__(self):
        if not math.isfinite(self.alignment_max_time_delta_seconds) or self.alignment_max_time_delta_seconds < 0 or not 0 <= self.alignment_min_text_similarity <= 1 or not 1 <= self.alignment_max_group <= 5 or not math.isfinite(self.golden_timestamp_tolerance_seconds) or self.golden_timestamp_tolerance_seconds < 0:
            raise ValueError("EVALUATION_CONFIG_INVALID")

    def to_dict(self):
        return asdict(self)


def align(reference, model, config):
    n,m = len(reference),len(model)
    costs = [[float("inf") for _ in range(m+1)] for _ in range(n+1)]
    paths = {}
    costs[0][0] = 0
    def update(i,j,a,b,cost):
        if cost < costs[a][b] - 1e-12:
            costs[a][b] = cost
            paths[(a,b)] = (i,j)
    for i in range(n+1):
        for j in range(m+1):
            base = costs[i][j]
            for a,b in [(1,1)]+[(1,k) for k in range(2,config.alignment_max_group+1)]+[(k,1) for k in range(2,config.alignment_max_group+1)]:
                if i+a > n or j+b > m:
                    continue
                ref, hyp = reference[i:i+a], model[j:j+b]
                if abs(ref[0]["start_seconds"]-hyp[0]["start_seconds"]) > config.alignment_max_time_delta_seconds:
                    continue
                # Group spans must remain local, preventing distant material from
                # being absorbed merely to make similarity look better.
                if max(ref[-1]["start_seconds"]-ref[0]["start_seconds"],hyp[-1]["start_seconds"]-hyp[0]["start_seconds"]) > config.alignment_max_time_delta_seconds:
                    continue
                rt=tokens(" ".join(s["text"] for s in ref),True)
                mt=tokens(" ".join(s["text"] for s in hyp),True)
                edits=edit_counts(rt,mt)
                similarity=1-(edits["substitutions"]+edits["deletions"]+edits["insertions"])/max(len(rt),len(mt),1)
                if similarity >= config.alignment_min_text_similarity:
                    update(i,j,i+a,j+b,base+(1-similarity)*max(a,b)+.05*(a+b-2))
            if i<n: update(i,j,i+1,j,base+1)
            if j<m: update(i,j,i,j+1,base+1)
    groups=[]
    i,j=n,m
    while i or j:
        a,b=paths[(i,j)]
        groups.append({"reference_indices":list(range(a,i)),"model_indices":list(range(b,j)),
                       "resolved":i>a and j>b})
        i,j=a,b
    return list(reversed(groups))
