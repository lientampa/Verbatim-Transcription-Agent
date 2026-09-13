"""Global one-to-one anonymous mapping; never remap named labels to hide swaps."""
import re
from collections import Counter, defaultdict
from src.golden_evaluation.text_metrics import ratio


def anonymous(label):
    return re.fullmatch(r"Người nói \d+", label) is not None


def optimal_assignment(weights):
    """Deterministic Hungarian maximization on a square padded count matrix."""
    n=len(weights)
    if not n: return []
    u,v,p,way=[0]*(n+1),[0]*(n+1),[0]*(n+1),[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i; j0=0; minimum=[float("inf")]*(n+1); used=[False]*(n+1)
        while True:
            used[j0]=True; i0=p[j0]; delta=float("inf"); j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cost=-weights[i0-1][j-1]-u[i0]-v[j]
                    if cost<minimum[j]: minimum[j]=cost; way[j]=j0
                    if minimum[j]<delta: delta=minimum[j]; j1=j
            for j in range(n+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minimum[j]-=delta
            j0=j1
            if p[j0]==0: break
        while True:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
            if j0==0: break
    assignment=[None]*n
    for j in range(1,n+1): assignment[p[j]-1]=j-1
    return assignment


def speakers(pairs, golden_labels):
    anonymous_labels=sorted({m for g,m in pairs if anonymous(m)})
    # Explicitly correct named labels reserve their identities.
    reserved={m for g,m in pairs if not anonymous(m) and m in golden_labels}
    targets=sorted(set(golden_labels)-reserved)
    n=max(len(anonymous_labels),len(targets))
    counts=Counter(pairs)
    weights=[[counts[(targets[j],anonymous_labels[i])] if i<len(anonymous_labels) and j<len(targets) else 0 for j in range(n)] for i in range(n)]
    assignment=optimal_assignment(weights)
    mapping={label:targets[assignment[i]] for i,label in enumerate(anonymous_labels) if assignment[i]<len(targets) and weights[i][assignment[i]]>0}
    correct=sum(mapping.get(m,m)==g for g,m in pairs)
    by_golden=defaultdict(set); by_model=defaultdict(set)
    for g,m in pairs: by_golden[g].add(m); by_model[m].add(g)
    return {"speaker_attributed_segments":len(pairs),"speaker_correct_segments":correct,
            "speaker_accuracy":ratio(correct,len(pairs)),"anonymous_mapping":mapping,
            "speaker_fragmentation":sum(len(labels)>1 for labels in by_golden.values()),
            "speaker_swap":sum(len(labels)>1 for labels in by_model.values()),
            "under_identified_count":sum(anonymous(m) and not anonymous(g) for g,m in pairs),
            "unsupported_named_labels":sorted({m for g,m in pairs if not anonymous(m) and m not in golden_labels})}
