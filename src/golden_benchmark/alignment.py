"""Deterministic text-order alignment; human timing is supporting evidence only."""
from dataclasses import dataclass
from collections import Counter,defaultdict
from src.golden_evaluation.text_metrics import tokens,edit_counts
from src.golden_evaluation.speaker_metrics import optimal_assignment

@dataclass(frozen=True)
class AlignmentConfig:
    tolerance_seconds: float=15.0
    max_group: int=3
    min_similarity: float=.15
    def __post_init__(self):
        import math
        if not math.isfinite(self.tolerance_seconds) or self.tolerance_seconds<=0 or not 1<=self.max_group<=5 or not 0<=self.min_similarity<=1:
            raise ValueError('BENCHMARK_ALIGNMENT_CONFIG')

def align(human,machine,config=AlignmentConfig()):
    n,m=len(human),len(machine);cost=[[float('inf')]*(m+1) for _ in range(n+1)];path={};cost[0][0]=0
    def update(i,j,a,b,value):
        if value<cost[a][b]-1e-12:cost[a][b]=value;path[a,b]=(i,j)
    ht={(i,k):tokens(' '.join(s.text for s in human[i:i+k]),True) for i in range(n) for k in range(1,config.max_group+1) if i+k<=n}
    mt={(i,k):tokens(' '.join(s.text for s in machine[i:i+k]),True) for i in range(m) for k in range(1,config.max_group+1) if i+k<=m}
    for i in range(n+1):
        for j in range(m+1):
            for a,b in [(1,1)]+[(1,k) for k in range(2,config.max_group+1)]+[(k,1) for k in range(2,config.max_group+1)]:
                if i+a>n or j+b>m:continue
                hs,ms=human[i:i+a],machine[j:j+b];r,t=ht[i,a],mt[j,b]
                timing=all(s.timestamp_reliable for s in hs+ms)
                delta=abs(hs[0].seconds-ms[0].seconds)
                # Cheap timing gate, but lexical matches can recover bad human metadata.
                if timing and delta>config.tolerance_seconds and not set(r)&set(t):continue
                edits=edit_counts(r,t);distance=sum(edits[k] for k in ('substitutions','deletions','insertions'))
                similarity=1-distance/max(len(r),len(t),1)
                if similarity<config.min_similarity:continue
                if timing and delta>config.tolerance_seconds and (similarity<.9 or min(len(r),len(t))<5):continue
                penalty=.1*min(delta/config.tolerance_seconds,1) if timing else 0
                update(i,j,i+a,j+b,cost[i][j]+(1-similarity)*max(a,b)+.08*(a+b-2)+penalty)
            if i<n:update(i,j,i+1,j,cost[i][j]+1)
            if j<m:update(i,j,i,j+1,cost[i][j]+1)
    groups=[];i,j=n,m
    while i or j:
        a,b=path[i,j];groups.append(dict(human=list(range(a,i)),machine=list(range(b,j)),resolved=i>a and j>b));i,j=a,b
    return list(reversed(groups))

def speaker_metrics(human,machine,groups,excluded_labels=()):
    # Labels such as Anh/Chi are display names, not asserted real identities here.
    pairs=[]
    for g in groups:
        if not g['resolved']:continue
        h={human[i].speaker_id for i in g['human'] if human[i].speaker_display_name.strip() not in excluded_labels};m={machine[i].speaker_id for i in g['machine'] if machine[i].speaker_display_name.strip() not in excluded_labels}
        if len(h)==len(m)==1:pairs.append((next(iter(h)),next(iter(m))))
    hs=sorted({s.speaker_id for s in human if s.speaker_display_name.strip() not in excluded_labels});ms=sorted({s.speaker_id for s in machine if s.speaker_display_name.strip() not in excluded_labels});n=max(len(hs),len(ms));c=Counter(pairs)
    weights=[[c[hs[j],ms[i]] if i<len(ms) and j<len(hs) else 0 for j in range(n)] for i in range(n)]
    assignment=optimal_assignment(weights)
    mapping={m:hs[assignment[i]] for i,m in enumerate(ms) if assignment[i]<len(hs) and weights[i][assignment[i]]>0}
    byh=defaultdict(set);bym=defaultdict(set)
    for h,m in pairs:byh[h].add(m);bym[m].add(h)
    return dict(mapping=mapping,eligible_groups=len(pairs),errors=sum(mapping.get(m)!=h for h,m in pairs),
        fragmentation_candidates=sum(len(v)>1 for v in byh.values()),merge_candidates=sum(len(v)>1 for v in bym.values()),
        unmatched_machine_identities=[m for m in ms if m not in mapping],
        human_labels={s.speaker_id:s.speaker_display_name for s in human},machine_labels={s.speaker_id:s.speaker_display_name for s in machine},
        limitation='Group-count weighted label permutation; mixed-speaker groups excluded. Candidates require acoustic adjudication; not DER.')
