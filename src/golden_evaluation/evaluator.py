"""Deterministic offline metrics and review flags; never modifies transcripts."""
import math
import re
from collections import Counter
from copy import deepcopy
from src.golden_evaluation.fixtures import validate_reference, canonical_segments
from src.golden_evaluation.alignment import EvaluationConfig, align
from src.golden_evaluation.text_metrics import normalize, tokens, ratio, text_metrics
from src.golden_evaluation.speaker_metrics import speakers, anonymous


def event_counts(segments, kind, config):
    result=Counter()
    for segment in segments:
        if "events" in segment:
            values=[e["text"] for e in segment["events"] if e["type"]==kind]
        else:
            words=tokens(segment["text"],True)
            if kind=="filler": values=[w for w in words if w in config.fillers]
            elif kind=="repetition": values=[a+" "+b for a,b in zip(words,words[1:]) if a==b]
            else: values=[m.group(0).strip() for m in re.finditer(r"[^.!?…]+(?:\.\.\.|…)",segment["text"])]
        for value in values:
            key=normalize(value,True) if kind=="false_start" else " ".join(tokens(value,True))
            result[key]+=1
    return result


def model_events(segments, kind, config, reference_events):
    result=event_counts([{k:v for k,v in s.items() if k!="events"} for s in segments],kind,config)
    text=" ".join(s["text"] for s in segments)
    words=tokens(text,True)
    for key in reference_events:
        if kind=="false_start": count=normalize(text,True).count(key)
        else:
            phrase=tokens(key,True)
            count=sum(words[i:i+len(phrase)]==phrase for i in range(len(words)-len(phrase)+1)) if phrase else 0
        result[key]=max(result[key],count)
    return result


def event_metric(reference_count, model_count, match_count):
    return {"reference_count":reference_count,"model_count":model_count,"match_count":match_count,
            "precision":ratio(match_count,model_count),"recall":ratio(match_count,reference_count)}


def timestamp_metrics(errors, tolerance):
    ordered=sorted(errors)
    if not ordered:
        return {"count":0,"errors":[],"median":None,"p90":None,"max":None,
                **{key:ratio(0,0) for key in ("within_1s_rate","within_2s_rate","within_5s_rate","within_tolerance_rate")}}
    n=len(ordered)
    median=(ordered[(n-1)//2]+ordered[n//2])/2
    return {"count":n,"errors":list(errors),"median":median,"p90":ordered[math.ceil(.9*n)-1],"max":ordered[-1],
            **{key:ratio(sum(e<=limit for e in ordered),n) for key,limit in [("within_1s_rate",1),("within_2s_rate",2),("within_5s_rate",5),("within_tolerance_rate",tolerance)]}}


def evaluate(reference, recognized, source_fingerprint=None, config=None, provenance=None):
    config=config or EvaluationConfig()
    reference=validate_reference(reference,source_fingerprint)
    model=canonical_segments(recognized,reference["source"]["duration_seconds"])
    ref=reference["segments"]
    joined=lambda items:" ".join(s["text"] for s in items)
    text=text_metrics(joined(ref),joined(model))
    groups=align(ref,model,config)
    events={kind:[event_counts(ref,kind,config),None,0] for kind in ("filler","repetition","false_start")}
    for kind,counts in events.items(): counts[1]=model_events(model,kind,config,counts[0])
    uncertainty=[sum(s["text"].count("[không rõ]") for s in ref),sum(s["text"].count("[không rõ]") for s in model),0]
    errors,pairs,queue=[],[],[]
    covered=0; names_total=0; names_match=0
    golden_labels={s["speaker"] for s in ref}
    def review(g,m,flags,severity="HIGH"):
        if not flags: return
        all_segments=g+m
        start=min((s["start_seconds"] for s in all_segments),default=0)
        later=[s["start_seconds"] for s in ref+model if s["start_seconds"]>start]
        end=min(later,default=reference["source"]["duration_seconds"])
        queue.append({"fixture_id":reference["fixture_id"],"start_seconds":start,"end_seconds":end,
                      "golden_text":joined(g),"model_text":joined(m),"flags":sorted(set(flags)),"severity":severity})
    for group in groups:
        g=[ref[i] for i in group["reference_indices"]]; m=[model[i] for i in group["model_indices"]]
        flags=[]
        for seg in g: names_total+=len(seg.get("proper_names",[]))
        if not group["resolved"]:
            flags.append("ALIGNMENT_UNRESOLVED")
        else:
            edits=text_metrics(joined(g),joined(m))["relaxed_wer"]
            covered+=max(0,edits["reference_count"]-edits["deletions"]-edits["substitutions"])
            # For grouped alignment only the shared anchor is observed. Do not
            # invent intermediate word/segment timestamps.
            errors.append(abs(g[0]["start_seconds"]-m[0]["start_seconds"]))
            if errors[-1]>config.golden_timestamp_tolerance_seconds: flags.append("TIMESTAMP_ERROR")
            for gs in g:
                for ms in m: pairs.append((gs["speaker"],ms["speaker"]))
            for kind,counts in events.items():
                gr=event_counts(g,kind,config); mr=model_events(m,kind,config,gr)
                matches=sum((gr & mr).values()); counts[2]+=matches
                if matches<sum(gr.values()): flags.append(kind.upper()+"_LOSS")
            gc=sum(s["text"].count("[không rõ]") for s in g); mc=sum(s["text"].count("[không rõ]") for s in m)
            uncertainty[2]+=min(gc,mc)
            if gc>mc and any(w!="[không rõ]" for w in tokens(joined(m),True)): flags.append("UNCERTAIN_OVERGUESS")
            if mc>gc: flags.append("FALSE_UNCERTAINTY")
            for seg in g:
                for name in seg.get("proper_names",[]):
                    if re.search(r"(?<!\w)"+re.escape(normalize(name))+r"(?!\w)", normalize(joined(m))): names_match+=1
                    else: flags.append("PROPER_NAME_MISMATCH")
        lexical=text_metrics(joined(g),joined(m))["relaxed_wer"]
        if lexical["deletions"]: flags.append("LIKELY_OMISSION")
        if lexical["insertions"]: flags.append("POSSIBLE_HALLUCINATION")
        if lexical["substitutions"]: flags.append("TEXT_DISAGREEMENT")
        review(g,m,flags)
    speaker=speakers(pairs,golden_labels)
    mapping=speaker["anonymous_mapping"]
    for group in groups:
        if not group["resolved"]: continue
        g=[ref[i] for i in group["reference_indices"]]; m=[model[i] for i in group["model_indices"]]
        flags=[]
        for gs in g:
            for ms in m:
                label=ms["speaker"]
                if mapping.get(label,label)!=gs["speaker"]: flags.append("SPEAKER_MISMATCH")
                if anonymous(label) and not anonymous(gs["speaker"]): flags.append("SPEAKER_UNDER_IDENTIFIED")
                if not anonymous(label) and label not in golden_labels: flags.append("SPEAKER_NAME_HALLUCINATION")
        if speaker["speaker_swap"]: flags.append("SPEAKER_SWAP_REVIEW")
        if speaker["speaker_fragmentation"]: flags.append("SPEAKER_FRAGMENTATION")
        review(g,m,flags)
    for region in reference.get("regions",[]):
        inside=[s for s in model if region["start_seconds"]<=s["start_seconds"]<region["end_seconds"]]
        if region["type"]=="overlap":
            if inside: review([],inside,["OVERLAP_REVIEW"],"MEDIUM")
        elif region.get("authoritative") and inside:
            review([],inside,["POSSIBLE_NON_SPEECH_HALLUCINATION"])
    lexical=text["relaxed_wer"]
    return {"fixture_id":reference["fixture_id"],"review_status":reference["review"]["status"],
            "duration_seconds":reference["source"]["duration_seconds"],"source_fingerprint":reference["source"]["fingerprint_sha256"],
            "reference_segments":len(ref),"model_segments":len(model),"config":config.to_dict(),
            "text":text,"verbatim":{kind:event_metric(sum(c[0].values()),sum(c[1].values()),c[2]) for kind,c in events.items()},
            "omission":ratio(lexical["deletions"],lexical["reference_count"]),"insertion":ratio(lexical["insertions"],lexical["reference_count"]),
            "alignment":groups,"timestamp":timestamp_metrics(errors,config.golden_timestamp_tolerance_seconds),"speaker":speaker,
            "uncertainty":event_metric(*uncertainty),"golden_speech_coverage":ratio(covered,lexical["reference_count"]),
            "proper_names":{"name_reference_count":names_total,"name_exact_match_count":names_match,"name_error_count":names_total-names_match,"name_accuracy":ratio(names_match,names_total)},
            "review_queue":queue,"provenance":deepcopy(provenance or {}),"execution_origin":"OFFLINE_STORED_CANONICAL",
            "evaluation_status":"PARTIAL" if any(not g["resolved"] for g in groups) else "PASS"}
