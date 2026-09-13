"""Denominator-weighted reports and a conservative, explicit release gate."""
from collections import Counter
import json
from pathlib import Path
from src.golden_evaluation.fixtures import REVIEWED
from src.golden_evaluation.text_metrics import ratio
from src.golden_evaluation.evaluator import timestamp_metrics, event_metric

THRESHOLDS={"MAX_STRICT_WER":None,"MAX_RELAXED_WER":None,"MAX_CER":None,
            "MIN_GOLDEN_SPEECH_COVERAGE":None,"MIN_SPEAKER_ACCURACY":None,
            "MAX_MEDIAN_TIMESTAMP_ERROR_SECONDS":None,"MIN_UNCERTAINTY_PRECISION":None}


def aggregate(results, tolerance=1):
    text={}
    for key in ("strict_wer","relaxed_wer","cer"):
        values={field:sum(r["text"][key][field] for r in results) for field in ("substitutions","deletions","insertions","reference_count")}
        values["error_rate"]=ratio(values["substitutions"]+values["deletions"]+values["insertions"],values["reference_count"])
        text[key]=values
    combine=lambda field:ratio(sum(r[field]["numerator"] for r in results),sum(r[field]["denominator"] for r in results))
    verbatim={kind:event_metric(*(sum(r["verbatim"][kind][key] for r in results) for key in ("reference_count","model_count","match_count"))) for kind in ("filler","repetition","false_start")}
    names={key:sum(r["proper_names"][key] for r in results) for key in ("name_reference_count","name_exact_match_count","name_error_count")}
    names["name_accuracy"]=ratio(names["name_exact_match_count"],names["name_reference_count"])
    return {"fixture_count":len(results),"duration_seconds":sum(r["duration_seconds"] for r in results),
            "reference_segments":sum(r["reference_segments"] for r in results),"model_segments":sum(r["model_segments"] for r in results),
            "text":text,"verbatim":verbatim,"omission":combine("omission"),"insertion":combine("insertion"),
            "golden_speech_coverage":combine("golden_speech_coverage"),
            "speaker_accuracy":ratio(sum(r["speaker"]["speaker_correct_segments"] for r in results),sum(r["speaker"]["speaker_attributed_segments"] for r in results)),
            "timestamp":timestamp_metrics([e for r in results for e in r["timestamp"]["errors"]],tolerance),
            "uncertainty":event_metric(*(sum(r["uncertainty"][key] for r in results) for key in ("reference_count","model_count","match_count"))),
            "proper_names":names,"critical_review_count":sum(q["severity"]=="CRITICAL" for r in results for q in r["review_queue"]),
            "high_review_count":sum(q["severity"]=="HIGH" for r in results for q in r["review_queue"])}


def check_thresholds(metrics, thresholds, profile):
    import math
    if profile not in ("evaluation-only","release-candidate") or set(thresholds)-set(THRESHOLDS):
        raise ValueError("THRESHOLD_CONFIG_INVALID")
    configured=dict(THRESHOLDS,**thresholds)
    if any(v is not None and (isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0) for v in configured.values()):
        raise ValueError("THRESHOLD_CONFIG_INVALID")
    values={"MAX_STRICT_WER":metrics["text"]["strict_wer"]["error_rate"]["value"],
            "MAX_RELAXED_WER":metrics["text"]["relaxed_wer"]["error_rate"]["value"],
            "MAX_CER":metrics["text"]["cer"]["error_rate"]["value"],
            "MIN_GOLDEN_SPEECH_COVERAGE":metrics["golden_speech_coverage"]["value"],
            "MIN_SPEAKER_ACCURACY":metrics["speaker_accuracy"]["value"],
            "MAX_MEDIAN_TIMESTAMP_ERROR_SECONDS":metrics["timestamp"]["median"],
            "MIN_UNCERTAINTY_PRECISION":metrics["uncertainty"]["precision"]["value"]}
    decisions={key:("DISABLED" if limit is None else "UNAVAILABLE" if values[key] is None else
                     "PASS" if (values[key]<=limit if key.startswith("MAX") else values[key]>=limit) else "FAIL") for key,limit in configured.items()}
    status="FAIL" if "FAIL" in decisions.values() else "PARTIAL" if any(v in ("DISABLED","UNAVAILABLE") for v in decisions.values()) else "PASS"
    return {"status":status,"profile":profile,"configured":configured,"decisions":decisions}


def release_gate(reviewed_count, live_completed_count, threshold_status, critical_reviews=0, mandatory_count=None):
    """Future Stage 7B supplies actual execution evidence; CLI 7A always passes 0."""
    if not reviewed_count: return {"status":"PARTIAL","reason":"NO_HUMAN_REVIEWED_GOLDEN_FIXTURES"}
    if live_completed_count<max(reviewed_count,mandatory_count or 0): return {"status":"PARTIAL","reason":"LIVE_PRODUCTION_E2E_INCOMPLETE"}
    if critical_reviews or threshold_status=="FAIL": return {"status":"FAIL","reason":"QUALITY_REVIEW_OR_THRESHOLD_FAILURE"}
    if threshold_status!="PASS": return {"status":"PARTIAL","reason":"MANDATORY_METRICS_OR_THRESHOLDS_UNAVAILABLE"}
    return {"status":"PASS","reason":"REVIEWED_LIVE_QUALITY_GATE_PASSED"}


def build_report(results, config, profile="evaluation-only", thresholds=None, errors=None):
    errors=errors or []
    reviewed=[r for r in results if r["review_status"] in REVIEWED]
    synthetic=[r for r in results if r["review_status"]=="SYNTHETIC_TEST"]
    qualified=aggregate(reviewed,config.golden_timestamp_tolerance_seconds)
    threshold_result=check_thresholds(qualified,thresholds or {},profile)
    gate=release_gate(len(reviewed),0,threshold_result["status"],qualified["critical_review_count"])
    queue=[item for r in results for item in r["review_queue"]]
    summary={"framework_status":"FAIL" if errors else "PASS","golden_corpus_status":"PARTIAL" if not reviewed else "AVAILABLE_OFFLINE",
             "live_status":"NOT_RUN","acoustic_qualification":"UNAVAILABLE","release_candidate_readiness":gate["status"],"reason":gate["reason"],
             "fixture_count":len(results),"human_reviewed_fixture_count":len(reviewed),"synthetic_fixture_count":len(synthetic),
             "top_review_flags":dict(sorted(Counter(flag for q in queue for flag in q["flags"]).items(),key=lambda item:(-item[1],item[0]))),
             "readiness_matrix":{"pipeline_execution":"NOT_RUN","committed_coverage":"NOT_RUN","text_fidelity":"UNAVAILABLE",
                "verbatim_preservation":"UNAVAILABLE","speaker_attribution":"UNAVAILABLE","timestamp_accuracy":"UNAVAILABLE",
                "uncertainty_handling":"UNAVAILABLE","long_audio_e2e":"NOT_RUN","resume_e2e":"PARTIAL"},
             "config":config.to_dict(),"thresholds":threshold_result,"errors":errors}
    return {"summary":summary,"metrics":{"reviewed_reference_comparison":qualified,"synthetic_framework_tests":aggregate(synthetic,config.golden_timestamp_tolerance_seconds)},
            "fixtures":results,"review_queue":queue}


def write_report(report, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for filename,key in [("summary.json","summary"),("metrics.json","metrics"),("fixtures.json","fixtures")]:
        (output/filename).write_text(json.dumps(report[key],ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"review_queue.jsonl").write_text("".join(json.dumps(q,ensure_ascii=False,allow_nan=False)+"\n" for q in report["review_queue"]),encoding="utf-8")
    summary=report["summary"]
    lines=["# Offline golden evaluation", "",f"Framework: {summary['framework_status']}",f"Golden corpus: {summary['golden_corpus_status']}",
           "Live: NOT_RUN", "Acoustic qualification: UNAVAILABLE",f"Release: {summary['release_candidate_readiness']}",f"Reason: {summary['reason']}",
           "", "Synthetic scores test the evaluator, not acoustic quality.", "", "| Fixture | Review status | Strict WER | Relaxed WER | CER |", "|---|---|---|---|---|"]
    for result in report["fixtures"]:
        lines.append("| "+" | ".join([result["fixture_id"],result["review_status"]]+[str(result["text"][key]["error_rate"]["value"]) for key in ("strict_wer","relaxed_wer","cer")])+" |")
    lines.extend(["","## Review examples",""])
    for item in report["review_queue"][:20]:
        lines.extend([f"- {item['fixture_id']} @ {item['start_seconds']}: {', '.join(item['flags'])}",
                      f"  Golden: {item['golden_text'][:240]}",f"  Model: {item['model_text'][:240]}"])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
