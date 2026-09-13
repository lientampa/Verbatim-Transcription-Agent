"""Stage 7A offline CLI. No provider execution, imports or live switch."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from src.golden_evaluation.fixtures import load_fixture, GoldenFixtureError
from src.golden_evaluation.alignment import EvaluationConfig
from src.golden_evaluation.evaluator import evaluate
from src.golden_evaluation.reporting import build_report, write_report
from src.model_policy import is_model_allowed


def run(fixture_dir, output_dir, fixture=None, profile="evaluation-only", thresholds=None, config=None):
    config=config or EvaluationConfig()
    fixture_dir=Path(fixture_dir)
    results, errors, seen=[],[],set()
    paths=sorted(fixture_dir.rglob("reference.json"))
    for path in paths:
        try:
            reference=load_fixture(path)
            if fixture and reference["fixture_id"]!=fixture: continue
            if reference["fixture_id"] in seen: raise GoldenFixtureError("GOLDEN_FIXTURE_INVALID: duplicate fixture_id")
            seen.add(reference["fixture_id"])
            if reference["review"]["status"]=="EXAMPLE_ONLY": continue
            model_path=path.with_name("model_output.json")
            if not model_path.exists():
                errors.append({"fixture_id":reference["fixture_id"],"reason":"MODEL_OUTPUT_UNAVAILABLE"}); continue
            output=json.loads(model_path.read_text(encoding="utf-8"))
            if not isinstance(output, dict):
                raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: output must be an object")
            if output.get("source_fingerprint")!=reference["source"]["fingerprint_sha256"]:
                raise GoldenFixtureError("GOLDEN_SOURCE_MISMATCH")
            provenance=output.get("provenance",{})
            if not isinstance(provenance, dict):
                raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: provenance must be an object")
            actual=provenance.get("actual_models_used",[])
            if not isinstance(actual, list) or any(not isinstance(name, str) or not name.strip() for name in actual):
                raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: actual_models_used must be a list of names")
            if any(not is_model_allowed(name) for name in actual):
                raise GoldenFixtureError("MODEL_POLICY_VIOLATION: LITE_MODEL_DISABLED")
            result=evaluate(reference,output["segments"],output["source_fingerprint"],config,provenance)
            results.append(result)
        except (ValueError,OSError,KeyError,TypeError) as exc:
            errors.append({"fixture":str(path),"reason":str(exc)})
    if fixture and fixture not in seen: errors.append({"fixture_id":fixture,"reason":"FIXTURE_NOT_FOUND"})
    report=build_report(results,config,profile,thresholds,errors)
    write_report(report,output_dir)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline Stage 7A metrics; never invokes Gemini")
    parser.add_argument("--fixture-dir",default=os.getenv("GOLDEN_FIXTURE_DIR","tests/golden_audio"))
    parser.add_argument("--fixture")
    parser.add_argument("--output-dir",default="evaluation/run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--threshold-profile",choices=["evaluation-only","release-candidate"],default="evaluation-only")
    parser.add_argument("--thresholds",type=Path,help="JSON thresholds; no release limits are assumed")
    parser.add_argument("--alignment-max-time-delta",type=float,default=15)
    parser.add_argument("--alignment-min-text-similarity",type=float,default=.35)
    parser.add_argument("--timestamp-tolerance",type=float,default=float(os.getenv("GOLDEN_TIMESTAMP_TOLERANCE_SECONDS","1")))
    args=parser.parse_args(argv)
    config=EvaluationConfig(alignment_max_time_delta_seconds=args.alignment_max_time_delta,alignment_min_text_similarity=args.alignment_min_text_similarity,golden_timestamp_tolerance_seconds=args.timestamp_tolerance)
    thresholds=json.loads(args.thresholds.read_text(encoding="utf-8")) if args.thresholds else None
    report=run(args.fixture_dir,args.output_dir,args.fixture,args.threshold_profile,thresholds,config)
    print("FRAMEWORK_STATUS="+report["summary"]["framework_status"]+" GOLDEN_CORPUS_STATUS="+report["summary"]["golden_corpus_status"]+" LIVE_STATUS=NOT_RUN RELEASE_STATUS=PARTIAL")
    return 1 if report["summary"]["framework_status"]=="FAIL" else 0


if __name__=="__main__":
    raise SystemExit(main())
