"""Synthetic evaluator tests, never acoustic evidence or provider execution."""
import json
from copy import deepcopy
from pathlib import Path
import unicodedata
import pytest

from src.golden_evaluation.fixtures import validate_reference, load_fixture, GoldenFixtureError
from src.golden_evaluation.text_metrics import text_metrics, edit_counts, tokens
from src.golden_evaluation.alignment import EvaluationConfig
from src.golden_evaluation.evaluator import evaluate
from src.golden_evaluation.reporting import aggregate, release_gate, check_thresholds, THRESHOLDS
from src.evaluate_golden import run, main


def segment(text="anh nói rồi",start=10,speaker="A",**extra):
    return dict(start_seconds=start,speaker=speaker,text=text,uncertain="[không rõ]" in text,**extra)


def reference(segments=None):
    return dict(schema_version="1",fixture_id="synthetic",language="vi",source_mode="AUDIO",
                source=dict(fingerprint_sha256="0"*64,duration_seconds=4000),review=dict(status="SYNTHETIC_TEST"),segments=segments if segments is not None else [segment()])


def score(golden, model, **kwargs):
    return evaluate(reference([segment(golden)]),[segment(model)],**kwargs)


def flags(result):
    return {f for item in result["review_queue"] for f in item["flags"]}


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail("Stage 7A invoked a provider")
    monkeypatch.setattr("google.genai.Client",forbidden)
    monkeypatch.setenv("RUN_LIVE_GEMINI_E2E","1")  # Even accidental opt-in cannot enable 7A.


@pytest.mark.parametrize("change",[
    lambda r:r.update(schema_version="2"),lambda r:r.update(language="en"),lambda r:r.update(source_mode="TEXT_TIMESTAMP"),
    lambda r:r["source"].update(fingerprint_sha256="bad"),lambda r:r["source"].update(duration_seconds=0),
    lambda r:r["segments"][0].update(start_seconds=4001),lambda r:r["segments"][0].update(start_seconds=float("nan")),
    lambda r:r["segments"][0].update(speaker=" "),lambda r:r["segments"][0].update(text=""),
    lambda r:r["segments"][0].update(uncertain=True),lambda r:r["review"].update(status="AUTO_REVIEWED"),
    lambda r:r.update(segments=[segment(start=20),segment(start=10)])])
def test_invalid_reference(change):
    r=reference(); change(r)
    with pytest.raises(GoldenFixtureError,match="GOLDEN_FIXTURE_INVALID"): validate_reference(r)


def test_valid_reference_nonmutation_and_source_binding(tmp_path):
    from src.source_identity import sha256_file
    r=reference(); before=deepcopy(r)
    assert validate_reference(r)==r and r==before
    with pytest.raises(GoldenFixtureError,match="GOLDEN_SOURCE_MISMATCH"): validate_reference(r,"1"*64)
    # Temporary synthetic byte fixture simulates an external reviewed-file record.
    audio=tmp_path/"private.audio"; audio.write_bytes(b"synthetic bytes, not real audio")
    r["source"].update(audio_path=str(audio),fingerprint_sha256=sha256_file(audio))
    path=tmp_path/"reference.json"; path.write_text(json.dumps(r))
    assert load_fixture(path)==r
    audio.write_bytes(b"changed")
    with pytest.raises(GoldenFixtureError,match="GOLDEN_SOURCE_MISMATCH"): load_fixture(path)


@pytest.mark.parametrize("golden,model,counts",[("anh nói rồi","anh nói rồi",(0,0,0)),("anh đã nói rồi","anh nói rồi",(0,1,0)),("anh nói rồi","anh đã nói rồi",(0,0,1)),("anh đi Hà Nội","anh đi Đà Nẵng",(2,0,0))])
def test_edit_counts(golden,model,counts):
    metric=text_metrics(golden,model)["relaxed_wer"]
    assert tuple(metric[k] for k in ("substitutions","deletions","insertions"))==counts
    assert metric["error_rate"]["value"]==sum(counts)/len(tokens(golden,True))


def test_cer_unicode_empty_and_punctuation():
    assert text_metrics("Chí",unicodedata.normalize("NFD","Chí"))["cer"]["error_rate"]["value"]==0
    assert text_metrics("Chí","Chi")["cer"]["error_rate"]["value"]==1/3
    metrics=text_metrics("Dạ, em hiểu.","Dạ em hiểu")
    assert metrics["strict_wer"]["error_rate"]["value"]>0 and metrics["relaxed_wer"]["error_rate"]["value"]==0
    empty=edit_counts([],['x'])
    assert empty["insertions"]==1 and empty["error_rate"]["value"] is None
    assert tokens("cái [không rõ] rồi",True)==["cái","[không rõ]","rồi"]


def test_perfect_verbatim_and_deterministic_nonmutation():
    r=reference([segment("Ờ, anh... anh chưa biết.")]); m=deepcopy(r["segments"]); before=deepcopy((r,m))
    result=evaluate(r,m)
    assert result==evaluate(r,m) and (r,m)==before
    for key in ("strict_wer","relaxed_wer","cer"): assert result["text"][key]["error_rate"]["value"]==0
    for key in ("filler","repetition","false_start"): assert result["verbatim"][key]["recall"]["value"]==1


@pytest.mark.parametrize("g,m,kind",[("Ờ, em nghĩ là được.","Em nghĩ là được.","filler"),("anh anh nói rồi","anh nói rồi","repetition"),("em định... à không, anh định nói","anh định nói","false_start")])
def test_verbatim_loss(g,m,kind):
    result=score(g,m)
    assert result["verbatim"][kind]["recall"]["value"]==0
    assert kind.upper()+"_LOSS" in flags(result)


def test_explicit_filler_overrides_heuristic():
    r=reference([segment("thì anh nói",events=[dict(type="filler",text="thì")])])
    result=evaluate(r,[segment("anh nói")])
    assert result["verbatim"]["filler"]["reference_count"]==1
    assert result["verbatim"]["filler"]["recall"]["value"]==0
    assert score("thì anh nói","thì anh nói")["verbatim"]["filler"]["reference_count"]==0


@pytest.mark.parametrize("g,m,flag",[("cái đó [không rõ] rồi","cái đó bị hỏng rồi","UNCERTAIN_OVERGUESS"),("cái đó bị hỏng rồi","cái đó [không rõ] rồi","FALSE_UNCERTAINTY")])
def test_uncertainty_disagreement(g,m,flag):
    assert flag in flags(score(g,m))


def test_uncertainty_match():
    result=score("cái đó [không rõ] rồi","cái đó [không rõ] rồi")
    assert result["uncertainty"]["precision"]["value"]==result["uncertainty"]["recall"]["value"]==1


def test_timestamp_exact_distribution():
    r=reference([segment("một",10),segment("hai",20),segment("ba",30)])
    m=[segment("một",10.5),segment("hai",22),segment("ba",29)]
    metric=evaluate(r,m)["timestamp"]
    assert metric["errors"]==[.5,2,1] and metric["median"]==1 and metric["p90"]==2 and metric["max"]==2
    assert metric["within_1s_rate"]["value"]==2/3 and metric["within_2s_rate"]["value"]==1


def test_anonymous_mapping_and_under_identification():
    r=reference([segment("một",10,"Anh Nam"),segment("hai",20,"Chị Hoa")])
    result=evaluate(r,[segment("một",10,"Người nói 2"),segment("hai",20,"Người nói 1")])
    assert result["speaker"]["speaker_accuracy"]["value"]==1
    assert result["speaker"]["anonymous_mapping"]=={"Người nói 1":"Chị Hoa","Người nói 2":"Anh Nam"}
    assert "SPEAKER_UNDER_IDENTIFIED" in flags(result)


def test_cross_block_swap_and_fragmentation():
    r=reference([segment("một",10,"A"),segment("hai",1300,"B"),segment("ba",2600,"A")])
    m=[segment("một",10,"A"),segment("hai",1300,"A"),segment("ba",2600,"B")]
    result=evaluate(r,m)
    assert result["speaker"]["speaker_accuracy"]["value"]==1/3
    assert {"SPEAKER_SWAP_REVIEW","SPEAKER_FRAGMENTATION"}<=flags(result)
    assert evaluate(r,r["segments"])["speaker"]["speaker_accuracy"]["value"]==1


def test_name_and_named_speaker_diagnostics():
    r=reference([segment("Nguyễn Chí Thanh",proper_names=["Nguyễn Chí Thanh"])])
    result=evaluate(r,[segment("Nguyễn Chi Thanh",speaker="Anh Hùng")])
    assert result["proper_names"]["name_error_count"]==1
    assert {"PROPER_NAME_MISMATCH","SPEAKER_NAME_HALLUCINATION"}<=flags(result)


def test_non_speech_and_unresolved():
    r=reference(); r["regions"]=[dict(start_seconds=30,end_seconds=40,type="silence",authoritative=True)]
    result=evaluate(r,[segment("nội dung lạ",35)])
    assert {"ALIGNMENT_UNRESOLVED","POSSIBLE_NON_SPEECH_HALLUCINATION"}<=flags(result)
    assert result["timestamp"]["median"] is None and result["golden_speech_coverage"]["value"]==0


@pytest.mark.parametrize("reverse",[False,True])
def test_grouped_alignment(reverse):
    one=[segment("anh đã nói rồi")]
    two=[segment("anh đã",10),segment("nói rồi",11)]
    g,m=(two,one) if reverse else (one,two)
    result=evaluate(reference(g),m)
    assert len(result["alignment"])==1 and result["alignment"][0]["resolved"]
    assert result["text"]["strict_wer"]["error_rate"]["value"]==0
    assert result["timestamp"]["count"]==1


def test_same_time_preserves_order():
    r=reference([segment("một",10,"A"),segment("hai",10,"B")])
    result=evaluate(r,r["segments"])
    assert result["alignment"]==[{"reference_indices":[0],"model_indices":[0],"resolved":True},{"reference_indices":[1],"model_indices":[1],"resolved":True}]


def test_weighted_aggregate():
    results=[score("anh đã nói rồi","anh nói rồi"),score("một hai ba bốn năm sáu","một hai ba bốn năm sáu")]
    metrics=aggregate(results)
    assert metrics["text"]["strict_wer"]["error_rate"]["value"]==.1
    assert metrics["golden_speech_coverage"]["value"]==.9


def test_gate_requires_review_and_live_and_thresholds():
    assert release_gate(0,10,"PASS")["status"]=="PARTIAL"
    assert release_gate(1,0,"PASS")["status"]=="PARTIAL"
    assert release_gate(1,1,"PARTIAL")["status"]=="PARTIAL"
    assert release_gate(1,1,"FAIL")["status"]=="FAIL"
    assert release_gate(1,1,"PASS",1)["status"]=="FAIL"
    assert release_gate(1,1,"PASS")["status"]=="PASS"  # Gate simulation, not acoustic evidence.


def test_thresholds_are_explicit_and_null_by_default():
    metrics=aggregate([score("anh nói rồi","anh nói rồi")])
    assert check_thresholds(metrics,{},"release-candidate")["status"]=="PARTIAL"
    assert check_thresholds(metrics,{"MAX_STRICT_WER":0},"evaluation-only")["decisions"]["MAX_STRICT_WER"]=="PASS"
    with pytest.raises(ValueError): check_thresholds(metrics,{"unknown":1},"evaluation-only")


def test_empty_and_synthetic_cli_offline(tmp_path):
    empty=tmp_path/"empty"; empty.mkdir()
    result=run(empty,tmp_path/"empty_report")
    assert result["summary"]["framework_status"]=="PASS"
    assert result["summary"]["reason"]=="NO_HUMAN_REVIEWED_GOLDEN_FIXTURES"
    root=Path(__file__).parent/"golden_audio"
    assert main(["--fixture-dir",str(root),"--output-dir",str(tmp_path/"report")])==0
    summary=json.loads((tmp_path/"report"/"summary.json").read_text(encoding="utf-8"))
    assert summary["human_reviewed_fixture_count"]==0 and summary["synthetic_fixture_count"]==1
    assert summary["release_candidate_readiness"]=="PARTIAL" and summary["live_status"]=="NOT_RUN"
    metrics=json.loads((tmp_path/"report"/"metrics.json").read_text(encoding="utf-8"))
    assert metrics["reviewed_reference_comparison"]["text"]["strict_wer"]["error_rate"]["value"] is None
    assert set(p.name for p in (tmp_path/"report").iterdir())=={"summary.json","metrics.json","fixtures.json","review_queue.jsonl","report.md"}


def test_pending_review_excluded_and_lite_rejected(tmp_path):
    directory=tmp_path/"fixture"; directory.mkdir()
    r=reference(); r["review"]["status"]="PENDING_MANUAL_REVIEW"
    (directory/"reference.json").write_text(json.dumps(r))
    model=dict(source_fingerprint="0"*64,segments=r["segments"],provenance={"actual_models_used":[]})
    (directory/"model_output.json").write_text(json.dumps(model))
    result=run(directory,tmp_path/"report")
    assert result["summary"]["human_reviewed_fixture_count"]==0
    model["provenance"]["actual_models_used"]=["gemini-3.5-flash-lite"]
    (directory/"model_output.json").write_text(json.dumps(model))
    result=run(directory,tmp_path/"report2")
    assert result["summary"]["framework_status"]=="FAIL"
    assert "LITE_MODEL_DISABLED" in result["summary"]["errors"][0]["reason"]


def test_proper_name_requires_word_boundary():
    result=evaluate(reference([segment("Chí nói rồi",proper_names=["Chí"])]),[segment("Chính nói rồi")])
    assert result["proper_names"]["name_exact_match_count"]==0


def test_model_event_annotations_cannot_hide_inserted_fillers():
    result=evaluate(reference([segment("ờ anh nói rồi")]),[segment("ờ ừ anh nói rồi",events=[])])
    assert result["verbatim"]["filler"]["model_count"]==2
    assert result["verbatim"]["filler"]["precision"]["value"]==.5


def test_reviewed_offline_report_never_qualifies():
    from src.golden_evaluation.reporting import build_report
    r=reference(); r["review"]["status"]="HUMAN_REVIEWED"
    report=build_report([evaluate(r,r["segments"])],EvaluationConfig())
    assert report["summary"]["human_reviewed_fixture_count"]==1
    assert report["summary"]["live_status"]=="NOT_RUN"
    assert report["summary"]["acoustic_qualification"]=="UNAVAILABLE"
    assert report["summary"]["release_candidate_readiness"]=="PARTIAL"


def test_reviewed_loader_requires_audio(tmp_path):
    r=reference(); r["review"]["status"]="HUMAN_REVIEWED"
    path=tmp_path/"reference.json"; path.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(GoldenFixtureError): load_fixture(path)


def test_example_only_excluded_without_model_output(tmp_path):
    r=reference(); r["review"]["status"]="EXAMPLE_ONLY"
    (tmp_path/"reference.json").write_text(json.dumps(r),encoding="utf-8")
    result=run(tmp_path,tmp_path/"report")
    assert result["summary"]["fixture_count"]==0
    assert result["summary"]["framework_status"]=="PASS"


@pytest.mark.parametrize("model",[None,{},[None],["bad"]])
def test_malformed_canonical_segments_rejected(model):
    with pytest.raises(GoldenFixtureError,match="MODEL_TRANSCRIPT_INVALID"):
        evaluate(reference(),model)
