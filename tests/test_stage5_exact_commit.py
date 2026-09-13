"""Durable manifest, exact production restart, and physical cache isolation."""
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.checkpoint import CheckpointManager, CheckpointError, CheckpointStatus
from src.source_identity import SourceIdentity
from src.main import run_pipeline
from src.gemini_client import GeminiClient
from tests.test_stage4_audio_retry import run_audio_case


def audio_manager(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"AAAA")
    manager = CheckpointManager(tmp_path / "checkpoint.json")
    manager.create_new_job(audio.name, source_identity=SourceIdentity.capture(audio, 1800))
    return manager, audio


def commit(manager, start=0, end=600, number=1):
    block_id = f"BLOCK_{number:03d}"
    return manager.commit_block(block_id, 198, [{"source_index": 198, "timestamp": "00:08:54", "text": "verbatim"}],
        metric=dict(source_mode="AUDIO", segment_count=1, block_id=block_id,
            actual_start_offset=start, actual_end_offset=end, received_last_source_index=218,
            coverage={"decision": "COVERAGE_PASS"}, fidelity_decision="ACCEPT"))


def test_duplicate_commit_is_durable_noop(tmp_path):
    manager, _ = audio_manager(tmp_path)
    commit(manager)
    before = manager.checkpoint_path.read_bytes()
    commit(manager)
    assert manager.checkpoint_path.read_bytes() == before
    assert len(manager.current_data.block_metrics) == len(manager.current_data.confirmed_segments) == 1
    assert manager.current_data.next_audio_start_us == 600_000000
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        commit(manager, end=900)
    assert manager.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("start,reason", [(590,"OVERLAP"),(620,"GAP")])
def test_gap_overlap_cannot_commit_or_load(tmp_path, start, reason):
    manager, _ = audio_manager(tmp_path)
    commit(manager)
    before = manager.checkpoint_path.read_bytes()
    with pytest.raises(CheckpointError, match=reason):
        commit(manager, start=start, end=1200, number=2)
    assert manager.checkpoint_path.read_bytes() == before
    commit(manager, start=600, end=1200, number=2)
    data = manager.current_data.to_dict()
    data["block_metrics"][1]["actual_start_offset"] = start
    manager.checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match=reason):
        CheckpointManager(manager.checkpoint_path).load()


@pytest.mark.parametrize("stage", ["dump", "fsync"])
def test_partial_write_keeps_previous_checkpoint(tmp_path, monkeypatch, stage):
    manager, _ = audio_manager(tmp_path)
    commit(manager)
    before = manager.checkpoint_path.read_bytes()
    def fail_dump(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("injected temp write failure")
    def fail_sync(fd):
        raise OSError("injected fsync failure")
    monkeypatch.setattr("src.checkpoint.json.dump" if stage == "dump" else "src.checkpoint.os.fsync", fail_dump if stage == "dump" else fail_sync)
    with pytest.raises(CheckpointError):
        commit(manager, start=600, end=1200, number=2)
    assert manager.checkpoint_path.read_bytes() == before
    assert CheckpointManager(manager.checkpoint_path).load().next_audio_start_us == 600_000000
    assert manager.current_data.next_audio_start_us == 600_000000


def test_content_identity_ignores_rename_and_detects_same_stat_replacement(tmp_path):
    manager, audio = audio_manager(tmp_path)
    first = SourceIdentity.capture(audio, 1800)
    assert SourceIdentity.capture(audio, 1800).fingerprint == first.fingerprint
    renamed = audio.rename(tmp_path / "renamed.wav")
    assert SourceIdentity.capture(renamed, 1800).fingerprint == first.fingerprint
    stat = renamed.stat()
    renamed.write_bytes(b"AAAB")
    os.utime(renamed, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = SourceIdentity.capture(renamed, 1800)
    assert changed.fingerprint != first.fingerprint
    with pytest.raises(CheckpointError, match="CHECKPOINT_SOURCE_MISMATCH"):
        manager.validate_resume(changed)


@pytest.mark.parametrize("change", [lambda d: d.update(schema_version=99), lambda d: d.update(source_mode="TEXT_TIMESTAMP"), lambda d: d.update(next_audio_start_us=1801_000000)])
def test_version_mode_eof_rejected(tmp_path, change):
    manager, _ = audio_manager(tmp_path)
    data = manager.current_data.to_dict()
    change(data)
    manager.checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError):
        CheckpointManager(manager.checkpoint_path).load()


def test_text_source_index_semantics_unchanged(tmp_path):
    manager = CheckpointManager(tmp_path / "text.json")
    manager.create_new_job("source.txt")
    manager.commit_block("BLOCK_001", 2, [{"source_index":1,"text":"a"},{"source_index":2,"text":"b"}])
    manager.commit_block("BLOCK_001", 2, [{"source_index":1,"text":"a"},{"source_index":2,"text":"b"}])
    data = CheckpointManager(manager.checkpoint_path).load()
    assert data.source_mode == "TEXT_TIMESTAMP"
    assert data.next_source_index == 3 and len(data.confirmed_segments) == 2
    assert data.next_audio_start_us == 0


@pytest.mark.parametrize("interrupt_at", ["before_second_commit", "after_first_commit", "failed_second_shrink"])
def test_production_max_first_exact_resume(tmp_path, monkeypatch, interrupt_at):
    observed = []
    control = {"resuming": False}
    original_commit = CheckpointManager.commit_block
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=1200,
            initial_target_input_tokens=3000, coverage_max_generations=3)
        monkeypatch.setattr("src.main.load_config", lambda **kw: config)
        original_generate = GeminiClient.generate_transcription
        def generate(self, gemini_file, **kwargs):
            observed.append(list(gemini_file.bounds))
            data = json.loads(original_generate(self, gemini_file=gemini_file, **kwargs))
            start, end = gemini_file.bounds
            if interrupt_at == "failed_second_shrink" and start == 600 and not control["resuming"]:
                if end <= 900:
                    raise KeyboardInterrupt("failed shrink chain never committed")
                seconds = int(start + 1)
            else:
                seconds = int(1 if start == 0 and end == 1200 else end - 6)
            data["segments"][0]["timestamp"] = f"{seconds//60:02d}:{seconds%60:02d}"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze", lambda self,path,offset,duration,threshold: duration)
        def interrupted_commit(self, **kw):
            if interrupt_at == "before_second_commit" and kw["block_id"] == "BLOCK_002":
                raise KeyboardInterrupt("validated but uncommitted")
            value = original_commit(self, **kw)
            if interrupt_at == "after_first_commit":
                raise KeyboardInterrupt("durably committed")
            return value
        monkeypatch.setattr(CheckpointManager, "commit_block", interrupted_commit)
    with pytest.raises(KeyboardInterrupt):
        run_audio_case(tmp_path, monkeypatch, setup=setup, fail_count=0, total_duration=1800)
    path = tmp_path / "state" / "checkpoint.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["next_audio_start_us"] == 600_000000
    assert observed[:2] == [[0,1200],[0,600]]
    if interrupt_at == "before_second_commit":
        assert observed[-1] == [600,1800]
    monkeypatch.setattr(CheckpointManager, "commit_block", original_commit)
    observed.clear()
    control["resuming"] = True
    assert run_pipeline(base_dir=tmp_path) == 0
    assert observed == [[600,1800]]
    final = json.loads(path.read_text(encoding="utf-8"))
    assert [m["actual_end_offset"] for m in final["block_metrics"]] == [600,1800]
    assert len(final["confirmed_segments"]) == 2


def test_eof_recovers_missing_outputs_without_provider(tmp_path, monkeypatch, capsys):
    result, _, _, _, cp = run_audio_case(tmp_path, monkeypatch, fail_count=0)
    assert result == 0
    path = tmp_path / "state" / "checkpoint.json"
    cp["status"] = "PROCESSING"  # Crash after final durable commit, before rendering.
    path.write_text(json.dumps(cp))
    (tmp_path / "output" / "transcript.txt").unlink()
    monkeypatch.setattr(GeminiClient, "__init__", lambda *a, **kw: pytest.fail("provider initialized at EOF"))
    assert run_pipeline(base_dir=tmp_path) == 0
    assert "status=SOURCE_COMPLETE" in capsys.readouterr().out
    assert (tmp_path / "output" / "transcript.txt").exists()


@pytest.mark.parametrize("expired", [True, False])
def test_expired_or_missing_upload_reuses_exact_physical_slice(tmp_path, monkeypatch, expired):
    client = GeminiClient.__new__(GeminiClient)
    calls = []
    path = tmp_path / "slice.wav"
    def get(**kwargs):
        exc = RuntimeError("missing file")
        exc.code = 404
        raise exc
    client.client = SimpleNamespace(files=SimpleNamespace(get=get))
    monkeypatch.setattr(client, "upload_audio", lambda audio_path: calls.append(audio_path) or "fresh")
    remote = SimpleNamespace(name="files/old", expiration_time=datetime.now(timezone.utc)-timedelta(seconds=1) if expired else None)
    assert client.refresh_audio_upload(remote, path) == "fresh"
    assert calls == [path]


def test_production_orchestrator_changed_source_has_distinct_slice_key(tmp_path, monkeypatch):
    import hashlib
    from src.block_builder import DabbAudioOrchestrator, AudioBlockBuilder
    from src.config import load_config
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    config = replace(load_config(tmp_path), next_target_policy="MAX_FIRST", max_duration_seconds=600)
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"AAAA")
    monkeypatch.setattr("src.block_builder._find_binary", lambda name: name)
    monkeypatch.setattr(AudioBlockBuilder, "get_duration", lambda self: 1800)
    generated = []
    def ffmpeg(cmd, **kw):
        Path(cmd[-1]).write_bytes(audio.read_bytes())
        generated.append(cmd)
        return SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr("src.block_builder.subprocess.run", ffmpeg)
    first = next(DabbAudioOrchestrator(audio, config).iter_adaptive_blocks(start_from_seconds=600)).source_block
    assert (first.start_time_seconds, first.end_time_seconds) == (600,1200)
    old_bytes = first.file_path.read_bytes()
    audio.write_bytes(b"AAAB")
    second = next(DabbAudioOrchestrator(audio, config).iter_adaptive_blocks(start_from_seconds=600)).source_block
    assert second.file_path != first.file_path and len(generated) == 2
    assert first.file_path.read_bytes() == old_bytes
    for block in (first, second):
        manifest = json.loads(block.file_path.with_suffix(".json").read_text())
        key = hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True).encode()).hexdigest()
        assert block.file_path.stem == key
        assert manifest["identity"]["start_us"] == 600_000000
        assert manifest["identity"]["end_us"] == 1200_000000
