"""Source-safe resume, semantic checkpoint rejection, and reusable slice identity."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.checkpoint import CheckpointManager, CheckpointError, CheckpointStatus
from src.source_identity import SourceIdentity
from src.main import run_pipeline
from src.block_builder import AudioBlockBuilder
from tests.test_stage4_audio_retry import run_audio_case


def seed(path, source_path, duration=1500, ends=(360, 780, 1020)):
    source = SourceIdentity.capture(source_path, duration)
    manager = CheckpointManager(path)
    manager.create_new_job(source_path.name, source_identity=source)
    start = 0
    for index, end in enumerate(ends, 1):
        block_id = f"BLOCK_{index:03d}"
        manager.commit_block(block_id, index, [{"source_index": index, "text": "Vâng."}],
            metric=dict(block_id=block_id, actual_start_offset=start, actual_end_offset=end,
                        coverage={"decision": "COVERAGE_PASS"},
                        fidelity_decision="ACCEPT"), adaptive_state={"target_tokens": 1200})
        start = end
    return manager, source


@pytest.mark.parametrize("adaptive", [{"target_tokens": 1200}, {}, {"target_tokens": "invalid"}])
def test_multiblock_direct_restart_and_safe_adaptive_fallback(tmp_path, monkeypatch, adaptive):
    def setup(config):
        manager, _ = seed(config.checkpoint_file_path, config.audio_dir / "source.wav")
        manager.current_data.adaptive_state = adaptive
        manager.save()
    result, slices, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch,
        fail_count=0, setup=setup, total_duration=1500)
    assert result == 0
    assert slices[0][0] == 1020
    assert calls[0][0].bounds[0] == 1020
    assert 'start=1020' in calls[0][1]
    assert '- first_source_index: 4' in calls[0][1]
    assert '- block_id: "BLOCK_004"' in calls[0][1]
    assert checkpoint["next_audio_start_us"] == 1500_000000


def test_completed_source_replaced_in_place_is_rejected_without_writing(tmp_path, monkeypatch):
    result, _, _, calls, _ = run_audio_case(tmp_path, monkeypatch, fail_count=0)
    assert result == 0
    checkpoint = tmp_path / "state" / "checkpoint.json"
    original = checkpoint.read_bytes()
    count = len(calls)
    # Same name and byte count, different content.
    (tmp_path / "audio" / "source.wav").write_bytes(b"SOURCE")
    assert run_pipeline(base_dir=tmp_path) == 1
    assert checkpoint.read_bytes() == original
    assert len(calls) == count


def test_matching_completed_source_can_be_renamed_and_not_reprocessed(tmp_path, monkeypatch):
    result, _, _, calls, _ = run_audio_case(tmp_path, monkeypatch, fail_count=0)
    assert result == 0
    (tmp_path / "audio" / "source.wav").rename(tmp_path / "audio" / "renamed.wav")
    count = len(calls)
    assert run_pipeline(base_dir=tmp_path) == 0
    assert len(calls) == count


def test_partial_checkpoint_survives_source_rename(tmp_path, monkeypatch):
    def setup(config):
        audio = config.audio_dir / "source.wav"
        seed(config.checkpoint_file_path, audio, duration=900, ends=(360,))
        audio.rename(config.audio_dir / "renamed.wav")
    result, slices, _, calls, _ = run_audio_case(tmp_path, monkeypatch, setup=setup, fail_count=0)
    assert result == 0
    assert slices[0][0] == calls[0][0].bounds[0] == 360


def test_invalid_checkpoint_pipeline_never_slices_or_rewrites(tmp_path, monkeypatch):
    def setup(config):
        manager, _ = seed(config.checkpoint_file_path, config.audio_dir / "source.wav", duration=900, ends=(360,))
        data = manager.current_data.to_dict()
        data["next_audio_start_us"] = 100_000000
        config.checkpoint_file_path.write_text(json.dumps(data))
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup)
    assert result == 1
    assert not slices and not uploads and not calls
    assert checkpoint["next_audio_start_us"] == 100_000000


def test_output_and_cached_files_do_not_imply_progress(tmp_path, monkeypatch):
    def setup(config):
        config.output_transcript_path.write_text("previous unrelated output")
        (config.cache_dir / "BLOCK_001.wav").write_bytes(b"stale")
    result, _, _, calls, _ = run_audio_case(tmp_path, monkeypatch, setup=setup, fail_count=0)
    assert result == 0
    assert calls[0][0].bounds[0] == 0


@pytest.mark.parametrize("change", [
    lambda d: d.update(next_audio_start_us=700_000000),
    lambda d: d.update(next_audio_start_us=-1),
    lambda d: d.update(next_audio_start_us=1600_000000),
    lambda d: d.update(next_audio_start_us=float("nan")),
    lambda d: d["block_metrics"][-1].update(actual_end_offset=float("inf")),
    lambda d: d["block_metrics"][-1].update(actual_start_offset=-10),
    lambda d: d.update(current_block_id="BLOCK_999"),
    lambda d: d.update(next_source_index=99),
    lambda d: d.update(status="COMPLETED"),
    lambda d: d["source_identity"].update(fingerprint="bad"),
    lambda d: d.update(last_confirmed_audio_end=999),
])
def test_corrupt_checkpoint_rejected_and_bytes_preserved(tmp_path, change):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"source")
    path = tmp_path / "checkpoint.json"
    seed(path, audio)
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(CheckpointError, match="CHECKPOINT_INVALID"):
        CheckpointManager(path).load()
    assert path.read_bytes() == before


def test_legacy_resume_is_explicitly_untrusted(tmp_path, monkeypatch):
    def setup(config):
        config.checkpoint_file_path.write_text(json.dumps(dict(job_id="old", status="COMPLETED")))
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup)
    assert result == 1
    assert not slices and not uploads and not calls
    assert checkpoint == dict(job_id="old", status="COMPLETED")


def test_atomic_commit_failure_keeps_persisted_and_in_memory_resume(tmp_path, monkeypatch):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"source")
    path = tmp_path / "checkpoint.json"
    manager, source = seed(path, audio, ends=(360,))
    before = path.read_bytes()
    original_replace = Path.replace
    def fail_replace(self, target):
        if Path(target) == path:
            raise OSError("simulated disk failure")
        return original_replace(self, target)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(CheckpointError):
        manager.commit_block("BLOCK_002", 2, [{"source_index": 2, "text": "Vâng."}],
            metric=dict(block_id="BLOCK_002", actual_start_offset=360, actual_end_offset=780,
                        fidelity_decision="ACCEPT"))
    assert path.read_bytes() == before
    assert manager.current_data.next_audio_start_us == 360_000000
    assert CheckpointManager(path).load().next_audio_start_us == 360_000000


def test_duration_mismatch_is_not_silently_accepted(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"source")
    manager, source = seed(tmp_path / "checkpoint.json", audio)
    with pytest.raises(CheckpointError, match="source duration mismatch"):
        manager.validate_resume(SourceIdentity.capture(audio, 1600))


def test_cache_source_boundaries_sidecar_and_artifact_are_verified(tmp_path, monkeypatch):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"AAAA")
    monkeypatch.setattr("src.block_builder._find_binary", lambda name: name)
    monkeypatch.setattr(AudioBlockBuilder, "get_duration", lambda self: 1000)
    commands = []
    def ffmpeg(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(audio.read_bytes() + repr(cmd[2:-1]).encode())
        return SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr("src.block_builder.subprocess.run", ffmpeg)
    builder = AudioBlockBuilder(audio, cache_dir=tmp_path / "cache")
    big = builder.slice_time_range("BLOCK_001", 0, 600)
    small = builder.slice_time_range("BLOCK_001", 0, 360)
    assert big != small
    assert builder.slice_time_range("OTHER_LOGICAL_ID", 0, 360) == small
    assert len(commands) == 2
    # Missing record and changed artifact must both rebuild, not trust existence.
    small.with_suffix(".json").unlink()
    assert builder.slice_time_range("BLOCK_001", 0, 360) == small
    assert len(commands) == 3
    small.write_bytes(b"corrupt")
    builder.slice_time_range("BLOCK_001", 0, 360)
    assert len(commands) == 4
    record = json.loads(small.with_suffix(".json").read_text())
    record["identity"]["end_us"] = 600_000000
    small.with_suffix(".json").write_text(json.dumps(record))
    builder.slice_time_range("BLOCK_001", 0, 360)
    assert len(commands) == 5
    distinct = builder.slice_time_range("BLOCK_001", 0, 360.000001)
    assert distinct != small
    audio.write_bytes(b"BBBB")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        builder.slice_time_range("BLOCK_001", 0, 360)
    new_builder = AudioBlockBuilder(audio, cache_dir=tmp_path / "cache")
    assert new_builder.slice_time_range("BLOCK_001", 0, 360) != small


def test_change_during_active_job_stops_before_commit(tmp_path, monkeypatch):
    from src.source_identity import SourceIdentity
    original = SourceIdentity.assert_unchanged
    def check(self):
        # Change on the second check: source identity must remain stable throughout.
        check.count += 1
        if check.count == 2:
            Path(self.path).write_bytes(b"changed source")
        original(self)
    check.count = 0
    monkeypatch.setattr(SourceIdentity, "assert_unchanged", check)
    result, _, _, _, checkpoint = run_audio_case(tmp_path, monkeypatch, fail_count=0)
    assert result == 1
    assert checkpoint["next_audio_start_us"] == 0
    assert not checkpoint["confirmed_segments"]
