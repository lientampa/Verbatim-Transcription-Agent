"""Recoverable publication of a validated output set; one writer per directory."""
import json
import os
import shutil
from pathlib import Path

NAMES = ("transcript.txt", "transcript.docx", "subtitle.srt", "render_manifest.json")


def recover_outputs(directory):
    journal = directory / ".render-transaction"
    marker = journal / "pending.json"
    if not marker.exists():
        return
    old = json.loads(marker.read_text(encoding="utf-8"))
    for name in NAMES:
        target = directory / name
        if old[name]:
            backup = journal / name
            if not target.exists() or target.read_bytes() != backup.read_bytes():
                restore = journal / (name + ".restore")
                shutil.copyfile(backup, restore)
                os.replace(restore, target)
        elif target.exists():
            target.unlink()
    marker.unlink()
    shutil.rmtree(journal)


def publish_outputs(stage, directory):
    recover_outputs(directory)
    journal = directory / ".render-transaction"
    journal.mkdir(exist_ok=True)
    old = {}
    for name in NAMES:
        target = directory / name
        old[name] = target.exists()
        if old[name]:
            shutil.copyfile(target, journal / name)
            with (journal / name).open("r+b") as stream:
                os.fsync(stream.fileno())
        with (stage / name).open("r+b") as stream:
            os.fsync(stream.fileno())
    with (journal / "pending.json").open("w", encoding="utf-8") as stream:
        json.dump(old, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for name in NAMES:
            os.replace(stage / name, directory / name)
    except BaseException:
        recover_outputs(directory)
        raise
    (journal / "pending.json").unlink()
    shutil.rmtree(journal)
