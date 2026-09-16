"""EXPERIMENT_OBSERVABILITY_CHANGE: best-effort diagnostics, no engine policy."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


class Diagnostics:
    def __init__(self, work):
        self.work = Path(work)
        self.state = dict(run_id=self.work.name, final_state='RUN_STARTED',
                          termination_reason=None, last_stage='INITIALIZATION', last_request_attempt=None)
        self.write_failures = 0
        self.event('RUN_STARTED')
        self.persist()

    def safe_write(self, operation):
        try:
            operation()
        except Exception:
            self.write_failures += 1
            try:
                sys.__stderr__.write('[diagnostic_write_failure]\n')
                sys.__stderr__.flush()
            except Exception:
                pass

    def event(self, kind, **data):
        record = dict(run_id=self.work.name, event=kind, timestamp=datetime.now(timezone.utc).isoformat(),
                      monotonic=time.monotonic(), **data)
        def append():
            with (self.work / 'diagnostic_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        self.safe_write(append)

    def persist(self):
        self.state['timestamp'] = datetime.now(timezone.utc).isoformat()
        self.state['diagnostic_write_failures'] = self.write_failures
        def save():
            temp = self.work / 'run_state.json.tmp'
            with temp.open('w', encoding='utf-8') as stream:
                json.dump(self.state, stream, ensure_ascii=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(self.work / 'run_state.json')
        self.safe_write(save)

    def stage(self, value):
        self.state['last_stage'] = value
        self.event(value)
        self.persist()

    def exception(self, exc, stage):
        # Arbitrary provider exception text can embed headers, prompts or reference data.
        # Preserve type/code and stack locations, redact message and source/locals entirely.
        frames = [dict(file=f.filename, line=f.lineno, function=f.name)
                  for f in traceback.extract_tb(exc.__traceback__)]
        code = getattr(exc, 'code', getattr(exc, 'status_code', None))
        return dict(exception_type=type(exc).__name__, exception_message='[REDACTED_UNTRUSTED_EXCEPTION_MESSAGE]',
                    provider_code=code if isinstance(code, int) else None,
                    traceback=frames, stage=stage)

    def request_context(self, metadata):
        result = {key:metadata.get(key) for key in ('block_id', 'provider_attempt')}
        result['attempt'] = result.pop('provider_attempt')
        result.update(physical_generation=None, physical_start=None, physical_end=None)
        try:
            text = (self.work / 'runtime.log').read_text(encoding='utf-8')
            matches = list(re.finditer(r'physical_generation=(\d+) start=([\d.]+) end=([\d.]+)', text))
            if matches:
                generation, start, end = matches[-1].groups()
                result.update(physical_generation=int(generation), physical_start=float(start), physical_end=float(end))
        except Exception:
            pass  # Unknown metadata must never prevent the provider call.
        return result

    def request(self, call, **metadata):
        start = time.monotonic()
        self.state['last_request_attempt'] = metadata.get('attempt')
        self.state['last_stage'] = 'ABOUT_TO_CALL_SDK'
        self.event('REQUEST_START', result_category='ABOUT_TO_CALL_SDK', **metadata)
        self.persist()
        try:
            response = call()
        except BaseException as exc:
            self.state['last_stage'] = 'SDK_RAISED'
            self.event('REQUEST_ERROR', elapsed_ms=(time.monotonic()-start)*1000,
                       result_category='SDK_RAISED', **metadata, **self.exception(exc, 'SDK_RAISED'))
            self.persist()
            raise
        self.state['last_stage'] = 'SDK_RETURNED'
        self.event('REQUEST_END', elapsed_ms=(time.monotonic()-start)*1000,
                   result_category='SDK_RETURNED', **metadata)
        self.persist()
        return response

    def conclude(self, state, reason):
        self.state.update(final_state=state, termination_reason=reason)

    def checkpoint_reason(self):
        try:
            checkpoint = json.loads((self.work/'state/checkpoint.json').read_text(encoding='utf-8'))
            message = checkpoint.get('error_message') or ''
            # Retain reason codes only, never provider message content.
            return sorted(set(re.findall(r'\b[A-Z][A-Z_]{3,}\b', message)))
        except Exception:
            return []

    @contextmanager
    def lifecycle(self):
        self.state['final_state'] = 'RUNNING'
        self.persist()
        try:
            yield self
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            reason = 'KEYBOARD_INTERRUPT' if interrupted else ('EVALUATION_FAILED' if self.state['last_stage']=='EVALUATION' else 'APPLICATION_EXCEPTION')
            self.conclude('RUN_INTERRUPTED' if interrupted else 'RUN_FAILED', reason)
            self.event('RUN_EXCEPTION', **self.exception(exc, self.state['last_stage']))
            raise
        finally:
            try:
                cp = json.loads((self.work/'state/checkpoint.json').read_text(encoding='utf-8'))
                self.state['checkpoint_audio_end'] = cp.get('next_audio_start_us', 0)/1_000_000
            except Exception:
                self.state['checkpoint_audio_end'] = None
            self.state['candidate_artifact_exists'] = (self.work/'output/transcript.txt').exists()
            self.event('RUN_FINALIZED', **{k:v for k,v in self.state.items() if k not in ('run_id','timestamp')})
            self.persist()


def inspect_state(work):
    state = json.loads((Path(work)/'run_state.json').read_text(encoding='utf-8'))
    if state['final_state'] in ('RUN_STARTED', 'RUNNING'):
        return dict(state, inspection='RUN_INCOMPLETE_NO_FINALIZATION')
    return dict(state, inspection='FINALIZATION_RECORDED')
