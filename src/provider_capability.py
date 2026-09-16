"""Job-local request-contract evidence, separate from transcription retry/state."""
from dataclasses import dataclass, asdict
import hashlib
import io
import json
import math
import struct
import wave
import httpx
from uuid import uuid4

import jsonschema
from google.genai import types
from src.model_policy import is_model_allowed


@dataclass(frozen=True)
class RequestPath:
    model: str
    adapter: str
    api_method: str
    endpoint: str


@dataclass
class CapabilityEvidence:
    path: RequestPath
    audio_input: bool | None = None
    system_instruction: bool | None = None
    structured_output: bool | None = None
    schema_compatible: bool | None = None
    request_endpoint_usable: bool | None = None
    reason: str = 'UNKNOWN_REQUEST_ELIGIBILITY_FAILURE'
    prompt_version: str | None = None
    prompt_sha256: str | None = None
    provider_error_code: int | None = None
    system_instruction_attached: bool = False
    audio_sha256: str | None = None
    schema_sha256: str | None = None

    @property
    def state(self):
        if self.eligible:
            return 'VERIFIED_ELIGIBLE'
        if self.reason in ('ENDPOINT_MODEL_NOT_FOUND', 'SYSTEM_INSTRUCTION_UNSUPPORTED',
                           'AUDIO_INPUT_UNSUPPORTED', 'STRUCTURED_OUTPUT_UNSUPPORTED',
                           'SCHEMA_UNSUPPORTED', 'LITE_MODEL_FORBIDDEN'):
            return 'VERIFIED_INELIGIBLE'
        if self.reason in ('TRANSIENT_PROVIDER_FAILURE', 'TRANSIENT_UPLOAD_FAILURE'):
            return 'UNVERIFIED_TRANSIENT'
        return 'UNVERIFIED_UNKNOWN'

    @property
    def eligible(self):
        return is_model_allowed(self.path.model) and all(value is True for value in (
            self.audio_input, self.system_instruction, self.structured_output,
            self.schema_compatible, self.request_endpoint_usable))

    def report(self):
        return dict(asdict(self), capability_state=self.state, eligible=self.eligible, lite=not is_model_allowed(self.path.model))


def tiny_audio():
    """One second public synthetic tone: no speech, reference or private audio."""
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(struct.pack('<h', int(1600 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(16000)))
    return output.getvalue()


class CapabilityGate:
    """Cache belongs to one SDK client/job; never persisted or shared by account."""
    def __init__(self, sdk, schema, endpoint):
        self.sdk, self.schema, self.endpoint = sdk, schema, endpoint
        self.results = {}
        self.audio_asset = None
        self.probe_counts = {}
        self.max_probes_per_path = 2  # Initial probe plus at most one on-demand fallback re-probe.

    def path(self, model, adapter):
        method = 'models.generate_content' if adapter == 'generate_content' else 'interactions.create'
        return RequestPath(model, adapter, method, self.endpoint)

    def can_reprobe(self, model, adapter):
        path = self.path(model, adapter)
        previous = self.results.get(path)
        return bool(previous and previous.state == 'UNVERIFIED_TRANSIENT'
                    and self.probe_counts.get(path, 0) < self.max_probes_per_path
                    and adapter == 'generate_content' and is_model_allowed(model))

    def check(self, model, adapter, *, reprobe=False):
        path = self.path(model, adapter)
        if path in self.results and not (reprobe and self.can_reprobe(model, adapter)):
            return self.results[path]
        if path in self.results:
            print(f'[CAPABILITY_REPROBE] model={model} adapter={adapter} reason=FALLBACK_NEEDED probe={self.probe_counts.get(path, 0)+1} limit={self.max_probes_per_path}', flush=True)
        evidence = CapabilityEvidence(path)
        self.results[path] = evidence
        if not is_model_allowed(model):
            evidence.reason = 'LITE_MODEL_FORBIDDEN'
        elif adapter != 'generate_content':
            # Local adapter demonstrably omits instruction; provider support is a separate audit.
            evidence.system_instruction = False
            evidence.reason = 'SYSTEM_INSTRUCTION_UNSUPPORTED'
        else:
            self.probe_counts[path] = self.probe_counts.get(path, 0) + 1
            self._probe(evidence)
        print('[MODEL_CAPABILITY] ' + json.dumps(evidence.report()), flush=True)
        if not evidence.eligible:
            print(f'[MODEL_INELIGIBLE] model={model} adapter={adapter} reason={evidence.reason}', flush=True)
        return evidence

    def _probe(self, evidence):
        marker = 'probe_' + uuid4().hex
        instruction = ('AUDIO_REQUEST_CONTRACT_PROBE_V1\nThis is a request capability test, not a transcription. '
                       'Return one schema 1.0 segment with text exactly ' + marker +
                       ', timestamp 00:00:00, speaker Probe, source_index 1. '
                       'Use job_id PROBE, session_id PROBE, block_id PROBE, first_source_index 1, '
                       'last_source_index 1, status CONFIRMED. Do not transcribe the tone.')
        evidence.prompt_version = instruction.splitlines()[0]
        evidence.prompt_sha256 = hashlib.sha256(instruction.encode()).hexdigest()
        audio = tiny_audio()
        evidence.audio_sha256 = hashlib.sha256(audio).hexdigest()
        evidence.schema_sha256 = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        config = types.GenerateContentConfig(system_instruction=instruction, temperature=0,
                    response_mime_type='application/json', response_json_schema=self.schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    max_output_tokens=1024, http_options=types.HttpOptions(timeout=30000,
                        retry_options=types.HttpRetryOptions(attempts=1)))
        evidence.system_instruction_attached = config.system_instruction == instruction
        # The nonce exists only in the system instruction, proving more than file existence.
        if self.audio_asset is None:
            try:
                self.audio_asset = self.sdk.files.upload(file=io.BytesIO(audio),
                    config=types.UploadFileConfig(mime_type='audio/wav', display_name='public-capability-tone-1s'))
            except Exception as exc:
                code = getattr(exc, 'code', getattr(exc, 'status_code', None))
                evidence.provider_error_code = code if isinstance(code, int) else None
                evidence.reason = 'TRANSIENT_UPLOAD_FAILURE' if self.transient(exc, code) else 'CAPABILITY_PROBE_FAILED'
                return  # Files API failure is not generation endpoint evidence.
        try:
            response = self.sdk.models.generate_content(model=evidence.path.model,
                contents=[self.audio_asset,
                          'Process the attached test audio according to the system instruction.'], config=config)
            evidence.request_endpoint_usable = True
            evidence.audio_input = True
            payload = json.loads(response.text)
            jsonschema.validate(payload, self.schema)
            evidence.structured_output = evidence.schema_compatible = True
            evidence.system_instruction = any(s.get('text') == marker for s in payload['segments'])
            evidence.reason = 'REQUEST_CONTRACT_SUPPORTED' if evidence.eligible else 'CAPABILITY_PROBE_FAILED'
        except Exception as exc:
            code = getattr(exc, 'code', getattr(exc, 'status_code', None))
            evidence.provider_error_code = code if isinstance(code, int) else None
            if code == 404:
                evidence.request_endpoint_usable = False
                evidence.reason = 'ENDPOINT_MODEL_NOT_FOUND'
            elif self.transient(exc, code):
                evidence.reason = 'TRANSIENT_PROVIDER_FAILURE'
            elif isinstance(exc, jsonschema.ValidationError):
                evidence.schema_compatible = False
                evidence.reason = 'SCHEMA_UNSUPPORTED'
            else:
                evidence.reason = 'CAPABILITY_PROBE_FAILED'

    @staticmethod
    def transient(exc, code):
        return code in (408, 429, 500, 502, 503, 504) or (code is None and
            isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)))

    def invalidate(self, model, adapter, code):
        """Deterministic request failure affects only the exact path in this job."""
        path = self.path(model, adapter)
        if code == 404 and path in self.results:
            self.results[path].request_endpoint_usable = False
            self.results[path].reason = 'ENDPOINT_MODEL_NOT_FOUND'
