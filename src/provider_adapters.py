"""Provider contracts stop here; downstream consumes canonical schema 1.0."""
import hashlib
import ast
import json
import math
import re
from dataclasses import dataclass
from types import SimpleNamespace
from google.genai import types


@dataclass(frozen=True)
class ModelCapabilities:
    provider_adapter: str
    supports_thinking: bool
    supported_thinking_levels: tuple[str, ...]
    preferred_transcription_thinking_level: str | None
    supports_function_calling: bool
    supports_structured_output: bool


def capabilities(model):
    name=model.removeprefix("models/")
    if name=="gemini-3.5-transcribe":
        return ModelCapabilities("transcribe",False,(),None,False,False)
    if name=="gemini-2.5-flash":
        return ModelCapabilities("generate_content",True,(),None,True,True)
    if re.match(r"gemini-3(?:\.|-)",name):
        levels=("minimal","low","medium","high") if re.match(r"gemini-3(?:\.[56])?-flash(?:-|$)",name) else ("low","medium","high")
        return ModelCapabilities("generate_content",True,levels,levels[0],True,True)
    return ModelCapabilities("generate_content",False,(),None,True,True)


class ProviderAdapterError(ValueError):
    """Non-transient adapter mismatch with privacy-safe stage diagnostics."""
    def __init__(self, reason, stage="ANNOTATION_PARSE", detail=None):
        self.reason=reason
        self.stage=stage
        self.detail=detail
        super().__init__(reason + (": " + detail if detail else ""))
        print(f"[TRANSCRIBE_ADAPTER_ERROR] model=gemini-3.5-transcribe stage={stage} reason={reason} detail={detail}",flush=True)


class AnnotationSemanticError(ProviderAdapterError):
    """Interpretable contract, invalid individual model response."""



def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WordInfoEvidence:
    physical_start: float
    physical_end: float
    endpoint: float | None
    payload_hash: str
    segment_hash: str

    def matches(self, transcript, start, end):
        return self.physical_start==start and self.physical_end==end and self.payload_hash==payload_digest(transcript.to_dict())



class GeminiGenerateContentAdapter:
    def generate(self,sdk,model,asset,prompt,config):
        return sdk.models.generate_content(model=model,contents=[asset,prompt],config=config)


def field(value,key,default=None):
    return value.get(key,default) if isinstance(value,dict) else getattr(value,key,default)


def seconds(value):
    if not isinstance(value,str) or not re.fullmatch(r"\d+(?:\.\d+)?s",value):
        raise ProviderAdapterError("TRANSCRIBE_TIMESTAMP_UNSUPPORTED")
    result=float(value[:-1])
    if not math.isfinite(result):
        raise ProviderAdapterError("TRANSCRIBE_TIMESTAMP_UNSUPPORTED")
    return result


def word_timing(word, physical_start, duration, word_index=0):
    """Validate audio seconds before conversion; byte indices never supply time.

    Retain the existing 1 microsecond numerical tolerance (not a rounding policy).
    Missing end is allowed; missing start makes timed mapping unavailable.
    """
    try:
        start=seconds(field(word,"start_offset"))
        end_raw=field(word,"end_offset")
        end=seconds(end_raw) if end_raw is not None else None
        if not 0<=start<=duration+1e-6:
            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={word_index} START_OFFSET_OUT_OF_BOUNDS")
        if end is not None and not start<=end<=duration+1e-6:
            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={word_index} END_OFFSET_OUT_OF_BOUNDS")
        return start,end,physical_start+start,None if end is None else physical_start+end
    except ProviderAdapterError:
        def safe(value):
            if value is None or type(value) in (int,float): return str(value)[:80]
            if isinstance(value,str): return json.dumps(value[:80],ensure_ascii=True)
            return "<"+type(value).__name__+">"
        details=" ".join(f"{key}={safe(field(word,key))}" for key in
                         ("start_offset","end_offset","start_index","end_index"))
        print(f"[TRANSCRIBE_WORDINFO_PARSE_ERROR] word_index={word_index} {details} slice_duration={duration} physical_start={physical_start}",flush=True)
        raise


def absolute_timestamp(timestamp, block_start, block_end, basis="BLOCK_RELATIVE"):
    """Basis is an explicit contract setting, never inferred from numeric values."""
    if basis not in ("BLOCK_RELATIVE", "SOURCE_ABSOLUTE"):
        raise ProviderAdapterError("TRANSCRIBE_TIMESTAMP_BASIS_UNSUPPORTED")
    result=timestamp+block_start if basis=="BLOCK_RELATIVE" else timestamp
    if not math.isfinite(result) or not block_start<=result<=block_end+1e-6:
        raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail="TIMESTAMP_OUT_OF_PHYSICAL_BOUNDS")
    return result


class TranscriptionModelAdapter:
    TIMESTAMP_BASIS="BLOCK_RELATIVE"  # SDK WordInfo offsets are relative to uploaded audio.

    @staticmethod
    def preflight(sdk):
        try:
            if not callable(getattr(getattr(sdk,"interactions",None),"create",None)):
                return False,"INTERACTIONS_ENDPOINT_UNAVAILABLE"
            from google.genai._gaos.types.interactions.generationconfig import GenerationConfig
            from google.genai._gaos.types.interactions.wordinfo import WordInfo
            payload={"transcription_config":{"mode":{"type":"verbatim","diarization_mode":"speaker","timestamp_granularities":["word"]}}}
            encoded=GenerationConfig.model_validate(payload).model_dump(exclude_none=True)
            if encoded!=payload or not {"text","speaker","start_offset","end_offset","start_index","end_index"}<=set(WordInfo.model_fields):
                return False,"SDK_NATIVE_CONTRACT_MISMATCH"
            return True,"SDK_NATIVE_CONTRACT_AVAILABLE"
        except Exception as exc:
            return False,"SDK_NATIVE_CONTRACT_UNAVAILABLE:"+type(exc).__name__

    def generate(self,sdk,model,asset,prompt,config=None):
        print("[TRANSCRIBE_ADAPTER] stage=REQUEST_BUILD",flush=True)
        uri,mime=field(asset,"uri"),field(asset,"mime_type")
        if not isinstance(uri,str) or not isinstance(mime,str):
            raise ProviderAdapterError("TRANSCRIBE_AUDIO_REFERENCE_UNAVAILABLE",stage="REQUEST_BUILD")
        try:
            response=sdk.interactions.create(model=model,
                input=[{"type":"audio","uri":uri,"mime_type":mime}],
                generation_config={"transcription_config":{"mode":{"type":"verbatim",
                    "diarization_mode":"speaker","timestamp_granularities":["word"]}}})
        except Exception as exc:
            if not isinstance(getattr(exc,"status_code",None),int):
                raise
            error=ProviderAdapterError("TRANSCRIBE_PROVIDER_HTTP_ERROR",stage="PROVIDER_REQUEST",detail=f"HTTP_{exc.status_code}")
            error.code=exc.status_code
            try:
                error.details=json.loads(exc.body).get("error",{}).get("details",[])
            except (AttributeError,ValueError,TypeError):
                error.details=[]
            raise error from exc
        print("[TRANSCRIBE_ADAPTER] stage=PROVIDER_RESPONSE_PARSE",flush=True)
        status=field(response,"status")
        if status in ("incomplete","budget_exceeded"):
            raise ProviderAdapterError("OUTPUT_TRUNCATED: transcription interaction incomplete",stage="PROVIDER_RESPONSE_PARSE")
        if status!="completed":
            raise ProviderAdapterError("TRANSCRIBE_MODEL_FAILURE",stage="PROVIDER_RESPONSE_PARSE",detail=f"status={status}")
        try:
            text=self.canonical(response,prompt)
        except ProviderAdapterError as exc:
            exc.provider_callable=True
            raise
        usage=field(response,"usage",{})
        normalized_usage=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=field(usage,"total_input_tokens"),
            candidates_token_count=field(usage,"total_output_tokens"),
            thoughts_token_count=field(usage,"total_thought_tokens"),
            total_token_count=field(usage,"total_tokens"))
        return SimpleNamespace(text=text,usage_metadata=normalized_usage,wordinfo_evidence=self.evidence,
            candidates=[SimpleNamespace(finish_reason="STOP",content=SimpleNamespace(parts=[SimpleNamespace(text=text,thought=False)]))])

    def canonical(self,response,prompt):
        print("[TRANSCRIBE_ADAPTER] stage=CANONICAL_MAPPING",flush=True)
        self.evidence=None
        identity={}
        for key in ("job_id","session_id","block_id"):
            match=re.search(r'- '+key+r': "([^"\n]+)"',prompt)
            if not match: raise ProviderAdapterError("TRANSCRIBE_CONTEXT_MISSING",stage="CANONICAL_MAPPING")
            identity[key]=match.group(1)
        bounds=re.search(r'Absolute audio boundaries \(seconds\): start=(\d+(?:\.\d+)?), end=(\d+(?:\.\d+)?)',prompt)
        if not bounds: raise ProviderAdapterError("TRANSCRIBE_AUDIO_BOUNDARIES_MISSING",stage="CANONICAL_MAPPING")
        start,end=map(float,bounds.groups())
        if not math.isfinite(start+end) or end<=start: raise ProviderAdapterError("TRANSCRIBE_AUDIO_BOUNDARIES_INVALID",stage="CANONICAL_MAPPING")
        # Native IDs are scoped to a request, not evidence of cross-block identity.
        context=re.search(r'đã dùng trong các block được xác nhận: (\[.*?\])\.',prompt)
        prior=ast.literal_eval(context.group(1)) if context else []
        next_label=max([int(m.group(1)) for label in prior if isinstance(label,str)
                        for m in [re.fullmatch(r"Người nói (\d+)",label)] if m]+[0])+1
        speakers={};segments=[];previous=-1;previous_end=None;previous_speaker=None;max_end=None;word_count=0
        print("[TRANSCRIBE_ADAPTER] stage=ANNOTATION_PARSE timestamp_basis=BLOCK_RELATIVE",flush=True)
        for step in field(response,"steps",[]) or []:
            if field(step,"type")!="model_output": continue
            for content in field(step,"content",[]) or []:
                if field(content,"type")!="text": continue
                words=[w for w in field(content,"annotations",[]) or [] if field(w,"type")=="word_info"]
                raw=field(content,"text","")
                if not isinstance(raw,str):
                    raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail="CONTENT_TEXT_TYPE")
                values=[]
                for index,word in enumerate(words):
                    value=field(word,"text")
                    if value is None:
                        left,right=field(word,"start_index"),field(word,"end_index")
                        encoded=raw.encode("utf-8")
                        if type(left) is not int or type(right) is not int or not 0<=left<right<=len(encoded):
                            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={index} MISSING_TEXT_AND_BYTE_SPAN")
                        try:
                            value=encoded[left:right].decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={index} INVALID_UTF8_BYTE_SPAN") from exc
                    if not isinstance(value,str) or not value.strip():
                        raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={index} WORD_TEXT_TYPE_OR_EMPTY")
                    values.append(value)
                # Locate words in provider text; keep punctuation/pauses between them.
                spans=[];cursor=0
                for index,value in enumerate(values):
                    word=words[index]
                    left,right=field(word,"start_index"),field(word,"end_index")
                    if left is not None or right is not None:
                        encoded=raw.encode("utf-8")
                        if type(left) is not int or type(right) is not int or not 0<=left<right<=len(encoded):
                            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail="TEXT_INDEX_BOUNDS")
                        try:
                            position=len(encoded[:left].decode("utf-8"))
                            associated=encoded[left:right].decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail="TEXT_INDEX_UTF8") from exc
                        if associated!=value or position<cursor:
                            raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail="TEXT_INDEX_CORRESPONDENCE")
                    else:
                        position=raw.find(value,cursor)
                    if position<0 or any(c.isalnum() for c in raw[cursor:position]):
                        raise AnnotationSemanticError("TRANSCRIBE_ANNOTATIONS_INCOMPLETE",detail="WORD_TEXT_COVERAGE")
                    spans.append(position);cursor=position+len(value)
                if not words or any(c.isalnum() for c in raw[cursor:]):
                    raise AnnotationSemanticError("TRANSCRIBE_ANNOTATIONS_INCOMPLETE",detail="WORD_TEXT_COVERAGE")
                chunks=[raw[0 if i==0 else spans[i]:spans[i+1] if i+1<len(spans) else len(raw)] for i in range(len(spans))]
                for index,(word,value) in enumerate(zip(words,chunks)):
                    relative,finish,absolute,_=word_timing(word,start,end-start,index)
                    speaker=field(word,"speaker")
                    if speaker is not None and not isinstance(speaker,str):
                        raise AnnotationSemanticError("TRANSCRIBE_ANNOTATION_INVALID",detail=f"word={index} SPEAKER_TYPE={type(speaker).__name__}")
                    if not previous<=relative:
                        safe_speaker=lambda value: json.dumps(value[:40] if isinstance(value,str) else None,ensure_ascii=True)
                        print(f"[TRANSCRIBE_TEMPORAL_OVERLAP] word_index={index} previous_start={previous} current_start={relative} previous_end={previous_end} current_end={finish} speaker_previous={safe_speaker(previous_speaker)} speaker_current={safe_speaker(speaker)} delta_seconds={relative-previous} physical_duration={end-start} classification={'CROSS_SPEAKER_OVERLAP' if speaker!=previous_speaker else 'TEXT_AUDIO_ORDER_DIFFERENCE'}",flush=True)

                    word_count+=1
                    if finish is not None: max_end=finish if max_end is None else max(max_end,finish)
                    previous=relative
                    previous_end=finish;previous_speaker=speaker
                    # Optional speaker is genuinely unattributed, never inherited from a neighbor.
                    speaker=speaker or None
                    if speaker not in speakers:
                        speakers[speaker]=f"Người nói {next_label}";next_label+=1
                    # Bound utterance groups to keep coverage evidence near their last word.
                    if segments and segments[-1]["speaker"]==speakers[speaker] and 0<=absolute-segments[-1]["_start"]<15:
                        segments[-1]["text"]+=value
                    else:
                        sec=int(absolute)
                        segments.append(dict(source_index=len(segments)+1,text=value,speaker=speakers[speaker],
                            timestamp=f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}",_start=absolute))
        if not segments: raise ProviderAdapterError("TRANSCRIBE_ANNOTATIONS_UNAVAILABLE")
        for segment in segments:
            segment.pop("_start")
            segment["text"]=segment["text"].strip()
        payload=dict(schema_version="1.0",**identity,first_source_index=1,
            last_source_index=len(segments),status="CONFIRMED",segments=segments)
        endpoint=None if max_end is None else start+max_end
        self.evidence=WordInfoEvidence(start,end,endpoint,payload_digest(payload),payload_digest(segments))
        print(f"[TRANSCRIBE_COVERAGE_ENDPOINT] basis=MAX_WORD_END_OFFSET relative_end={max_end} absolute_end={endpoint} word_count={word_count}",flush=True)
        return json.dumps(payload,ensure_ascii=False)
