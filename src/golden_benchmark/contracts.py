"""Pure contracts and lossless input import for partially reliable human metadata."""
from dataclasses import dataclass
from enum import Enum
import re

class ErrorType(str, Enum):
    EXACT_MATCH='EXACT_MATCH'
    MINOR_ORTHOGRAPHIC='MINOR_ORTHOGRAPHIC'
    WORD_SUBSTITUTION='WORD_SUBSTITUTION'
    WORD_INSERTION='WORD_INSERTION'
    WORD_OMISSION='WORD_OMISSION'
    PHRASE_SUBSTITUTION='PHRASE_SUBSTITUTION'
    PHRASE_INSERTION='PHRASE_INSERTION'
    PHRASE_OMISSION='PHRASE_OMISSION'
    HALLUCINATION='HALLUCINATION'
    UNSUPPORTED_REPLACEMENT='UNSUPPORTED_REPLACEMENT'
    OVER_NORMALIZATION='OVER_NORMALIZATION'
    ENTITY_ERROR='ENTITY_ERROR'
    NUMBER_ERROR='NUMBER_ERROR'
    UNCERTAINTY_ERROR='UNCERTAINTY_ERROR'
    SPEAKER_ERROR='SPEAKER_ERROR'
    SPEAKER_FRAGMENTATION='SPEAKER_FRAGMENTATION'
    SPEAKER_MERGE='SPEAKER_MERGE'
    TIMESTAMP_ERROR='TIMESTAMP_ERROR'
    COVERAGE_ERROR='COVERAGE_ERROR'

class Severity(str, Enum):
    INFO='INFO'
    MINOR='MINOR'
    MAJOR='MAJOR'
    CRITICAL='CRITICAL'

class SemanticRisk(str, Enum):
    LOW='LOW'
    MEDIUM='MEDIUM'
    HIGH='HIGH'

@dataclass(frozen=True)
class Segment:
    index: int
    speaker_id: str
    speaker_display_name: str
    text: str
    seconds: float
    timestamp_reliable: bool
    line: int

HEADER=re.compile(r'\[(\d{2}:\s*\d{2}:\s*\d{2})\s+- ([^\]\r\n]+)\](:?)')

def parse_transcript(text: str, duration: float):
    """Never repairs source; anomalies annotate the parsed evaluation view only."""
    headers=list(HEADER.finditer(text));segments=[];anomalies=[];speakers={};previous=-1
    if not headers: raise ValueError('BENCHMARK_NO_SEGMENTS')
    prefix=text[:headers[0].start()].strip()
    if prefix: anomalies.append(dict(reason='NON_DIALOGUE_PREFIX',text=prefix))
    for i,match in enumerate(headers):
        timestamp,label,colon=match.groups()
        h,m,s=map(int,timestamp.split(':'))
        if m>=60 or s>=60: raise ValueError('BENCHMARK_TIMESTAMP_SYNTAX')
        seconds=h*3600+m*60+s;reliable=seconds>=previous and seconds<=duration and not any(c.isspace() for c in timestamp)
        line=text.count('\n',0,match.start())+1
        if not reliable: anomalies.append(dict(reason='TIMESTAMP_REFERENCE_ANOMALY',index=i,line=line,timestamp=timestamp,previous_seconds=previous))
        if not colon: anomalies.append(dict(reason='MISSING_DELIMITER',index=i,line=line))
        body=text[match.end():headers[i+1].start() if i+1<len(headers) else len(text)].strip()
        if not body: raise ValueError('BENCHMARK_EMPTY_UTTERANCE')
        if re.search(r'\[\d[^\]\n]* - ',body): raise ValueError('BENCHMARK_UNPARSED_HEADER')
        speakers.setdefault(label.strip(),f'SPEAKER_{len(speakers)+1:03d}')
        segments.append(Segment(i,speakers[label.strip()],label,body,seconds,reliable,line))
        previous=seconds
    return segments,anomalies
