# Milestone 3.1 — Fidelity Hardening + Golden Test Regression

## Goal

Biến pipeline từ *"JSON hợp lệ + structural PASS"* thành *"audio-grounded verbatim transcription với cơ chế chống đoán và phát hiện lỗi fidelity"*. Không phá vỡ DABB, checkpoint, merger, renderer.

---

## Root Cause (tóm tắt)

> Xem [MILESTONE_3_1_ROOT_CAUSE.md](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/390b9094-58e8-40c2-9640-58d4bce80c96/MILESTONE_3_1_ROOT_CAUSE.md)

- Validator chỉ structural — content sai vẫn PASS
- Không có TYPE C (Fidelity) retry
- Không có audio recheck
- System prompt chưa đủ mạnh override language model bias

---

## User Review Required

> [!IMPORTANT]
> **Audio Recheck (Phần VI)** gửi lại audio block tới Gemini lần 2 với prompt kiểm định. Điều này tốn thêm ~1 API call/block khi risk cao. Có hai chế độ:
> - `ALWAYS_RECHECK`: Mọi block đều recheck (accuracy cao nhất, cost gấp đôi)
> - `RISK_BASED` (default): Chỉ recheck block có fidelity risk indicators (dấu hiệu đáng ngờ)
>
> **Bạn muốn dùng chế độ nào?** Tôi sẽ implement `RISK_BASED` mặc định với flag `AUDIO_RECHECK_MODE` trong `.env`.

> [!WARNING]
> **System Prompt V5** sẽ **thay thế hoàn toàn** V4 hiện tại. Cấu trúc được viết lại theo format audio-grounded. Nếu bạn muốn giữ lại sections cụ thể từ V4, hãy ghi chú trước khi approve.

> [!IMPORTANT]
> **Golden Test** yêu cầu `new recording 5.m4a` phải có mặt trong thư mục `audio/` để chạy integration test. Test sẽ được đánh dấu `pytest.mark.integration` và skip nếu file không tồn tại. Fixture expected JSON phải được review thủ công.

---

## Proposed Changes

---

### Component 1: System Prompt

#### [MODIFY] [system_prompt.txt](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/prompts/system_prompt.txt)

Thay toàn bộ nội dung bằng System Prompt V5 audio-grounded theo spec Phần III:
- CORE IDENTITY mới: "AUDIO-GROUNDED VIETNAMESE VERBATIM TRANSCRIPTION ENGINE"
- RULE 1–11 theo spec (Audio Authority, Verbatim First, Uncertainty > Guessing, Partial Uncertainty, No Semantic Correction, No Context Completion, Timestamp, Speaker, No Meta Text, Segment Boundary, Golden Test Expectation)
- Giữ lại Section III (Chống Hallucination) từ V4.1 đã implement
- Giữ lại Section IV (JSON Output schema)

---

### Component 2: ContentFidelityValidator

#### [NEW] [fidelity_validator.py](file:///d:/ZEC/Vietnamese Verbatim Transcription Agent/src/fidelity_validator.py)

Multi-tier validator tách biệt hoàn toàn khỏi `TranscriptValidator`:

```
FidelityValidationResult
    status: PASS | FAIL | REVIEW
    risk_level: LOW | MEDIUM | HIGH
    issues: list[FidelityIssue]

FidelityIssue
    source_index: int
    indicator: str
    detail: str

ContentFidelityValidator
    check_suspiciously_polished(segments)    → detects formal language replacing colloquial
    check_unknown_token_rate(segments)       → [không rõ] count / total words
    check_semantic_substitution_patterns()   → known bad substitutions
    check_excessive_technical_terms()        → thuật ngữ xuất hiện đột ngột không có context
    check_coherence_vs_audio_offset()        → timestamp drift indicators
    validate(block_result) → FidelityValidationResult
```

Validator này **KHÔNG sửa** transcript — chỉ PASS/FAIL/REVIEW.

---

### Component 3: AudioGroundedReview

#### [NEW] [audio_review.py](file:///d:/ZEC/Vietnamese Verbatim Transcription Agent/src/audio_review.py)

```
AudioReviewResult
    status: PASS | FAIL | UNCERTAIN
    issues: list[ReviewIssue]
    reviewed_segments: list[corrected segments]  ← chỉ [không rõ] replacement, NO rewrite

AudioGroundedReviewer
    __init__(gemini_client, system_prompt_path)
    
    should_review(fidelity_result) → bool
        # RISK_BASED: chỉ review nếu risk_level == HIGH hoặc MEDIUM
    
    review_block(gemini_file, block_result, fidelity_issues) → AudioReviewResult
        # Gửi lại audio + candidate transcript + review prompt
        # Review prompt: "Nghe audio và xác nhận từng từ. Nếu candidate chứa từ
        #   không thể xác minh từ audio → thay bằng [không rõ]. KHÔNG rewrite."
        # Returns structured JSON với status và corrections
```

---

### Component 4: Updated TranscriptValidator (StructuralValidator)

#### [MODIFY] [transcript_validator.py](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/src/transcript_validator.py)

- Đổi tên nội bộ thành `StructuralValidator` (backward compatible — giữ class name `TranscriptValidator`)
- Thêm `ValidationResult.validation_tier: str = "STRUCTURAL"` field
- Giữ nguyên toàn bộ logic hiện tại (coverage, duplicate, order, range, empty, timestamp, speaker, meta, loop)
- Thêm detection: `check_timestamp_is_estimated()` → warning nếu không có source timestamp

---

### Component 5: Updated FailureClassifier

#### [MODIFY] [failure_classifier.py](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/src/failure_classifier.py)

Thêm `FailureType.FIDELITY_FAILURE` (TYPE C):
```python
FIDELITY_FAILURE = "FIDELITY_FAILURE"
```

Thêm classification keywords cho fidelity:
- `"fidelity"`, `"audio review fail"`, `"high risk"`, `"semantic substitution"`

**Behavior của FIDELITY_FAILURE:**
- KHÔNG trigger DABB shrink
- Trigger audio recheck → nếu fail → replace với `[không rõ]` → retry validate
- Nếu vẫn fail sau max retries → raise TranscriptionError

---

### Component 6: Updated Transcriber

#### [MODIFY] [transcriber.py](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/src/transcriber.py)

Thêm `fidelity_validator` và `audio_reviewer` vào pipeline:

```python
def transcribe_block(...) -> TranscriptionOutcome:
    # 1. Gemini API call
    # 2. JSON parse
    # 3. Schema validate
    # 4. Structural validate (existing)
    # 5. Fidelity validate (NEW)
    # Stage 1: trả về đối tượng có tên trường, không dùng tuple positional.
    # block_result, structural_validation, fidelity_validation
```

Giữ backward compat: `transcribe()` và `transcribe_and_validate()` vẫn hoạt động.

---

### Component 7: Updated Main Pipeline

#### [MODIFY] [main.py](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/src/main.py)

Thêm fidelity validation vào block loop:

```
Gemini transcription
    ↓
Parse
    ↓
Schema validation
    ↓
Structural validation           (existing)
    ↓ FAIL → retry (TYPE A)
Fidelity validation             (NEW)
    ↓ FAIL/REVIEW → audio recheck
Audio recheck (if HIGH risk)    (NEW)
    ↓ FAIL → [không rõ] substitution → re-validate
    ↓ PASS
Checkpoint commit               (only here)
    ↓
Merge
```

Thêm TYPE C retry logic:
```python
if fidelity_result.status in ("FAIL", "REVIEW"):
    if audio_reviewer.should_review(fidelity_result):
        review_result = audio_reviewer.review_block(...)
        if review_result.status == "FAIL":
            # Apply [không rõ] substitutions → retry validate
            ...
```

Thêm metrics tracking vào checkpoint.

---

### Component 8: Metrics Tracking

#### [MODIFY] [checkpoint.py](file:///d:/ZEC/Vietnamese%20Verbatim%20Transcription%20Agent/src/checkpoint.py)

Thêm `block_metrics: list[dict]` field vào `CheckpointData`:
```python
{
    "block_id": "BLOCK_001",
    "schema_pass": true,
    "structural_pass": true,
    "fidelity_pass": true | false | "REVIEW",
    "audio_review_count": 0,
    "retry_count": 0,
    "uncertain_segment_count": 2,
    "unknown_token_count": 3,
    "input_tokens": 1200,
    "output_tokens": 1450,
    "output_input_ratio": 1.21
}
```

---

### Component 9: Golden Test Fixtures

#### [NEW] tests/golden/new_recording_5m4a.metadata.json

```json
{
    "audio_file": "new recording 5.m4a",
    "known_bad_substitutions": [
        {
            "source_phrase": "kỹ thuật của Bộ Quốc phòng",
            "observed_bad_output": "pháp khác ra ngay thì đều một quốc phòng"
        },
        ...7 examples total
    ],
    "review_status": "PENDING_MANUAL_REVIEW",
    "notes": "Expected transcript requires manual review before marking PASS"
}
```

#### [NEW] tests/golden/new_recording_5m4a.expected.json

Placeholder — **không auto-generate từ AI output**. Phải được review thủ công.

---

### Component 10: Updated Tests

#### [NEW] tests/test_fidelity_validator.py

Test `ContentFidelityValidator`:
- Test PASS với clean transcript
- Test FAIL với semantic substitution patterns
- Test REVIEW với suspiciously polished language
- Test unknown token rate threshold
- Test các known bad substitution examples từ golden test

#### [NEW] tests/test_audio_review.py

Test `AudioGroundedReviewer` với mock:
- Test should_review() logic
- Test review_block() với mock Gemini response
- Test [không rõ] substitution
- Test PASS/FAIL/UNCERTAIN handling

#### [MODIFY] tests/test_milestone3_acceptance.py

Thêm:
- Test fidelity FAIL không được checkpoint
- Test TYPE C retry không trigger DABB shrink
- Test [không rõ] replacement sau audio recheck FAIL
- Test known bad substitution từ golden examples

#### [NEW] tests/test_golden_regression.py

```python
@pytest.mark.integration
@pytest.mark.skipif(not Path("audio/new recording 5.m4a").exists(), ...)
class TestGoldenRegression:
    def test_no_known_bad_substitutions(...)
    def test_uncertainty_preferred_over_guessing(...)
    def test_structural_pass(...)
```

---

## Verification Plan

### Automated Tests

```bash
# Run toàn bộ unit tests (không cần audio file)
pytest tests/ -q --ignore=tests/test_golden_regression.py

# Run golden regression (cần audio file)
pytest tests/test_golden_regression.py -v -m integration

# Run full suite
pytest -q
```

### Manual Verification

1. Kiểm tra `prompts/system_prompt.txt` đã được replace đúng V5
2. Chạy pipeline với `new recording 5.m4a` và so sánh before/after
3. Review `state/checkpoint.json` để xác nhận `block_metrics` được ghi
4. Kiểm tra output text KHÔNG chứa các known bad substitutions

---

## Open Questions

> [!IMPORTANT]
> **Q1:** `AUDIO_RECHECK_MODE` — bạn muốn `RISK_BASED` (default, chỉ recheck HIGH risk) hay `ALWAYS` (mọi block đều recheck)?

> [!IMPORTANT]
> **Q2:** Threshold `unknown_token_rate` — nếu `[không rõ]` chiếm > X% total words thì coi là REVIEW. Tôi đề xuất X=30%. Bạn có muốn điều chỉnh không?

> [!IMPORTANT]
> **Q3:** Khi audio recheck FAIL và apply `[không rõ]` replacement — bạn muốn pipeline **tiếp tục** với [không rõ] đã replace, hay **halt** và yêu cầu manual review?

---

## DABB Compatibility

DABB được giữ nguyên 100%:
- `DynamicAdaptiveBlockBuilder` không thay đổi
- `DabbAudioOrchestrator` không thay đổi
- TYPE C Fidelity failure KHÔNG trigger `on_block_failure()` với SIZE_FAILURE
- Block shrink chỉ xảy ra với SIZE_FAILURE như hiện tại
