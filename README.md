# Vietnamese Verbatim Transcription Agent (Bản thử nghiệm MVP 1)

AI Agent chuyên phiên âm nguyên văn (verbatim transcription) tiếng Việt từ các file ghi âm, sử dụng **Google Gemini Files API** và SDK chính thức `google-genai`.

---

## 📌 Mục tiêu dự án & Giới hạn của MVP 1

> [!IMPORTANT]
> **Tuyên bố quan trọng về MVP 1:**
> - Bản thử nghiệm **MVP 1** tập trung chứng minh tính khả thi của pipeline kỹ thuật:
>   `Audio → Python → Gemini Files API → Gemini Transcriber → output/transcript.txt → state/checkpoint.json`
> - MVP 1 được tối ưu để thử nghiệm với file âm thanh độ dài ngắn và vừa (khoảng 3 – 10 phút).
> - **Chưa bao gồm:** Xử lý file 500MB+ tự động cắt block, speaker diarization bằng mô hình chuyên biệt riêng, Validator AI riêng, xuất file DOCX/SRT hay giao diện web/desktop. Các tính năng này nằm trong lộ trình các Milestone tiếp theo (xem mục Lộ trình bên dưới).

---

## 🏗️ Kiến trúc hệ thống

```text
Audio File (audio/sample.mp3)
        ↓
AudioManager (Quét định dạng, không load cả file vào RAM)
        ↓
GeminiClient (Gemini Files API + Exponential Backoff Retry)
        ↓
GeminiTranscriber (Kết hợp system_prompt.txt verbatim 17 nguyên tắc)
        ↓
output/transcript.txt (Nội dung phiên âm nguyên văn UTF-8)
        ↓
state/checkpoint.json (Lưu trạng thái an toàn với Atomic Write)
```

Các interface cho tương lai đã được chuẩn bị sẵn:
- `BlockBuilder` (`src/block_builder.py`): Chuẩn bị cho Milestone 2 để chia block audio xác thực theo `SOURCE_INDEX = 001, 002...`
- `ControlRouter` (`src/control_router.py`): Tách bạch giữa **CONTROL INPUT** (`START`, `RESUME`, `STOP`...) và **SOURCE INPUT** (lời nói thực tế).

---

## 🚀 Hướng dẫn cài đặt & Sử dụng (Dành cho người mới bắt đầu)

### Bước 1: Cài đặt Python 3.11+
Đảm bảo máy tính đã cài đặt Python 3.11 trở lên.
Kiểm tra bằng cách mở PowerShell hoặc Command Prompt:
```bash
python --version
```

### Bước 2: Mở thư mục dự án
Mở terminal tại thư mục chứa dự án:
```bash
cd "d:\ZEC\Vietnamese Verbatim Transcription Agent"
```

### Bước 3: Tạo và kích hoạt môi trường ảo (Virtual Environment)
Trên Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Bước 4: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 5: Cấu hình API Key trong file `.env`
Sao chép file mẫu `.env.example` thành `.env`:
```powershell
copy .env.example .env
```
Mở file `.env` và thay thế `YOUR_API_KEY_HERE` bằng API Key thực tế từ Google AI Studio:
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_DELAY_SECONDS=2
TIMEOUT_SECONDS=300
```
*(Ghi chú: Bạn có thể chọn model `gemini-2.5-flash` hoặc `gemini-3.7-flash`).*

### Bước 6: Đặt file âm thanh cần phiên âm
Đặt một file ghi âm (định dạng `.mp3`, `.m4a`, `.wav`, `.mp4`, `.aac`, `.flac`, hoặc `.ogg`) vào thư mục `audio/`.
Ví dụ:
```text
audio/cuoc_hop_01.mp3
```

### Bước 7: Khởi chạy chương trình
Chạy lệnh CLI:
```bash
python -m src.main
```

Nếu file audio này đã được phiên âm hoàn tất trước đó, hệ thống sẽ bảo vệ tránh trừ quota không cần thiết (Idempotency). Để buộc chạy lại từ đầu và ghi đè transcript cũ:
```bash
python -m src.main --force
```

### Bước 8: Kiểm tra kết quả
Sau khi xử lý hoàn tất, kết quả xuất hiện tại:
- `output/transcript.txt`: Toàn bộ nội dung lời thoại nguyên văn (kèm từ đệm, ngập ngừng, timestamp nếu có).
- `state/checkpoint.json`: File lưu thông tin job, Gemini file ID, trạng thái `COMPLETED` và mốc thời gian.

---

## 🧪 Chạy kiểm thử tự động (Unit Tests)

Dự án đi kèm bộ test kiểm thử đầy đủ các chức năng quản lý checkpoint, xử lý lỗi thiếu API key, thiếu file audio và tính năng Idempotency:
```bash
pytest tests/test_checkpoint.py -v
```

---

## ❓ Xử lý lỗi thường gặp (Troubleshooting)

| Vấn đề / Thông báo lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `GEMINI_API_KEY is not set or contains placeholder value` | Chưa tạo file `.env` hoặc chưa điền API Key thật. | Tạo file `.env` từ `.env.example` và nhập `GEMINI_API_KEY`. |
| `No audio file found in '.../audio'` | Thư mục `audio/` đang trống hoặc chỉ chứa file ẩn. | Đặt ít nhất một file âm thanh (ví dụ: `test.mp3`) vào thư mục `audio/`. |
| `none have supported extensions` | File trong `audio/` không đúng định dạng âm thanh. | Sử dụng các định dạng hỗ trợ: `.mp3`, `.m4a`, `.wav`, `.mp4`, `.aac`, `.flac`, `.ogg`. |
| `Failed to upload audio after 3 attempts` | Kết nối mạng chập chờn hoặc API Key không hợp lệ. | Kiểm tra đường truyền Internet, tính hợp lệ của API Key, hoặc tăng `RETRY_MAX_ATTEMPTS` trong `.env`. |
| `This audio file has already been successfully transcribed` | File đã có bản phiên âm thành công trước đó (Idempotency). | Sử dụng tham số `python -m src.main --force` nếu muốn phiên âm lại. |

---

## 🗺️ Lộ trình phát triển (Milestones)

- **Milestone 1 (Hiện tại - MVP 1):** Pipeline hoàn chỉnh từ Audio nguyên file qua Gemini Files API → `transcript.txt` + `checkpoint.json`.
- **Milestone 2:** Tích hợp `BlockBuilder` chia nhỏ file ghi âm cực dài (500MB+) thành các block có `SOURCE_INDEX` tuần tự (`001`, `002`...).
- **Milestone 3:** Tích hợp `Validator` tự động kiểm tra tính đầy đủ, chống hallucination, retry block lỗi mà không mất các block đã đạt.
- **Milestone 4:** Cơ chế `RESUME` theo checkpoint phân đoạn nguồn (`last_confirmed_source_index`).
- **Milestone 5:** Bộ `OutputMerger` xuất đa định dạng: `TXT`, `DOCX`, và `SRT` (phụ đề chuẩn).
- **Milestone 6:** Giao diện người dùng (UI) tương tác trực quan.
