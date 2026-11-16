# Báo cáo Day 7 — Retrieval & Context Engineering

**Sinh viên:** Nguyễn Minh Quân
**MSSV:** 2A202603747

## 1. Dữ liệu và benchmark

- **Nguồn:** *Think Stats, 2nd Edition* — Chapters 3–5.
  Source: https://greenteapress.com/thinkstats2/thinkstats2.pdf
- **Phạm vi index:** printed pages 31–73, tổng cộng 43 trang.
- **Corpus dùng trong Day 7:** 157 chunks từ Strategy B của Day 6.
- **Vector store:** PostgreSQL 16 + pgvector, bảng `lab_chunks`, chạy bằng Docker.
- **Embedding model:** `intfloat/e5-small-v2`, 384 chiều.
  - Document prefix ở Day 6: `passage: `
  - Query prefix ở Day 7: `query: `
  - Cosine similarity qua toán tử pgvector `<=>`.
- **Agent model:** `qwen2.5:3b`, chạy local bằng Ollama 0.40.16 qua endpoint `http://localhost:11434/v1`.
- **Benchmark:** 24 câu.
  - Dev: 15 câu có evidence + 2 câu thiếu evidence.
  - Test: 5 câu có evidence + 2 câu thiếu evidence.
- Mỗi câu answerable có `source_id`, trang nguồn và đoạn evidence. Câu thiếu evidence được gắn `answerable: false`; agent phải abstain thay vì dùng kiến thức ngoài corpus.

## 2. Các cấu hình retrieval

| Cấu hình | Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median latency |
|---|---|---:|---:|---:|---:|---:|
| A | TF-IDF lexical baseline, top-k=3 | dev | 5/15 | 9/15 | 6/15 | 0.40 ms |
| B | Dense E5 + pgvector Strategy B, top-k=3 | dev | **8/15** | **12/15** | **12/15** | 43.86 ms |
| C | Hybrid RRF: Dense + TF-IDF, candidate-k=10, RRF-k=60 | dev | 6/15 | 11/15 | 11/15 | 46.24 ms |
| B (đã chốt) | Dense E5 + pgvector Strategy B, top-k=3 | test | 2/5 | 3/5 | 3/5 | 46.49 ms |

### Lựa chọn cấu hình

Dense retrieval (B) được chọn dựa trên dev vì có Hit@1, Evidence Hit@3 và Context Evidence Rate cao nhất. Hybrid RRF không vượt dense trên benchmark này; TF-IDF nhanh hơn nhưng yếu hơn về recall ngữ nghĩa.

Sau khi chọn B trên dev, cấu hình được giữ nguyên khi chạy test: Strategy B, E5-small-v2, top-k=3 và context budget 2000 ký tự.

### Phân tích lỗi retrieval

- Dense retrieval tốt hơn TF-IDF ở các câu diễn đạt khác với câu chữ gốc, ví dụ CDF và IQR.
- Test chỉ đạt Evidence Hit@3 là 3/5, cho thấy một số evidence bị nằm ngoài top-3.
- Ví dụ `q05`: evidence chuẩn nằm ở chunk `b-thinkstats2-c3-c5-p32-001`, nhưng dense top-3 trả về các chunk cùng trang 32 khác; agent do đó abstain an toàn.
- PDF extraction có hiện tượng dính chữ như `Aprobabilityis`. Evidence matching được chuẩn hóa Unicode, khoảng trắng, dấu trừ và dấu ngoặc kép khi đánh giá; text gốc vẫn được giữ cho context.

## 3. Agent và context engineering

Agent dùng vòng lặp tối đa 3 lượt:

1. Model nhận question và retrieval state.
2. Nếu chưa có context, model trả về action `search`.
3. Retriever dense trả các chunk top-k từ pgvector.
4. Context được deduplicate theo chunk ID và giới hạn 2000 ký tự.
5. Model trả `answer` kèm citation hoặc `abstain`.
6. Citation chỉ hợp lệ về cấu trúc khi ID xuất hiện trong context và trong answer dưới dạng `[chunk-id]`.

Prompt agent được viết bằng tiếng Anh, yêu cầu:

- search trước khi abstain;
- chỉ trả lời từ retrieved context;
- không dùng kiến thức ngoài corpus;
- copy ID thật từ `context_ids` vào answer và mảng citations;
- không dùng placeholder như `[chunk-id]`.

## 4. Agent test và audit thủ công

| Chỉ số test | Kết quả |
|---|---:|
| Tool use | 7/7 |
| Tổng lượt search | 7 |
| Provider errors | 0 |
| Grounded answer grade 0 | 5/5 |
| Grounded answer grade 1 | 0/5 |
| Grounded answer grade 2 | 0/5 |
| Mean grounded-answer grade | 0.0/2 |
| Citation validity | 1/2 |
| Answer có citation | 2/2 |
| Abstention đúng trên câu thiếu evidence | 2/2 |
| False abstention trên câu answerable | 3/5 |

### Ví dụ trace

| Query | Kết quả | Nhận xét |
|---|---|---|
| q16 — sampling with replacement | Abstain | False abstention: corpus có evidence ở trang 56. |
| q17 — quartiles | Answered với citation `p53-002` | Answer chứa placeholder; citation không hỗ trợ trực tiếp quartiles 25/50/75. |
| q18 — `Prob` method | Abstain | False abstention: corpus có evidence ở trang 32. |
| q19 — inverse CDF sampling | Answered với citation `p69-002` | Citation hỗ trợ inverse CDF, nhưng answer chỉ là ID nên grade 0. |
| q20 — lognormal model | Abstain | False abstention. |
| q23, q24 — câu ngoài corpus | Abstain | Đúng; agent không bịa câu trả lời. |

## 5. Kết luận và giới hạn

Dense E5 + pgvector là cấu hình retrieval tốt nhất trên dev và được giữ cố định cho test. Retrieval có thể lấy được evidence cho nhiều câu, nhưng test cho thấy recall top-3 vẫn còn hạn chế.

Agent local Qwen 3B sử dụng tool đúng cho toàn bộ test và abstain đúng với câu ngoài corpus. Tuy nhiên chất lượng generation/citation chưa tốt: một số output copy placeholder trong schema thay vì tạo câu trả lời hoàn chỉnh; đồng thời có false abstention ở câu answerable. Đây là giới hạn của model local nhỏ và prompt/citation compliance, không phải lý do để sửa kết quả test sau khi đã chạy.

## 6. Cách tái lập

1. Bật Docker Desktop và database Day 6:

```powershell
cd "<DAY6_REPO>\starter"
docker compose -p day6_student_starter up -d --wait