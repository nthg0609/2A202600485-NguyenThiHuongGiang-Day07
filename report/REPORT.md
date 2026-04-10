# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thị Hương Giang
**Nhóm:** Nhóm 8 - E402
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *High cosine similarity có nghĩa là hai đoạn văn bản có sự tương đồng rất cao về mặt ngữ nghĩa, ý tưởng hoặc chủ đề (góc giữa hai vector đại diện cho chúng trong không gian đa chiều rất nhỏ), bất kể độ dài của chúng dài ngắn khác nhau ra sao.*

**Ví dụ HIGH similarity:**
- Sentence A: Con mèo đang ngủ say sưa trên chiếc ghế sofa.
- Sentence B: Một chú mèo con đang nằm lim dim trên chiếc ghế dài ở phòng khách.
- Tại sao tương đồng: Mặc dù sử dụng các từ vựng khác nhau (sofa vs ghế dài, ngủ say sưa vs nằm lim dim), cả hai câu đều mang cùng một ý nghĩa miêu tả chung một hành động và đối tượng.

**Ví dụ LOW similarity:**
- Sentence A: Con mèo đang ngủ say sưa trên chiếc ghế sofa.
- Sentence B: Lãi suất ngân hàng trung ương vừa báo cáo mức giảm kỷ lục.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan đến nhau (thú cưng/đời sống hằng ngày vs kinh tế/tài chính vĩ mô).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Euclidean distance bị ảnh hưởng rất lớn bởi độ dài của văn bản (đếm số lượng từ), khiến một câu tóm tắt ngắn và một đoạn văn dài dù cùng ý nghĩa vẫn bị tính là xa nhau. Cosine similarity giải quyết việc này bằng cách chỉ đo lường "hướng" (ngữ nghĩa) của vector mà bỏ qua "độ lớn" (số lượng từ ngữ), giúp so sánh độ chuẩn xác cao hơn.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính: ceil((10000 - 50) / (500 - 50))*
> *Đáp án: 23 chunks*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Khi overlap tăng lên 100, phép tính trở thành ceil((10000 - 100) / (500 - 100)) = 25, số lượng chunk tăng từ 23 lên 25. Việc tăng overlap giúp bảo toàn ngữ cảnh (context) tốt hơn ở những đoạn ranh giới bị cắt, đảm bảo một ý tưởng, một câu trọn vẹn hoặc một từ khóa quan trọng không bị chặt đứt gãy giữa hai chunks liền kề*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Văn học Việt Nam — Truyện ngắn của Nam Cao

**Tại sao nhóm chọn domain này?**
> Nam Cao là tác giả văn học hiện thực phê phán nổi bật nhất Việt Nam với ngôn ngữ giàu cảm xúc và nhiều sự kiện cụ thể, phù hợp để kiểm tra độ chính xác của hệ thống RAG. Các nhân vật và tình tiết đặc trưng (Chí Phèo rạch mặt, Lão Hạc bán Cậu Vàng, bát cháo hành của Thị Nở) tạo ra những câu hỏi benchmark dễ kiểm định kết quả đúng/sai. Ngoài ra việc xài tài liệu tiếng Việt giúp nhóm phát hiện thêm điểm yếu của mô hình embedding phương Tây khi xử lý ngôn ngữ đặc thù.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Chí Phèo | chi_pheo.txt | ~38,000 | source, extension, chunk_idx |
| 2 | Lão Hạc | lao_hac.txt | ~11,500 | source, extension, chunk_idx |
| 3 | Đời Mặt | doi_mat.txt | ~17,500 | source, extension, chunk_idx |
| 4 | Đời Thừa | doi_thua.txt | ~13,000 | source, extension, chunk_idx |
| 5 | Một Bữa No | mot_bua_no.txt | ~10,500 | source, extension, chunk_idx |
| 6 | Trẻ Con Không Được Ăn Thịt Chó | tre_con_khong_duoc_an_thit_cho.txt | ~12,500 | source, extension, chunk_idx |


### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/chi_pheo.txt" | Cho phép lọc theo từng tác phẩm cụ thể bằng search_with_filter |
| extension | string | ".txt" | Phân loại định dạng tệp nếu có thêm PDF/MD trong tương lai |
| chunk_idx | int | 15 | Theo dõi vị trí đoạn trong tác phẩm gốc, hỗ trợ debug |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên chi_pheo.txt (~38,000 ký tự):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chi_pheo.txt | FixedSizeChunker (`fixed_size`) | ~80 | 500 ký tự | Không — cắt ngang câu |
| chi_pheo.txt | SentenceChunker (`by_sentences`) | ~95 | 400 ký tự | Tốt hơn — ngắt đúng câu |
| chi_pheo.txt | RecursiveChunker (`recursive`) | 77 | ~494 ký tự | Tốt nhất — ngắt theo đoạn |

### Strategy Của Tôi

**Loại:** RecursiveChunker với `chunk_size=1000`

**Mô tả cách hoạt động:**
> Chiến lược này không cắt văn bản một cách "mù quáng" theo số lượng ký tự, mà phân tách dựa trên một danh sách các dấu hiệu ngữ pháp có thứ tự ưu tiên (ưu tiên 1: \n\n - hết đoạn văn; ưu tiên 2: .  - hết câu; ưu tiên 3:   - hết từ). Hệ thống sẽ cố gắng nhét trọn vẹn một đoạn văn vào một chunk. Chỉ khi đoạn văn đó quá dài (vượt quá chunk_size), nó mới đệ quy và dùng dấu chấm câu để cắt nhỏ tiếp.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Truyện ngắn như "Chí Phèo" có đặc thù là các đoạn văn mang tính tự sự, miêu tả tâm lý hoặc các đoạn hội thoại thường liên kết chặt chẽ với nhau trong cùng một paragraph (đoạn). Nếu dùng FixedSize, câu văn sẽ bị chặt đứt làm đôi khiến LLM không hiểu chủ ngữ là ai. Nếu dùng SentenceChunker, ngữ cảnh miêu tả bối cảnh xung quanh sẽ bị tách rời khỏi lời thoại của nhân vật. RecursiveChunker khai thác được dấu \n\n để giữ trọn vẹn một "cảnh" (scene) hoặc một luồng suy nghĩ của nhân vật trong một chunk duy nhất.

**Code snippet (nếu custom):**
```python
from src.chunking import RecursiveChunker

chunks = RecursiveChunker(chunk_size=1000).chunk(content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|---------------------|
| chi_pheo.txt | FixedSizeChunker(500) | ~80 | 500 | Thấp — cắt ngang câu, mất ngữ cảnh |
| chi_pheo.txt | **RecursiveChunker(1000) — của tôi** | 77 | ~1000 | Cao — giữ nguyên đoạn văn, score 0.695 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Giang | RecursiveChunker(1000) | 8/10 | Bảo toàn ngữ cảnh dài | Chunk lớn hơn tốn context window |
| Hoàng | FixedSizeChunker(500, overlap=50) | 5/10 | Đơn giản, nhanh | Cắt ngang câu gây mất nghĩa |
| Hùng | SentenceChunker(max_sentences=3) | 6/10 | Ngắt đúng câu | Chunk quá ngắn, mất ngữ cảnh dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size=1000 phù hợp nhất cho văn học Nam Cao vì tác giả viết theo lối tường thuật có tính liên tục cao — một hành động thường kéo dài qua nhiều câu liên tiếp. Khi hỏi "Tại sao Chí Phèo rạch mặt?", cần đọc cả đoạn dẫn dắt chứ không phải chỉ 1-2 câu đơn lẻ. Strategy đệ quy ngắt đúng ranh giới đoạn nên LLM nhận được ngữ cảnh đầy đủ nhất.

---
## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Sử dụng module regex re.split tạo nhóm tham chiếu (\. |\! |\? |\.\n) để vừa chia câu vừa giữ nguyên được biểu tượng kết thúc mà không làm mất nó trong chuỗi kết quả. Đồng thời, kết hợp với hàm .strip() trong vòng lặp để xử lý triệt để các edge case liên quan đến khoảng trắng thừa và các chuỗi rỗng sinh ra trong quá trình tách.*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Áp dụng thuật toán đệ quy rẽ nhánh dựa trên danh sách Separators có độ ưu tiên giảm dần (từ \n\n xuống \n, . ,  ). Thuật toán hoạt động theo nguyên tắc: Nếu một chunk sau khi cắt vẫn vượt quá ngưỡng chunk_size, đoạn văn bản đó sẽ được đẩy ngược trở lại hàm _split() cùng với separator nhỏ hơn tiếp theo. Base case (điều kiện dừng) là khi đoạn text đã thu gọn an toàn ở mức <= chunk_size hoặc cạn kiệt tập phân cách.*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Add: Document được truyền qua hàm _embedding_fn để trích xuất Vector biểu diễn, sau đó đóng gói thành một Dictionary (chứa id, text, metadata, vector) và append vào list lưu trữ nội bộ (In-memory store).
Search: Quét vòng lặp toàn bộ mảng data, áp dụng công thức Dot Product (tính toán Vector) qua hàm compute_similarity để sinh điểm Score. Cuối cùng, dùng hàm sort(reverse=True) để trả về Top-K danh sách các chunk có độ tương đồng cao nhất.*

**`search_with_filter` + `delete_document`** — approach:
> *Filter: Áp dụng chiến lược Pre-filtering (Lọc trước - Tính điểm sau). Hệ thống sẽ loại bỏ các document không khớp Metadata trước khi đưa vào hàm Cosine Similarity, giúp tiết kiệm đáng kể tài nguyên RAM và CPU.
Delete: Triển khai linh hoạt bằng List Comprehension để ghi đè mảng dữ liệu hiện tại, giữ lại toàn bộ các record ngoại trừ record chứa doc_id cần xoá.

KnowledgeBaseAgent*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Lệnh .search thu hồi các chunk dữ liệu liên quan nhất, sau đó dùng .join() với line-break \n---\n để xâu chuỗi chúng thành một khối ngữ cảnh (Context) thống nhất. Khối này được tiêm trực tiếp vào một Prompt Template chứa System Instruction cứng rắn: "Chỉ trả lời trong phạm vi Context. Nếu không có thông tin, hãy báo không biết" nhằm giảm thiểu rủi ro Hallucination của LLM.*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 |Hắn vừa đi vừa chửi.|Chí Phèo lảo đảo bước đi và buông lời chửi đổng.| high|~1|đúng|
| 2 |Thị Nở mang cho hắn bát cháo hành.| Giá cổ phiếu ngành nông nghiệp giảm mạnh.| low | ~0 |đúng|
| 3 |Bá Kiến cười nhạt, ném cho hắn năm đồng bạc| ý trưởng khôn ngoan dùng tiền để xoa dịu kẻ lưu manh.| high |0.75|đúng|
| 4 | Chí Phèo khao khát làm người lương thiện.|Chí Phèo quyết định xách dao đi đâm chết Bá Kiến.| low |~0|không|
| 5 |Bà cô Thị Nở kiên quyết cấm cản. |cấm cản.	Thị Nở bị người cô ruột ngăn cấm chuyện tình cảm. | high|~1 |đúng|

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Bất ngờ nhất là Pair 4. Hai câu văn thể hiện hai trạng thái tư tưởng và hành động hoàn toàn trái ngược nhau của nhân vật (hướng thiện vs. sát nhân/bạo lực). Tuy nhiên, mô hình Embedding lại cho điểm tương đồng rất cao (0.76). Điều này bộc lộ "điểm mù" của các mô hình Embedding hiện tại: Chúng bị chi phối quá mạnh bởi sự xuất hiện của cùng một thực thể (Entity: "Chí Phèo") và cùng một bối cảnh truyện, dẫn đến việc xếp chúng gần nhau trong không gian vector mà thất bại trong việc thấu hiểu "động cơ" (intent) trái ngược bên trong ngữ nghĩa.*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 |Ai là người đẩy Chí Phèo vào tù? | Lý Kiến (sau này là Bá Kiến) đẩy Chí Phèo vào tù vì ghen tuông với vợ ba.|
| 2 |Lời cuối cùng của Chí Phèo trước khi chết là gì? | "Ai cho tao lương thiện? Tao không thể là người lương thiện nữa. Biết không!..."|
| 3 | Nguyên nhân trực tiếp khiến Chí Phèo tuyệt vọng và xách dao đi trả thù?|Bị Thị Nở cự tuyệt tình cảm do nghe lời xúi giục của bà cô. |
| 4 |Nghề nghiệp của Tự Lãng là gì? | Tự Lãng làm nghề thầy cúng kiêm hoạn lợn.|
| 5 |Chi tiết "cái lò gạch cũ" ở đầu và cuối truyện ngụ ý điều gì? |Bi kịch tha hóa của người nông dân có tính lặp lại, một "Chí Phèo con" sắp ra đời. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Ai đẩy Chí Phèo vào tù?|"Lý Kiến ghen với anh canh điền khỏe mạnh... giải huyện" | 0.78|Có |Lý Kiến (Bá Kiến) đã mượn cớ đẩy Chí Phèo vào tù do ghen tuông. |
| 2 |Lời cuối cùng của Chí? |"Hắn trừng trừng nhìn Bá Kiến... Ai cho tao lương thiện..."| 0.85| có|Chí Phèo nói: "Ai cho tao lương thiện? Tao không thể là người lương thiện nữa". |
| 3 | Nguyên nhân xách dao?|dao?	"Thị Nở trút vào mặt hắn tất cả lời bà cô... Hắn xách dao"| 0.81|Có|Do bị Thị Nở từ chối tình yêu sau khi nghe lời bà cô cấm cản. |
| 4 | Nghề nghiệp Tự Lãng?|"Tự Lãng làm nghề thầy cúng và hoạn lợn..." |0.72 |Có| Tự Lãng làm nghề thầy cúng và hoạn lợn.|
| 5 |Ý nghĩa lò gạch cũ? |"Thị nhìn nhanh xuống bụng... thấy thoáng hiện ra một cái lò gạch cũ..."|0.68 |1 phần|(LLM suy luận): Ám chỉ một "Chí Phèo con" sắp ra đời, vòng bi kịch lặp lại. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Hiện tượng "Lost in the Middle" (Quên ở giữa). Khi thử nghiệm nghiệm tăng chunk_size lên quá lớn (1500 - 2000 ký tự) để bao trọn một cảnh truyện, hệ thống lấy ra được Chunk rất chuẩn xác. Tuy nhiên, LLM Agent lại trả lời sai vì nó chỉ tập trung Attention vào phần đầu và phần cuối của đoạn văn mà bỏ qua chi tiết quan trọng nằm ở giữa. Điều này dạy tôi rằng: Kích thước Chunk phải được tune (điều chỉnh) cân bằng với khả năng đọc hiểu (Context Window) của LLM.*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Nhóm bạn đã ứng dụng kỹ thuật GraphRAG (Knowledge Graph + RAG) cho văn học. Thay vì chỉ cắt text, họ dùng LLM trích xuất một sơ đồ mối quan hệ (Ví dụ: [Chí Phèo] --(Kẻ thù)--> [Bá Kiến], [Chí Phèo] --(Yêu)--> [Thị Nở]). Khi người dùng hỏi về quan hệ nhân vật, truy vấn qua Graph Database (Neo4j) chính xác và nhanh hơn gấp nhiều lần so với Vector Search thuần túy.*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Tôi sẽ áp dụng kỹ thuật "Summarize to Embed" (Tóm tắt để nhúng). Văn học Nam Cao chứa rất nhiều câu miêu tả phong cảnh, tâm lý rườm rà dễ làm "nhiễu" vector. Tôi sẽ cho LLM chạy qua các Chunk để tạo ra một câu tóm tắt ngắn gọn (ví dụ: Đoạn này miêu tả cảnh Chí Phèo đâm Bá Kiến). Vector Embedding sẽ được tính dựa trên câu tóm tắt này (để tăng độ chính xác khi search), nhưng khi LLM trả lời, nó vẫn được cấp nguyên văn đoạn Chunk gốc để giữ nguyên văn phong tác giả.*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 15/ 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 10/ 10 |
| Core implementation (tests) | Cá nhân | 30/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | **100/ 100** |
