Họ tên: Trần Long Khánh — MSSV 2A202602538

`dataset_version`: d10-21e4070301

Chạy trên Kaggle T4 (1 card), transformers 5.19.0 · peft 0.21.2 · bitsandbytes 0.50.2. E0–E4: một lần "Save & Run All" (run 357955219, 4 giờ 26 phút); E5: run 358086382.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | zero-shot | 0 | – | – | 0.086 | 0.062–0.115 | 0.90 | 0.110 | 0.024 | 0.203 |
| E1 | LoRA LLM | 33,030,144 | 8.05 | 43.2 | 0.269 | 0.232–0.310 | 1.19 | 0.182 | 0.192 | 0.513 |
| E2 | + projector | 142,135,296 | 8.54 | 43.6 | 0.349 | 0.307–0.394 | 1.04 | 0.220 | 0.262 | 0.636 |
| E3 | + LoRA ViT | 148,426,752 | 8.73 | 51.5 | **0.447** | 0.407–0.490 | 0.86 | 0.326 | 0.366 | 0.696 |
| E4 | coord = norm1000 (= E3, chấm lại) | 148,426,752 | 8.73 | 51.5 | 0.447 | 0.407–0.490 | 0.86 | 0.326 | 0.366 | 0.696 |
| E4 | coord = pixel | 148,426,752 | 8.70 | 51.0 | 0.392 | 0.350–0.435 | 0.94 | 0.274 | 0.304 | 0.625 |
| E5 | GRPO từ E3 (bonus) | 39,321,600 | 8.03 | 45.7 | 0.452 | 0.411–0.493 | 0.77 | 0.339 | 0.367 | 0.699 |

`parse_valid_rate`: E0 0.926 · E1 0.994 · E2–E5 1.000. VRAM = `torch.cuda.max_memory_allocated()`; phút = chỉ vòng huấn luyện.
Ảnh minh hoạ: `report/figures/e{0,1,3}_examples.png` (3 ảnh tốt nhất + 3 tệ nhất; GT nét đứt xanh lá, dự đoán nét liền).

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Tổng ≈ 4,44B: ViT ≈ 415M (24 block), trong đó projector = `visual.merger` (≈ 27M) + `visual.deepstack_merger_list` (≈ 82M, lấy đặc trưng ở lớp 5/11/17); LLM + embedding ≈ 4,0B (36 lớp).

- **E1**: LoRA r=16 trên 252 module LLM → **33,03M**. ViT, projector đóng băng.
- **E2**: + mở toàn bộ projector → 142,14M − 33,03M = **109,11M** (khớp 27M + 82M).
- **E3**: + LoRA trên 96 module ViT → **+6,29M** (tổng 348 module).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
**0.926** (23/311 hỏng). Model không bọc ```json và dùng đúng thang 0–1000. Hai lỗi chính:
1. **Lặp vô hạn bị cắt → JSON hỏng**: cả 23 ca hỏng đều rải một vết nứt thành chuỗi box crack đến hết 256 token:
   `[{"label": "crack", "bbox_2d": [575, 420, 600, 440]}, {"label": "crack", "bbox_2d": [575, 440, 600, 460]}, …`
2. **Bịa/sai box**: 203/320 box là crack nhưng F1 crack chỉ 0.024; 25 box chiếm > 25% khung (GT trung vị ≈ 0,7%). Kết quả P 0.122, R 0.067.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Chấm lại đúng file dự đoán của E1 bằng `--coord pixel`: **F1 0.269 → 0.000** (cả 519 box thành FP) dù JSON vẫn hợp lệ 99,4%. Trọng tài đổi `x·1000/640`, `y·1000/360` nên box bị kéo giãn 1,56× / 2,78× và lệch vị trí → IoU < 0.5 với mọi GT. Đây là lỗi "im lặng".
Chiều ngược lại (E4): train hẳn ở thang pixel cho F1 0.392 < 0.447, vì 0–1000 là quy ước grounding lúc tiền huấn luyện của Qwen3-VL. Ngoài ra đã kiểm tra round-trip YOLO → JSON → parser → pixel lệch ≤ 1 px.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Một ví dụ có 220 token ảnh + ~90 token prompt, còn đáp án chỉ ~20 token/box. Tôi gán `labels = -100` cho toàn bộ tiền tố và kiểm tra phần còn lại giải mã ra đúng JSON + `<|im_end|>`.
Nếu không mask: loss bị chi phối bởi việc dự đoán 220 token `<|image_pad|>` (vô nghĩa) và một prompt giống hệt ở mọi mẫu (học thuộc). Loss giảm "đẹp" nhưng tín hiệu cho toạ độ/nhãn bị loãng, và model có thể học cả việc nói như user. Với mask, loss E1 giảm 1.28 → 0.56. (Không chạy biến thể không mask nên đây là lập luận.)

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
`Linear4bit` lưu mã NF4 nén (2 giá trị/byte `uint8`) + scale theo khối, không phải tham số liên tục nên không cập nhật trực tiếp được, chỉ gắn LoRA lên trên.
**Đã gặp thật**: `llm_int8_skip_modules=["visual", ...]` không khớp `model.visual.merger…` trong transformers 5.19 (so khớp từ gốc), nên projector vẫn bị 4-bit. Dấu hiệu duy nhất: E3 chạy thử chỉ có 93,9M tham số thay vì ≈ 148M (hụt đúng nửa 109M do tensor nén). Sửa: dùng `"model.visual"` + assert merger là `nn.Linear` fp16 → đúng 148.426.752.
Phải dùng **fp32** vì `GradScaler` không unscale được grad fp16, và bước cập nhật nhỏ (lr ≤ 2e-4) bị làm tròn mất ở fp16. Thêm bẫy: torch mới giữ `grad_dtype` fp16 sau khi đổi `.data` → phải đặt `p.grad_dtype = torch.float32`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh là frame video, frame liền nhau gần như trùng. Chia ngẫu nhiên thì ảnh test có "bản sao" trong train → F1 đo khả năng nhớ, không phải tổng quát sang đoạn đường mới. Split gom theo phiên quay; đã assert không phiên nào nằm ở hai split.

| Split | Ảnh | pothole | crack | manhole |
| --- | --- | --- | --- | --- |
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Hệ quả: phân bố lớp lệch (val chỉ 66 pothole).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | Δ F1 | Δ VRAM | Δ thời gian | CI95 chồng nhau? |
| --- | --- | --- | --- | --- |
| E1 → E2 (+ projector) | +0.080 | +0.49 GB | +0.4 phút | chạm nhau (0.307–0.310) |
| E2 → E3 (+ LoRA ViT) | +0.098 | +0.19 GB | +7.9 phút | không |
| E1 → E3 | +0.178 | +0.68 GB | +8.3 phút | không |

Mở projector gần như miễn phí (gradient không đi vào ViT). LoRA ViT tăng đều cả 3 lớp và giảm FP/ảnh 1.04 → 0.86, đổi lại +18% thời gian. **E3 đáng nhất**: +0.18 F1, VRAM 8,7 GB vẫn dư trên T4. Với box nhỏ và đặc trưng mặt đường, cho encoder thị giác thích nghi quan trọng hơn chỉ chỉnh LLM.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 (bootstrap theo ảnh) rộng ≈ **±0,04**. Chắc chắn: E0 < E1 < E3, E2 < E3. Chưa chắc: E1 vs E2, E3 vs E4-pixel, E3 vs E5 (khoảng chồng nhau). F1 từng lớp càng kém tin cậy (pothole E3 chỉ 31 TP).
CI thật còn rộng hơn: frame trong cùng phiên tương quan mạnh nên bootstrap theo ảnh đánh giá thấp phương sai (nên bootstrap theo phiên), và mỗi setup chỉ 1 seed.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
E3: **pothole kém nhất** (0.326), crack 0.366, manhole 0.696.
- **Pothole** (P 0.449, R 0.256): chủ yếu **bỏ sót** (90 FN / 38 FP, chỉ đoán 69 box cho 121 GT). Ổ gà nhỏ, ở xa, dễ lẫn vết vá/bóng; dữ liệu train tập trung ít phiên.
- **Crack** (P 0.358, R 0.375): **ranh giới mơ hồ**, một vết nứt có thể là 1 box dài hay nhiều box nhỏ. Hàng dưới `e3_examples.png`: box ngắn trong GT dài (IoU < 0.5) và một vết nứt bị tách 3 box (FP 3, FN 1).
- **Manhole**: hình dạng đều, mép rõ → dễ nhất; có ca nhầm thành crack khi nằm trong vùng nứt.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Đổi **một biến: thang toạ độ**, giữ nguyên E3. `pixel` đổi đồng thời nhãn, prompt và `--coord pixel`; `norm1000` là chính E3 chấm lại.

| coord | F1 | CI95 | F1 pothole / crack / manhole |
| --- | --- | --- | --- |
| norm1000 | 0.447 | 0.407–0.490 | 0.326 / 0.366 / 0.696 |
| pixel | 0.392 | 0.350–0.435 | 0.274 / 0.304 / 0.625 |

Pixel kém hơn 0.055 và cả 3 lớp đều giảm, tài nguyên như nhau. Hợp lý vì model phải học lại quy ước toạ độ mới (lại không đẳng hướng 640 × 360). Nhưng CI chồng nhau (0.407–0.435) nên **chưa khẳng định được ở mức 95%**; cần nhiều seed hoặc test lớn hơn. Kết luận an toàn: giữ thang 0–1000.

## E5 (bonus) — GRPO với reward IoU
**Thiết lập** (`lab_day10_e5.ipynb`, trl 1.9.2): từ adapter E3, chỉ train LoRA (39,3M; projector đóng băng), 60 bước, lr 5e-6, `beta=0`, 4 lần sinh/prompt, `max_completion_length=128`, fp16. Reward `0.2·valid + 0.8·ΣIoU / max(n_gt, n_pred, 1)`, có 6 test pass trước khi chạy. Chỉ lấy ảnh train có đáp án < 120 token để JSON đúng không bị cắt. Log: `report/e5_grpo_log.json`.

**Log**: reward theo batch dao động 0.36–0.51, không có xu hướng tăng rõ (3 lần log đầu 0.437, cuối 0.453); độ dài completion ổn định 47–60 token, bị cắt ≤ 6%.

**Reward hacking?** (trên test, `report/e5_analysis.json`)

| | Output `[]` | JSON hỏng | Box/ảnh | Box > 25% khung | Reward TB |
| --- | --- | --- | --- | --- | --- |
| GT | – | – | 1.87 | – | – |
| E3 (SFT) | 0 | 0 | 1.65 | 0 | 0.4745 |
| E5 (GRPO) | 0 | 0 | 1.54 | 0 | 0.4727 |

**Không có**: không trả `[]` để lấy điểm định dạng, không vẽ box khổng lồ. Model chỉ thận trọng hơn (ít box hơn, do mẫu số phạt box thừa): precision 0.479 → 0.502, FP/ảnh 0.86 → 0.77, recall 0.420 → 0.412.

**So với SFT**: F1 0.447 → 0.452, CI gần như trùng → **không cải thiện có ý nghĩa**. Nguyên nhân: chỉ 120 ảnh với lr nhỏ, 4 lần sinh/prompt nên advantage nhiễu. Cần nhiều bước và nhiều lần sinh hơn (vLLM), có thể bật KL.

## Nhìn lại
- **Ba lỗi môi trường công cụ chấm không bắt được** (evidence vẫn "hợp lệ"):
  1. `"visual"` không khớp tên module → projector bị 4-bit âm thầm; phát hiện nhờ đối chiếu số tham số.
  2. Torch giữ `grad_dtype` fp16 sau khi đổi `.data` sang fp32 → GradScaler báo lỗi.
  3. Image Kaggle mới có `torchao 0.10` → peft báo `ImportError`; chỉ lộ ra ở bản chạy commit, sửa bằng cách gỡ torchao.

  Hai lỗi đầu bắt được nhờ smoke test 2 bước với E3 trước khi train thật.
- **Dữ liệu**: 1 nhãn train bị parser từ chối vì box quá nhỏ, làm tròn về 0–1000 thì x1 = x2.
- **Chưa hiểu hết**: vì sao chỉ *bias* projector kẹt `grad_dtype` fp16; chưa đo tác động của việc không mask loss.
- **E5**: trl tự ép LoRA của model 4-bit về bf16 và mặc định bật bf16 (T4 không hỗ trợ) → phải đặt `fp16=True, bf16=False` và ép LoRA về fp32.
