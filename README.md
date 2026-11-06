# OCR tự động số hóa CCCD → Biểu mẫu đăng ký

Pipeline: Ảnh → Canny/Contour → Perspective Transform → EasyOCR → Regex/Heuristic → Form (Streamlit).

> Dự án học tập. Chỉ dùng dữ liệu **mô phỏng** hoặc thẻ **của chính bạn / người đã đồng ý**. Không đưa ảnh CCCD thật lên GitHub.

## Cấu trúc

```
cccd-ocr/
├── app.py                  # ứng dụng Streamlit
├── generate_synthetic.py   # sinh ảnh thẻ mô phỏng + nhãn
├── requirements.txt
├── setup.bat / run.bat     # cài đặt & chạy trên Windows (toàn bộ nằm ở ổ D)
├── faces/                  # bỏ ảnh chân dung (jpg/png) vào đây
├── README.md
└── .gitignore
```

## Cài đặt (Windows, đặt tại `D:\CODE_VISUAL\cccd-ocr`)

```bat
cd /d D:\CODE_VISUAL\cccd-ocr
setup.bat
run.bat
```

`setup.bat` tạo `venv`, cache pip/torch/tạm đều trong thư mục dự án trên ổ D. Mô hình EasyOCR tự tải vào `models/` (lần chạy đầu, cần mạng).

Cài thủ công:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tạo ảnh CCCD mô phỏng để huấn luyện / đánh giá

1. Chuẩn bị 10–50 ảnh chân dung (jpg/png) hợp lệ (xem mục dưới) và bỏ vào `faces/`. Nếu để trống sẽ dùng hình bóng người.
2. Sinh dữ liệu:

```bat
python generate_synthetic.py --n 500 --faces faces --out dataset
```

Kết quả:

```
dataset/
├── images/syn_0001.jpg ...      # ảnh thẻ trên nền ngẫu nhiên, xoay/nghiêng/mờ/chói/nhiễu
├── labels.csv                   # filename,id,name,dob  (để đo độ chính xác OCR)
└── labels_yolo/syn_0001.txt     # class cx cy w h (0=id, 1=name, 2=dob) cho YOLO
```

Tham số: `--n` số ảnh, `--seed` để tái lập, `--font` đường dẫn `.ttf` nếu máy không có Arial.

3. Đánh giá: chạy `app.py` trên ảnh trong `dataset/images/`, so với `labels.csv`.
4. Huấn luyện bộ phát hiện vùng (YOLO): chia `images/` + `labels_yolo/` thành train/val (80/20) rồi dùng với YOLOv8.

### Nguồn ảnh chân dung hợp lệ
- Ảnh của chính bạn hoặc người **đồng ý bằng văn bản**.
- Ảnh do AI tạo ra (người không có thật).
- Bộ ảnh có giấy phép cho phép dùng (ví dụ các bộ dữ liệu mở có ghi rõ license).

Không dùng ảnh CCCD thật của người khác. Thẻ sinh ra luôn có dòng chữ **"MẪU - DÙNG CHO HỌC TẬP"**, đừng xóa dòng này.

### Chụp ảnh thật để kiểm thử (tùy chọn)
Chỉ dùng thẻ của chính bạn, che số/ảnh khi chia sẻ. Chụp trên nền tương phản, đủ sáng, thấy đủ 4 góc.
