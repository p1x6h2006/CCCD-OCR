"""
OCR tự động số hóa CCCD và biểu mẫu
Pipeline: Ảnh -> Canny/Contour -> Perspective Transform -> EasyOCR -> Regex/Heuristic -> Form
Chạy:  streamlit run app.py
"""
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import easyocr

CARD_W, CARD_H = 856, 540  # tỉ lệ thẻ ID-1 (85.6 x 54 mm)

# Lưu mô hình EasyOCR ngay trong thư mục dự án (ví dụ D:\CODE_VISUAL\cccd-ocr\models)
# thay vì C:\Users\<tên>\.EasyOCR -> không làm đầy ổ C.
MODEL_DIR = Path(__file__).resolve().parent / "models"

STOP_WORDS = (
    "CONG HOA", "DOC LAP", "SOCIALIST", "CAN CUOC", "CITIZEN", "IDENTITY",
    "VIET NAM", "FULL NAME", "HO VA TEN", "QUOC TICH", "NATIONALITY",
)


# ----------------------------------------------------------------------------
# 1. TIỀN XỬ LÝ ẢNH: tìm thẻ + nắn phối cảnh
# ----------------------------------------------------------------------------
def order_points(pts):
    """Sắp xếp 4 điểm: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def find_card(img):
    """Trả về (4 điểm góc theo ảnh gốc | None, ảnh cạnh để minh họa)."""
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    min_area = 0.2 * small.shape[0] * small.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area and cv2.isContourConvex(approx):
            return approx.reshape(4, 2) / scale, edges
    return None, edges


def warp_card(img, quad):
    rect = order_points(quad)
    w = np.linalg.norm(rect[0] - rect[1])
    h = np.linalg.norm(rect[0] - rect[3])
    if h > w:  # thẻ bị chụp dọc -> xoay thứ tự điểm để ra ảnh nằm ngang
        rect = np.roll(rect, -1, axis=0)
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (CARD_W, CARD_H))


def enhance(img):
    """Tăng tương phản cục bộ (CLAHE) để OCR đọc tốt hơn."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------
# 2. OCR
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình EasyOCR (lần đầu hơi lâu)...")
def get_reader():
    MODEL_DIR.mkdir(exist_ok=True)
    return easyocr.Reader(
        ["vi", "en"], gpu=False,
        model_storage_directory=str(MODEL_DIR),
        user_network_directory=str(MODEL_DIR / "user_network"),
    )


def ocr_lines(img):
    """Chạy EasyOCR rồi gom các box cùng hàng thành một dòng."""
    results = get_reader().readtext(img, detail=1, paragraph=False)
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": text.strip(), "y": float(np.mean(ys)), "x": min(xs),
                      "h": max(ys) - min(ys), "conf": conf})
    if not items:
        return []
    items.sort(key=lambda d: d["y"])
    thr = 0.6 * float(np.median([d["h"] for d in items]))
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["y"] - np.mean([c["y"] for c in cur])) <= thr:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    rows.append(cur)
    return [" ".join(i["text"] for i in sorted(r, key=lambda d: d["x"])) for r in rows]


# ----------------------------------------------------------------------------
# 3. TRÍCH XUẤT THÔNG TIN (regex + heuristic)
# ----------------------------------------------------------------------------
def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").upper()


def is_upper_name(s):
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters) and " " in s.strip()


def clean_name(s):
    s = re.sub(r"[^A-Za-zÀ-ỹĐđ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().upper()


def parse_fields(lines):
    out = {"id": "", "name": "", "dob": ""}

    # --- Số CCCD: 12 chữ số ---
    for ln in lines:
        compact = re.sub(r"[ .:\-]", "", ln)
        m = re.search(r"\d{12}", compact)
        if m:
            out["id"] = m.group(0)
            break

    # --- Ngày sinh: dd/mm/yyyy ---
    date_re = r"(\d{2})\s*[/\-.]\s*(\d{2})\s*[/\-.]\s*(\d{4})"
    for ln in lines:
        if "SINH" in strip_accents(ln) or "BIRTH" in strip_accents(ln):
            m = re.search(date_re, ln)
            if m:
                out["dob"] = "/".join(m.groups())
                break
    if not out["dob"]:
        for ln in lines:
            m = re.search(date_re, ln)
            if m:
                out["dob"] = "/".join(m.groups())
                break

    # --- Họ tên: dòng ngay sau nhãn "Họ và tên / Full name" ---
    for i, ln in enumerate(lines):
        key = strip_accents(ln)
        if "HO VA TEN" in key or "FULL NAME" in key:
            after = ln.split(":")[-1] if ":" in ln else ""
            if is_upper_name(after):
                out["name"] = clean_name(after)
                break
            for nxt in lines[i + 1:i + 3]:
                if is_upper_name(nxt) and not any(w in strip_accents(nxt) for w in STOP_WORDS):
                    out["name"] = clean_name(nxt)
                    break
            if out["name"]:
                break

    # --- Fallback: dòng IN HOA dài nhất, không chứa từ khóa tiêu đề ---
    if not out["name"]:
        cands = [ln for ln in lines
                 if is_upper_name(ln) and not re.search(r"\d", ln)
                 and not any(w in strip_accents(ln) for w in STOP_WORDS)]
        if cands:
            out["name"] = clean_name(max(cands, key=len))
    return out


# ----------------------------------------------------------------------------
# 4. GIAO DIỆN STREAMLIT
# ----------------------------------------------------------------------------
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="OCR CCCD", page_icon="🪪", layout="wide")
st.title("🪪 OCR tự động số hóa CCCD → Biểu mẫu đăng ký")
st.caption("OpenCV (Canny + Perspective Transform) · EasyOCR · Streamlit")

mode = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Chụp bằng camera"], horizontal=True)
src = st.file_uploader("Chọn ảnh CCCD (mặt trước)", type=["jpg", "jpeg", "png"]) \
    if mode == "Tải ảnh lên" else st.camera_input("Đưa CCCD vào khung hình")

if src is not None:
    img = cv2.imdecode(np.frombuffer(src.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    quad, edges = find_card(img)

    if quad is not None:
        card = warp_card(img, quad)
        preview = img.copy()
        cv2.polylines(preview, [quad.astype(int)], True, (0, 255, 0), max(3, img.shape[1] // 300))
        status = "✅ Phát hiện được 4 góc thẻ, đã nắn phẳng."
    else:
        card = cv2.resize(img, (CARD_W, CARD_H))
        preview = img
        status = "⚠️ Không tìm thấy khung thẻ, dùng toàn bộ ảnh. Hãy chụp thẻ trên nền tương phản."

    with st.spinner("Đang nhận dạng chữ..."):
        lines = ocr_lines(enhance(card))
        fields = parse_fields(lines)

    st.info(status)
    c1, c2, c3, c4 = st.columns(4)
    c1.image(bgr2rgb(img), caption="1. Ảnh gốc", use_container_width=True)
    c2.image(edges, caption="2. Canny edges", use_container_width=True)
    c3.image(bgr2rgb(preview), caption="3. Khung thẻ phát hiện", use_container_width=True)
    c4.image(bgr2rgb(card), caption="4. Sau Perspective Transform", use_container_width=True)

    st.subheader("📝 Biểu mẫu đăng ký (tự động điền, có thể sửa)")
    uid = src.file_id if hasattr(src, "file_id") else str(len(src.getvalue()))
    with st.form("register"):
        name = st.text_input("Họ và tên", fields["name"], key=f"n{uid}")
        cid = st.text_input("Số CCCD", fields["id"], key=f"i{uid}")
        dob = st.text_input("Ngày sinh (dd/mm/yyyy)", fields["dob"], key=f"d{uid}")
        if st.form_submit_button("Đăng ký"):
            st.success(f"Đã ghi nhận: {name} | {cid} | {dob}")

    with st.expander("Văn bản OCR thô (để debug / viết báo cáo)"):
        st.code("\n".join(lines))
else:
    st.write("👆 Hãy tải ảnh hoặc chụp ảnh CCCD để bắt đầu.")
