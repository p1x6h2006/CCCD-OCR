#!/usr/bin/env python3
"""
Sinh dữ liệu synthetic thẻ CCCD (MẪU HỌC TẬP) với nhãn OBB chính xác.

- Vẽ thẻ, ghi lại hộp CHỮ SÁT (tight) của từng trường: id / name / dob.
- Áp phép xoay + perspective lên thẻ rồi đặt lên nền ngẫu nhiên.
- Cùng ma trận biến đổi được áp lên 4 góc của mỗi hộp  =>  nhãn OBB khớp 100% với chữ.
- Xuất:
    out/images/{train,val}/*.jpg
    out/labels/{train,val}/*.txt        (OBB:  cls x1 y1 x2 y2 x3 y3 x4 y4, chuẩn hoá 0-1)
    out/labels_hbb/{train,val}/*.txt    (hộp thẳng: cls xc yc w h, ôm sát 4 góc - để so sánh)
    out/data.yaml
    out/debug/*.jpg                     (ảnh vẽ nhãn để kiểm tra, dùng --debug N)

Dùng:
    python gen_synthetic_obb.py --out dataset_obb --n 1500 --val 0.1 --debug 20
Train:
    yolo obb train model=yolo11n-obb.pt data=dataset_obb/data.yaml imgsz=960 epochs=100
"""
import argparse, math, os, random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["id", "name", "dob"]
CARD_W, CARD_H = 856, 540          # kích thước thẻ phẳng
OUT_W, OUT_H = 856, 540            # kích thước ảnh cuối (giống dataset cũ)

FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/tahomabd.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_CANDIDATES_REG = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/Library/Fonts/Arial.ttf",
]
FONT_MONO = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
             "C:/Windows/Fonts/consolab.ttf"]


def find_font(cands, extra=None):
    for p in ([extra] if extra else []) + cands:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("Không tìm thấy font .ttf, hãy dùng --font /duong/dan/font.ttf (cần hỗ trợ tiếng Việt)")


HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
DEM = ["Văn", "Thị", "Minh", "Xuân", "Quốc", "Thanh", "Hữu", "Ngọc", "Phương", "Đức", "Anh", "Bảo", "Gia", "Hoài"]
TEN = ["Hiếu", "Loan", "Trang", "Tuấn", "Hương", "Dũng", "Linh", "Phúc", "Thảo", "Khánh", "Nam", "Quỳnh", "Sơn", "Ánh", "Yến", "Đạt", "Trường"]
QUE = ["Thanh Hóa", "Quảng Nam", "Khánh Hòa", "Bình Thuận", "Hà Nội", "Nghệ An", "Đà Nẵng", "Cần Thơ", "Long An"]


def rand_name():
    parts = [random.choice(HO)]
    if random.random() < 0.85:
        parts.append(random.choice(DEM))
    if random.random() < 0.25:
        parts.append(random.choice(DEM))
    parts.append(random.choice(TEN))
    return " ".join(parts).upper()


def rand_id():
    return "".join(random.choice("0123456789") for _ in range(12))


def rand_dob():
    return f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(1950,2008)}"


def grad(w, h, c1, c2):
    t = np.linspace(0, 1, w)[None, :, None]
    a = np.array(c1, np.float32)[None, None, :]
    b = np.array(c2, np.float32)[None, None, :]
    img = a * (1 - t) + b * t
    if random.random() < 0.5:
        img = np.transpose(img, (1, 0, 2)) if w == h else img
    return np.repeat(img, h, axis=0).astype(np.uint8)


def draw_card(fonts):
    """Trả về (ảnh thẻ BGR, dict cls -> [x0,y0,x1,y1] sát chữ)."""
    base = random.choice([(205, 240, 235), (225, 245, 225), (215, 235, 250), (240, 240, 230), (235, 235, 245)])
    c2 = tuple(int(min(255, v + random.randint(-25, 15))) for v in base)
    card = grad(CARD_W, CARD_H, base, c2)
    im = Image.fromarray(card)
    d = ImageDraw.Draw(im)
    dark = (30, 30, 30)

    # tiêu đề
    d.text((CARD_W - 40, 28), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", font=fonts["s"], fill=(70, 70, 70), anchor="ra")
    d.text((CARD_W - 40, 54), "Độc lập - Tự do - Hạnh phúc", font=fonts["s"], fill=(70, 70, 70), anchor="ra")
    d.text((CARD_W // 2 + 60, 105), "CĂN CƯỚC CÔNG DÂN", font=fonts["title"], fill=(190, 40, 50), anchor="ma")
    # ảnh chân dung giả
    px, py = 38, 230
    pc = tuple(random.randint(40, 200) for _ in range(3))
    d.rectangle([px, py, px + 150, py + 200], fill=pc)
    d.ellipse([px + 40, py + 40, px + 110, py + 120], fill=(225, 190, 165))
    d.rectangle([px + 20, py + 130, px + 130, py + 200], fill=(240, 240, 240))
    # QR giả
    qx, qy = 38, 30
    for i in range(10):
        for j in range(10):
            if random.random() < 0.5:
                d.rectangle([qx + i * 8, qy + j * 8, qx + i * 8 + 7, qy + j * 8 + 7], fill=(40, 40, 40))

    X = 215
    boxes = {}

    def put(cls, text, y, font, fill, label, pad=3):
        d.text((X, y + 4), label, font=fonts["s"], fill=(60, 60, 60))
        lx = X + int(d.textlength(label, font=fonts["s"])) + random.randint(10, 24)
        d.text((lx, y), text, font=font, fill=fill)
        x0, y0, x1, y1 = d.textbbox((lx, y), text, font=font)
        boxes[cls] = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]

    idc = random.choice([(30, 30, 30), (200, 40, 50), (30, 30, 30)])
    put("id", rand_id(), 175, fonts["id"], idc, "Số / No.:")
    put("name", rand_name(), 250, fonts["name"], dark, "Họ và tên / Full name:")
    put("dob", rand_dob(), 320, fonts["dob"], dark, "Ngày sinh / Date of birth:")
    d.text((X, 370), "Giới tính / Sex:", font=fonts["s"], fill=(60, 60, 60))
    d.text((X, 410), "Quê quán / Place of origin:", font=fonts["s"], fill=(60, 60, 60))
    d.text((X, 435), random.choice(QUE), font=fonts["reg"], fill=dark)
    d.text((X, 475), "Nơi thường trú / Place of residence:", font=fonts["s"], fill=(60, 60, 60))
    return np.array(im)[:, :, ::-1].copy(), boxes


def make_background(w, h):
    kind = random.choice(["grad", "noise", "wood", "dark"])
    if kind == "grad":
        bg = grad(w, h, [random.randint(60, 255) for _ in range(3)], [random.randint(60, 255) for _ in range(3)])
    elif kind == "dark":
        bg = grad(w, h, [random.randint(10, 70) for _ in range(3)], [random.randint(10, 90) for _ in range(3)])
    elif kind == "wood":
        base = np.array([random.randint(90, 170), random.randint(110, 170), random.randint(150, 210)], np.float32)
        stripes = (np.sin(np.linspace(0, random.randint(30, 90), h))[:, None, None] * 18)
        bg = np.clip(base[None, None, :] + stripes + np.zeros((h, w, 3)), 0, 255).astype(np.uint8)
    else:
        bg = np.full((h, w, 3), random.randint(80, 220), np.uint8)
    n = np.random.normal(0, random.uniform(3, 14), bg.shape)
    return np.clip(bg.astype(np.float32) + n, 0, 255).astype(np.uint8)


def random_homography(max_rot, persp):
    """Ma trận 3x3 đưa thẻ phẳng vào khung ảnh: xoay + scale + dịch + perspective."""
    scale = random.uniform(0.55, 0.92)
    ang = math.radians(random.uniform(-max_rot, max_rot))
    cx, cy = CARD_W / 2, CARD_H / 2
    src = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    pts = []
    for x, y in src:
        dx, dy = (x - cx) * scale, (y - cy) * scale
        rx = dx * math.cos(ang) - dy * math.sin(ang)
        ry = dx * math.sin(ang) + dy * math.cos(ang)
        pts.append([rx, ry])
    pts = np.float32(pts)
    # perspective: xê dịch ngẫu nhiên từng góc
    pts += np.float32([[random.uniform(-persp, persp) * CARD_W * scale,
                        random.uniform(-persp, persp) * CARD_H * scale] for _ in range(4)])
    # đặt vào khung, đảm bảo nằm trọn trong ảnh
    mn, mx = pts.min(0), pts.max(0)
    size = mx - mn
    margin = 6
    if size[0] > OUT_W - 2 * margin or size[1] > OUT_H - 2 * margin:
        k = min((OUT_W - 2 * margin) / size[0], (OUT_H - 2 * margin) / size[1])
        pts *= k
        mn, mx = pts.min(0), pts.max(0)
        size = mx - mn
    ox = random.uniform(margin, OUT_W - margin - size[0])
    oy = random.uniform(margin, OUT_H - margin - size[1])
    dst = pts - mn + np.float32([ox, oy])
    return cv2.getPerspectiveTransform(src, dst)


def apply_h(H, pts):
    p = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def photo_aug(img):
    # mờ do rung / mất nét
    r = random.random()
    if r < 0.25:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.45:
        k = random.choice([5, 9, 13])
        kern = np.zeros((k, k), np.float32)
        ang = random.uniform(0, math.pi)
        c = k // 2
        cv2.line(kern, (int(c - c * math.cos(ang)), int(c - c * math.sin(ang))),
                 (int(c + c * math.cos(ang)), int(c + c * math.sin(ang))), 1, 1)
        img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1))
    # chói sáng
    if random.random() < 0.35:
        glare = np.zeros(img.shape[:2], np.float32)
        cv2.ellipse(glare, (random.randint(0, OUT_W), random.randint(0, OUT_H)),
                    (random.randint(60, 220), random.randint(30, 120)), random.randint(0, 180), 0, 360,
                    random.uniform(60, 140), -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 35)
        img = np.clip(img.astype(np.float32) + glare[:, :, None], 0, 255).astype(np.uint8)
    # độ sáng / tương phản / nhiễu
    a, b = random.uniform(0.7, 1.25), random.uniform(-30, 25)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255)
    img += np.random.normal(0, random.uniform(0, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.4:  # nén JPEG xấu
        q = random.randint(35, 80)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, 1)
    return img


def order_clockwise(q):
    """Sắp 4 điểm theo chiều kim đồng hồ bắt đầu từ góc trên-trái (ổn định cho OBB)."""
    q = np.asarray(q, np.float32)
    c = q.mean(0)
    idx = np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))
    q = q[idx]
    s = q.sum(1)
    start = int(np.argmin(s))
    return np.roll(q, -start, axis=0)


def make_sample(fonts, max_rot, persp):
    card, boxes = draw_card(fonts)
    H = random_homography(max_rot, persp)
    bg = make_background(OUT_W, OUT_H)
    warped = cv2.warpPerspective(card, H, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full(card.shape[:2], 255, np.uint8), H, (OUT_W, OUT_H))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[:, :, None] / 255.0
    img = (warped * mask + bg * (1 - mask)).astype(np.uint8)
    img = photo_aug(img)

    obb, hbb = [], []
    for ci, cls in enumerate(CLASSES):
        x0, y0, x1, y1 = boxes[cls]
        q = apply_h(H, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        q = order_clockwise(q)
        qn = q / np.array([OUT_W, OUT_H], np.float32)
        qn = np.clip(qn, 0, 1)
        obb.append(f"{ci} " + " ".join(f"{v:.6f}" for v in qn.reshape(-1)))
        mn, mx = qn.min(0), qn.max(0)
        hbb.append(f"{ci} {(mn[0]+mx[0])/2:.6f} {(mn[1]+mx[1])/2:.6f} {mx[0]-mn[0]:.6f} {mx[1]-mn[1]:.6f}")
    return img, obb, hbb


def draw_debug(img, obb):
    out = img.copy()
    colors = [(0, 0, 255), (0, 160, 0), (255, 0, 0)]
    for line in obb:
        p = line.split()
        c = int(p[0])
        pts = (np.float32(p[1:]).reshape(4, 2) * [OUT_W, OUT_H]).astype(np.int32)
        cv2.polylines(out, [pts], True, colors[c], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_obb")
    ap.add_argument("--n", type=int, default=1500, help="tổng số ảnh")
    ap.add_argument("--val", type=float, default=0.1, help="tỉ lệ ảnh val")
    ap.add_argument("--max-rot", type=float, default=35, help="góc xoay tối đa (độ)")
    ap.add_argument("--persp", type=float, default=0.06, help="độ méo perspective (0-0.15)")
    ap.add_argument("--debug", type=int, default=0, help="số ảnh debug vẽ nhãn")
    ap.add_argument("--font", default=None, help="font .ttf thường (hỗ trợ tiếng Việt)")
    ap.add_argument("--font-bold", default=None, help="font .ttf đậm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    reg = find_font(FONT_CANDIDATES_REG, a.font)
    bold = find_font(FONT_CANDIDATES_BOLD, a.font_bold)
    mono = find_font(FONT_MONO + FONT_CANDIDATES_BOLD, None)

    def mk_fonts():
        return {
            "s": ImageFont.truetype(reg, 17),
            "reg": ImageFont.truetype(reg, random.randint(24, 28)),
            "title": ImageFont.truetype(bold, random.randint(34, 40)),
            "id": ImageFont.truetype(random.choice([bold, mono]), random.randint(34, 42)),
            "name": ImageFont.truetype(bold, random.randint(26, 34)),
            "dob": ImageFont.truetype(bold, random.randint(24, 30)),
        }

    for sub in ["images", "labels", "labels_hbb"]:
        for sp in ["train", "val"]:
            os.makedirs(f"{a.out}/{sub}/{sp}", exist_ok=True)
    if a.debug:
        os.makedirs(f"{a.out}/debug", exist_ok=True)

    n_val = int(a.n * a.val)
    for i in range(a.n):
        sp = "val" if i < n_val else "train"
        name = f"syn_{i:05d}"
        img, obb, hbb = make_sample(mk_fonts(), a.max_rot, a.persp)
        cv2.imwrite(f"{a.out}/images/{sp}/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{a.out}/labels/{sp}/{name}.txt", "w").write("\n".join(obb) + "\n")
        open(f"{a.out}/labels_hbb/{sp}/{name}.txt", "w").write("\n".join(hbb) + "\n")
        if i < a.debug:
            cv2.imwrite(f"{a.out}/debug/{name}.jpg", draw_debug(img, obb))

    with open(f"{a.out}/data.yaml", "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    print(f"Xong: {a.n} ảnh ({a.n - n_val} train / {n_val} val) -> {a.out}")


if __name__ == "__main__":
    main()
