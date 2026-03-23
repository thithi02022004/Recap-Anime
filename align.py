#!/usr/bin/env python3
"""
Anime Recap Aligner
====================
Tìm các cảnh tương đồng giữa video recap đã edit và tập anime raw,
sau đó cắt video raw bằng FFmpeg.

Sử dụng:
  python align.py --edited recap.mp4 --raw episode.mp4
  python align.py --edited recap.mp4 --raw episode.mp4 --output D:/cuts --threshold 12
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Kiểu dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

class VideoFrame:
    __slots__ = ("time", "embedding")

    def __init__(self, time: float, embedding: np.ndarray):
        self.time = time
        self.embedding = embedding


class MatchSegment:
    __slots__ = ("edited_start", "edited_end", "raw_start", "raw_end", "frames")

    def __init__(
        self,
        edited_start: float,
        edited_end: float,
        raw_start: float,
        raw_end: float,
        frames: int,
    ):
        self.edited_start = edited_start
        self.edited_end = edited_end
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.frames = frames

    @property
    def raw_duration(self) -> float:
        return self.raw_end - self.raw_start


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Model
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(device: str):
    print("🧠 Đang tải mô hình CLIP (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device).eval()

    # Tự động bật fp16 khi chạy CUDA — Tensor Cores trên NVIDIA GPU tăng tốc 1.5–2.5×
    use_fp16 = (device == "cuda" and torch.cuda.is_available())
    if use_fp16:
        model = model.half()
        print(f"   ✅ CLIP sẵn sàng trên {device.upper()} (fp16 — Tensor Cores)")
    else:
        print(f"   ✅ CLIP sẵn sàng trên {device.upper()} (fp32)")

    return model, processor


def get_embeddings(
    images: List[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> np.ndarray:
    """Trả về ma trận embedding đã L2-normalize, shape (N, D)."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Cast pixel_values sang dtype của model để tương thích với fp16/fp32
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    with torch.no_grad():
        out = model.get_image_features(**inputs)
        # Handle depending on return type (backwards compatibility)
        if hasattr(out, "pooler_output"):
            feats = out.pooler_output
        elif hasattr(out, "image_embeds"):
            feats = out.image_embeds
        elif isinstance(out, dict) and "image_embeds" in out:
            feats = out["image_embeds"]
        elif isinstance(out, dict) and "pooler_output" in out:
            feats = out["pooler_output"]
        else:
            feats = out  # tensor

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()  # luôn trả về float32 để numpy không bị lỗi


# ─────────────────────────────────────────────────────────────────────────────
# Trích xuất frame
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    video_path: str,
    fps: int,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    label: str = "video",
    batch_size: int = 16,
) -> List[VideoFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / video_fps
    frame_step = max(1, round(video_fps / fps))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crop 20% viền để tránh logo / hardsub
    x0 = int(W * 0.20)
    y0 = int(H * 0.20)
    x1 = x0 + int(W * 0.60)
    y1 = y0 + int(H * 0.60)

    print(f"\n📹 Phân tích [{label}]: {Path(video_path).name}")
    print(f"   Thời lượng: {duration:.1f}s | {W}×{H} | lấy mẫu {fps}fps | ~{int(duration*fps)} frames")

    result: List[VideoFrame] = []
    buf_imgs: List[Image.Image] = []
    buf_times: List[float] = []
    frame_idx = 0

    pbar = tqdm(total=int(duration * fps), unit="frame", desc=f"  {label}", ncols=80)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            crop = frame[y0:y1, x0:x1]
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 224))
            buf_imgs.append(img)
            buf_times.append(frame_idx / video_fps)

            if len(buf_imgs) >= batch_size:
                embs = get_embeddings(buf_imgs, model, processor, device)
                for t, e in zip(buf_times, embs):
                    result.append(VideoFrame(t, e))
                pbar.update(len(buf_imgs))
                buf_imgs, buf_times = [], []

        frame_idx += 1

    if buf_imgs:
        embs = get_embeddings(buf_imgs, model, processor, device)
        for t, e in zip(buf_times, embs):
            result.append(VideoFrame(t, e))
        pbar.update(len(buf_imgs))

    pbar.close()
    cap.release()
    print(f"   ✅ {len(result)} embeddings")
    return result



# ─────────────────────────────────────────────────────────────────────────────
# So khớp cảnh
# ─────────────────────────────────────────────────────────────────────────────

def find_matches(
    edited: List[VideoFrame],
    raw: List[VideoFrame],
    threshold_pct: float = 15.0,
) -> List[MatchSegment]:
    sim_thr = 1.0 - threshold_pct / 100.0
    print(f"\n🔍 So khớp cảnh (ngưỡng cosine ≥ {sim_thr:.2f})...")

    # Ma trận embedding raw: (N_raw, D)
    raw_mat = np.stack([f.embedding for f in raw])  # (N, D)

    matches = []
    for ef in tqdm(edited, desc="  Matching", ncols=80):
        # Vectorized: cosine sim với toàn bộ raw (đã normalize → dot product)
        sims = raw_mat @ ef.embedding          # shape (N,)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        if best_sim >= sim_thr:
            rt = raw[best_i].time
            matches.append({
                "et": ef.time,
                "rt": rt,
                "offset": rt - ef.time,
                "sim": best_sim,
            })

    if not matches:
        return []

    # Gom thành chuỗi liên tục
    segments: List[MatchSegment] = []
    seq = [matches[0]]

    for m in matches[1:]:
        last = seq[-1]
        if (m["et"] - last["et"]) <= 2.5 and abs(m["offset"] - last["offset"]) <= 1.5:
            seq.append(m)
        else:
            if len(seq) >= 3:
                segments.append(_make_seg(seq))
            seq = [m]

    if len(seq) >= 3:
        segments.append(_make_seg(seq))

    print(f"   Tìm thấy {len(segments)} thô")
    
    # 4. Gộp các đoạn gần nhau trên edited timeline (để không bị chia cắt/lặp lại cùng 1 cảnh)
    segments = _merge_edited_segments(segments, max_edited_gap=4.0, max_raw_shift=2.0)

    print(f"   ✅ {len(segments)} đoạn cắt cuối cùng (đã gộp các đoạn gần nhau)")
    return segments



def _make_seg(seq: list) -> MatchSegment:
    return MatchSegment(
        edited_start=seq[0]["et"],
        edited_end=seq[-1]["et"],
        raw_start=seq[0]["rt"],
        raw_end=seq[-1]["rt"],
        frames=len(seq),
    )

def _merge_edited_segments(segments: List[MatchSegment], max_edited_gap: float = 4.0, max_raw_shift: float = 2.0) -> List[MatchSegment]:
    """Gộp các đoạn nếu chúng liên tục trên cả video recap và raw (bị AI rớt match ở giữa)."""
    if not segments:
        return []
        
    segments.sort(key=lambda s: s.edited_start)
    
    merged = [segments[0]]
    for curr in segments[1:]:
        prev = merged[-1]
        
        edited_gap = curr.edited_start - prev.edited_end
        raw_gap = curr.raw_start - prev.raw_end
        
        # Nếu khoảng trống giữa 2 cảnh <= 4 giây, VÀ độ lệch giữa raw/edited <= 2 giây (cùng 1 cảnh)
        if 0 <= edited_gap <= max_edited_gap and abs(edited_gap - raw_gap) <= max_raw_shift:
            # Gộp thành 1 phân đoạn liên tục duy nhất
            prev.edited_end = max(prev.edited_end, curr.edited_end)
            prev.raw_end = max(prev.raw_end, curr.raw_end)
            prev.frames += curr.frames
        else:
            merged.append(curr)
            
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg cắt video
# ─────────────────────────────────────────────────────────────────────────────

def fmt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def cut_video(
    raw_path: str,
    segments: List[MatchSegment],
    output_dir: str,
    buffer: float = 0.5,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    print(f"\n✂️  Cắt {len(segments)} đoạn → {output_dir}/")

    for i, seg in enumerate(segments, start=1):
        # Yêu cầu: Cut bằng chính xác độ dài của edit.
        target_dur = seg.edited_end - seg.edited_start
        out = os.path.join(output_dir, f"cut_{i:03d}.mp4")

        # Cắt chính xác target_dur tính từ điểm raw_start
        # QUAN TRỌNG: -ss phải đặt SAU -i để FFmpeg decode frame-accurate
        # (đặt trước -i → fast seek đến keyframe gần nhất → lệch vài giây)
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_path,
            "-ss", f"{seg.raw_start:.3f}",
            "-t", f"{target_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out,
        ]
        tqdm.write(f"  [{i:03d}] ✅ Cắt chuẩn {fmt(seg.raw_start)} len: {target_dur:.1f}s")

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            saved.append(out)
        else:
            tqdm.write(f"  [{i:03d}] ❌ Lỗi FFmpeg: {r.stderr[-200:]}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Tạo ảnh preview
# ─────────────────────────────────────────────────────────────────────────────

def generate_previews(
    edited_path: str,
    raw_path: str,
    segments: List[MatchSegment],
    output_dir: str
):
    preview_dir = os.path.join(output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    print(f"\n📸 Đang tạo ảnh preview cho {len(segments)} đoạn tại {preview_dir}/ ...")

    cap_e = cv2.VideoCapture(edited_path)
    cap_r = cv2.VideoCapture(raw_path)

    for i, seg in enumerate(segments, 1):
        cap_e.set(cv2.CAP_PROP_POS_MSEC, seg.edited_start * 1000)
        ret_e, frame_e = cap_e.read()
        
        cap_r.set(cv2.CAP_PROP_POS_MSEC, seg.raw_start * 1000)
        ret_r, frame_r = cap_r.read()
        
        if ret_e and ret_r:
            h, w = 360, 640
            frame_e = cv2.resize(frame_e, (w, h))
            frame_r = cv2.resize(frame_r, (w, h))
            
            combo = np.hstack((frame_e, frame_r))
            
            cv2.putText(combo, f"Edited: {fmt(seg.edited_start)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combo, f"Raw: {fmt(seg.raw_start)}", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out_path = os.path.abspath(os.path.join(preview_dir, f"preview_{i:03d}.jpg"))
            cv2.imwrite(out_path, combo)
            print(f"[PREVIEW] {out_path}")
            
    cap_e.release()
    cap_r.release()



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def detect_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Anime Recap Aligner — so khớp và cắt video dùng CLIP + FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python align.py --edited recap.mp4 --raw episode.mp4
  python align.py --edited recap.mp4 --raw D:/ep01.mp4 --output D:/cuts --threshold 12
  python align.py --edited recap.mp4 --raw episode.mp4 --no-cut   # chỉ xem kết quả
        """,
    )
    parser.add_argument("--edited",    required=True,  help="Đường dẫn video recap đã edit")
    parser.add_argument("--raw",       required=True,  help="Đường dẫn tập anime raw gốc")
    parser.add_argument("--output",    default=None,   help="Thư mục output (mặc định: cuts/ cạnh file raw)")
    parser.add_argument("--fps",       type=int,   default=3,    help="Số frame/giây để phân tích (default: 3)")
    parser.add_argument("--threshold", type=float, default=15.0, help="Ngưỡng khớp %% (default: 15, càng nhỏ càng chặt)")
    parser.add_argument("--buffer",    type=float, default=0.5,  help="Giây đệm đầu/cuối mỗi đoạn (default: 0.5)")
    parser.add_argument("--no-cut",    action="store_true",      help="Chỉ tìm khớp, không cắt video")
    parser.add_argument("--device",    default=None,             help="cuda / cpu / mps (tự detect nếu bỏ trống)")
    parser.add_argument("--batch",     type=int,   default=32,   help="Batch size khi chạy CLIP (default: 32)")
    args = parser.parse_args()

    # Kiểm tra file tồn tại
    for label, path in [("--edited", args.edited), ("--raw", args.raw)]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy file {label}: {path}")
            sys.exit(1)

    device = detect_device(args.device)
    output_dir = args.output or str(Path(args.raw).parent / "cuts")

    print("=" * 60)
    print("🎬 Anime Recap Aligner")
    print("=" * 60)
    print(f"   Edited  : {args.edited}")
    print(f"   Raw     : {args.raw}")
    print(f"   Output  : {output_dir}")
    print(f"   Device  : {device.upper()}")
    fp16_note = " + fp16 (Tensor Cores)" if device == "cuda" else ""
    print(f"   Model   : CLIP ViT-B/32{fp16_note}")
    print(f"   FPS     : {args.fps} | Ngưỡng: {args.threshold}% | Batch: {args.batch} | Buffer: {args.buffer}s")
    print("=" * 60)

    t0 = time.time()

    # 1. Load model
    model, processor = load_clip(device)

    # 2. Trích xuất embeddings
    edited_frames = extract_embeddings(args.edited, args.fps, model, processor, device, "Edited", args.batch)
    raw_frames    = extract_embeddings(args.raw,    args.fps, model, processor, device, "Raw",    args.batch)

    if len(edited_frames) == 0 or len(raw_frames) == 0:
        print("\n❌ LỖI: Không trích xuất được khung hình nào từ một trong 2 video.")
        print("Có thể file video bị hỏng (0 bytes) hoặc đuôi video (như .webm) không được hỗ trợ xử lý!")
        sys.exit(1)

    # 3. So khớp
    segments = find_matches(edited_frames, raw_frames, args.threshold)

    if not segments:
        print("\n❌ Không tìm thấy cảnh khớp. Thử tăng --threshold.")
        sys.exit(0)

    # In bảng kết quả
    print(f"\n{'─'*65}")
    print(f"{'#':<5}  {'Edited':<22}  {'Raw (mốc cắt)':<22}  {'Dài'}")
    print(f"{'─'*65}")
    for i, seg in enumerate(segments, 1):
        print(
            f"  {i:<3}  "
            f"{fmt(seg.edited_start)} → {fmt(seg.edited_end)}  "
            f"{fmt(seg.raw_start)} → {fmt(seg.raw_end)}  "
            f"{seg.raw_duration:.1f}s"
        )
    print(f"{'─'*65}")
    print(f"  Tổng: {len(segments)} đoạn")

    # Cắt
    if not args.no_cut:
        saved = cut_video(args.raw, segments, output_dir, args.buffer)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"✅ Hoàn tất! {len(saved)}/{len(segments)} đoạn đã lưu tại:")
        print(f"   📁 {os.path.abspath(output_dir)}")
        print(f"⏱️  Tổng thời gian: {elapsed:.0f}s")
        print(f"{'='*60}")
    else:
        print("\n(--no-cut: bỏ qua bước cắt video)")
        generate_previews(args.edited, args.raw, segments, output_dir)
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"✅ Hoàn tất! Đã lưu ảnh preview tại:")
        print(f"   📁 {os.path.abspath(os.path.join(output_dir, 'previews'))}")
        print(f"⏱️  Tổng thời gian: {elapsed:.0f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
