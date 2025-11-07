"""
Video Memory Orchestrator (Offline) with Qwen-VL on vLLM

Usage examples:

  python examples/offline_inference/video_memory_orchestrator.py \
      --video /path/to/video.mp4 \
      --fps 2 \
      --period-sec 3 \
      --window-size 16 \
      --alpha 0.95 \
      --scene-thresh 0.25 \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --max-tokens 256

Notes:
- Requires a VLM that supports video inputs in vLLM (e.g., Qwen2.5-VL).
- If OpenCV/Decord are unavailable, the script will raise a clear error.
- The memory modules are lightweight, untrained, and CPU-friendly by default.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import decord  # type: ignore
except Exception:  # pragma: no cover
    decord = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Video IO
# ------------------------------


def _ensure_video_backend() -> str:
    if decord is not None:
        return "decord"
    if cv2 is not None:
        return "opencv"
    raise RuntimeError(
        "No video backend available. Please install decord or opencv-python."
    )


def _video_fps_with_opencv(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def _iter_frames_opencv(path: str, every_n: int) -> Iterator[Tuple[np.ndarray, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    index = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if index % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb, index
            index += 1
    finally:
        cap.release()


def _iter_frames_decord(path: str, sample_stride: int) -> Iterator[Tuple[np.ndarray, int]]:
    vr = decord.VideoReader(path)
    for idx in range(0, len(vr), sample_stride):
        frame_nd = vr[idx]  # decord NDArray (HWC RGB uint8)
        frame = frame_nd.asnumpy()
        yield frame, idx


class VideoReader:
    def __init__(self, path: str, target_fps: float | None):
        self.path = path
        self.backend = _ensure_video_backend()
        if self.backend == "opencv":
            base_fps = _video_fps_with_opencv(path)
        else:
            # Decord VideoReader exposes len() but not fps reliably across codecs
            # Use a safe default; stride will still be honored by target_fps
            base_fps = 30.0
        self.base_fps = base_fps
        self.target_fps = target_fps or base_fps
        self.sample_stride = max(1, int(round(self.base_fps / self.target_fps)))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        if self.backend == "decord":
            yield from _iter_frames_decord(self.path, self.sample_stride)
        else:
            yield from _iter_frames_opencv(self.path, self.sample_stride)


# ------------------------------
# Frame Encoder (CLIP or fallback)
# ------------------------------


class FrameEncoder(nn.Module):
    def __init__(self, device: torch.device, embed_dim: int = 768):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        self._use_open_clip = False
        self._normalize: Optional[nn.Module] = None

        # Try open_clip first
        try:  # pragma: no cover
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.backbone = model.visual.eval().to(device)
            # Build a minimal preprocessor compatible with numpy HWC RGB
            self._preprocess = preprocess
            self.embed_dim = model.visual.output_dim
            self._use_open_clip = True
        except Exception:
            # Fallback: light conv encoder to keep dependencies minimal
            conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.backbone = conv.to(device)
            self._proj = nn.Linear(128, embed_dim, bias=False).to(device)
            self._normalize = nn.LayerNorm(embed_dim).to(device)

    @torch.inference_mode()
    def forward(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        if len(frames) == 0:
            return torch.empty(0, self.embed_dim, device=self.device)

        if self._use_open_clip:  # pragma: no cover
            # Use open_clip preprocess pipeline per frame
            imgs = [self._preprocess(img).unsqueeze(0) for img in frames]
            batch = torch.cat(imgs, dim=0).to(self.device)
            feats = self.backbone(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = F.normalize(feats, dim=-1)
            return feats

        # Fallback: simple conv encoder
        tensors: List[torch.Tensor] = []
        for img in frames:
            # img: HWC RGB uint8
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0).to(self.device)
        feats = self.backbone(batch).flatten(1)
        feats = self._proj(feats)
        if self._normalize is not None:
            feats = self._normalize(feats)
        feats = F.normalize(feats, dim=-1)
        return feats


# ------------------------------
# Memory Modules
# ------------------------------


class ShortWindowAttention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False)

    @torch.inference_mode()
    def forward(self, window_embeds: torch.Tensor) -> torch.Tensor:
        # window_embeds: [W, D]
        if window_embeds.numel() == 0:
            return torch.zeros((window_embeds.shape[-1],), device=window_embeds.device)
        scores = self.w2(torch.tanh(self.w1(window_embeds)))  # [W, 1]
        weights = torch.softmax(scores.squeeze(-1), dim=0)  # [W]
        summary = (weights.unsqueeze(-1) * window_embeds).sum(dim=0)  # [D]
        return summary


class LongTermSSM(nn.Module):
    def __init__(self, embed_dim: int, state_dim: int = 768, alpha: float = 0.95):
        super().__init__()
        self.alpha = float(alpha)
        self.proj = nn.Linear(embed_dim, state_dim, bias=False)
        self.state = None  # type: Optional[torch.Tensor]

    @torch.inference_mode()
    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def update(self, e_t: torch.Tensor, gate: float = 1.0) -> torch.Tensor:
        # e_t: [D]
        x = self.proj(e_t)
        if self.state is None:
            self.state = x
        else:
            a = self.alpha
            g = float(gate)
            self.state = a * self.state + (1.0 - a) * (g * x)
        return self.state


# ------------------------------
# Triggers
# ------------------------------


@dataclass
class TriggerConfig:
    period_frames: int
    scene_thresh: float
    cooldown_frames: int


class TriggerManager:
    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self._last_fire_idx: Optional[int] = None
        self._last_scene_idx: Optional[int] = None

    def periodic(self, frame_idx: int) -> bool:
        if self.cfg.period_frames <= 0:
            return False
        if self._last_fire_idx is None:
            self._last_fire_idx = frame_idx
            return True
        if frame_idx - self._last_fire_idx >= self.cfg.period_frames:
            self._last_fire_idx = frame_idx
            return True
        return False

    def scene_change(self, cos_distance: float, frame_idx: int) -> bool:
        if cos_distance < self.cfg.scene_thresh:
            return False
        if self._last_scene_idx is not None and (
            frame_idx - self._last_scene_idx < self.cfg.cooldown_frames
        ):
            return False
        self._last_scene_idx = frame_idx
        return True


# ------------------------------
# Snapshot textualization
# ------------------------------


def snapshot_to_text(
    long_state: torch.Tensor,
    short_state: torch.Tensor,
    tags: Sequence[str] | None,
    max_len: int = 32,
) -> str:
    def _fmt(vec: torch.Tensor) -> str:
        vec = vec.detach().float().cpu()
        if vec.numel() > max_len:
            vec = vec[:max_len]
        return ", ".join(f"{x:.3f}" for x in vec.tolist())

    tag_text = ", ".join(tags) if tags else ""
    return (
        "System memory snapshot.\n"
        f"LongMemory[head]: [{_fmt(long_state)}]\n"
        f"ShortMemory[head]: [{_fmt(short_state)}]\n"
        f"Tags: {tag_text}\n"
    )


# Optional: Zero-shot tag extraction via CLIP text encoder
class ZeroShotTextualizer:
    def __init__(self, labels: Sequence[str], device: torch.device):
        self.labels = list(labels)
        self.device = device
        self.ok = False
        try:  # pragma: no cover
            import open_clip  # type: ignore

            self._open_clip = open_clip
            self.model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.model = self.model.eval().to(device)
            with torch.inference_mode():
                tokenized = open_clip.tokenize(self.labels).to(device)
                self.text_feats = self.model.encode_text(tokenized)  # [L, D]
                self.text_feats = F.normalize(self.text_feats, dim=-1)
            self.ok = True
        except Exception:
            self.ok = False

    @torch.inference_mode()
    def propose_tags(self, visual_feat: torch.Tensor, top_k: int = 3) -> List[str]:
        if not self.ok or visual_feat.numel() == 0:
            return []
        v = visual_feat.detach()
        if v.ndim == 1:
            v = v.unsqueeze(0)
        v = F.normalize(v, dim=-1)
        sims = (v @ self.text_feats.T).mean(dim=0)  # [L]
        topk = torch.topk(sims, k=min(top_k, sims.numel()))
        return [self.labels[i] for i in topk.indices.tolist()]


# ------------------------------
# vLLM Client
# ------------------------------


class VLLMClient:
    def __init__(self, model: str, fps: float = 1.0, limit_mm_per_prompt: Optional[dict] = None):
        from vllm import LLM  # import lazily to keep CLI snappy

        self.llm = LLM(
            model=model,
            enforce_eager=True,
            limit_mm_per_prompt=limit_mm_per_prompt or {"video": 1},
            max_num_batched_tokens=128000,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": fps,
            },
        )

    def infer(
        self,
        prompt: str,
        video_frames: Sequence[np.ndarray],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Tuple[str, float]:
        from vllm import SamplingParams

        sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        # Stack frames into single video array: (num_frames, height, width, channels)
        video_array = np.stack(video_frames, axis=0) if video_frames else np.empty((0, 224, 224, 3), dtype=np.uint8)
        inputs = {"prompt": prompt, "multi_modal_data": {"video": video_array}}

        t0 = time.perf_counter()
        outputs = self.llm.generate([inputs], sampling_params=sampling)
        dt = time.perf_counter() - t0

        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return text, dt


# ------------------------------
# Orchestrator
# ------------------------------


def run_orchestrator(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Video reader
    reader = VideoReader(args.video, target_fps=args.fps)

    # Modules
    encoder = FrameEncoder(device=device, embed_dim=args.embed_dim).eval()
    short_mem = ShortWindowAttention(embed_dim=encoder.embed_dim, hidden_dim=256).to(device).eval()
    long_mem = LongTermSSM(embed_dim=encoder.embed_dim, state_dim=args.state_dim, alpha=args.alpha).to(device).eval()

    trigger_cfg = TriggerConfig(
        period_frames=int(round(reader.base_fps * args.period_sec)) if args.period_sec > 0 else 0,
        scene_thresh=float(args.scene_thresh),
        cooldown_frames=int(round(reader.base_fps * args.scene_cooldown_sec)),
    )
    triggers = TriggerManager(trigger_cfg)

    client = VLLMClient(model=args.model, fps=args.fps or 1.0)
    textualizer: Optional[ZeroShotTextualizer] = None
    if args.labels:
        try:
            with open(args.labels, "r", encoding="utf-8") as f:
                labels = [ln.strip() for ln in f.readlines() if ln.strip()]
            textualizer = ZeroShotTextualizer(labels=labels, device=device)
        except Exception:
            textualizer = None

    # Windows
    from collections import deque

    frame_window: Deque[np.ndarray] = deque(maxlen=args.window_size)
    embed_window: Deque[torch.Tensor] = deque(maxlen=args.window_size)
    prev_embed: Optional[torch.Tensor] = None

    # Rolling textual summary updated after each inference
    rolling_summary: List[str] = []

    for frame_rgb, idx in reader:
        frame_window.append(frame_rgb)

        # Encode current frame
        feats = encoder([frame_rgb])  # [1, D]
        e_t = feats[0]
        embed_window.append(e_t)

        # Scene score
        cos_dist = 0.0
        if prev_embed is not None and torch.isfinite(prev_embed).all():
            cos = F.cosine_similarity(prev_embed.unsqueeze(0), e_t.unsqueeze(0)).item()
            cos_dist = float(max(0.0, 1.0 - cos))
        prev_embed = e_t

        fired = triggers.periodic(idx) or triggers.scene_change(cos_dist, idx)
        if not fired:
            continue

        # Build memory snapshot
        window_mat = torch.stack(list(embed_window), dim=0) if len(embed_window) > 0 else torch.empty(0, encoder.embed_dim, device=device)
        short_vec = short_mem(window_mat)
        long_vec = long_mem.update(e_t, gate=1.0)

        # Textualize
        tags: List[str] = []
        if textualizer is not None and textualizer.ok:
            tags = textualizer.propose_tags(e_t, top_k=args.tag_top_k)
        snapshot_text = snapshot_to_text(long_vec, short_vec, tags)

        # Prompt assembly - Qwen2.5-VL format
        summary_prefix = ("\n".join(rolling_summary[-3:]) + "\n") if rolling_summary else ""
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
            f"{summary_prefix}{snapshot_text}\n"
            "Describe what just happened in the recent video segment concisely.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Copy out a slice of recent frames as the video segment
        segment_frames = list(frame_window)

        text, latency = client.infer(
            prompt=prompt,
            video_frames=segment_frames,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print("----- MEMORY SNAPSHOT RESPONSE -----")
        print(text.strip())
        print(f"(latency: {latency:.3f}s, frame_idx: {idx})")

        # Feedback: very light heuristic - keep a rolling text context
        if text:
            rolling_summary.append(text.strip())
            if len(rolling_summary) > 10:
                rolling_summary = rolling_summary[-10:]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video Memory Orchestrator with vLLM Qwen-VL")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--fps", type=float, default=None, help="Target sampling FPS (default: native fps)")
    p.add_argument("--window-size", type=int, default=16, help="Short memory window size in frames")
    p.add_argument("--alpha", type=float, default=0.95, help="Long memory decay (0..1), higher = longer memory")
    p.add_argument("--period-sec", type=float, default=3.0, help="Periodic trigger seconds (<=0 to disable)")
    p.add_argument("--scene-thresh", type=float, default=0.25, help="Scene-change cosine distance threshold [0..2]")
    p.add_argument("--scene-cooldown-sec", type=float, default=1.0, help="Cooldown after scene change (seconds)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="VLM model id/path for vLLM")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for responses")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--embed-dim", dest="embed_dim", type=int, default=768, help="Fallback encoder output dim")
    p.add_argument("--state-dim", type=int, default=768, help="Long memory state dimension")
    p.add_argument("--labels", type=str, default=None, help="Path to newline-separated labels for zero-shot tags")
    p.add_argument("--tag-top-k", type=int, default=3, help="Top-K tags to include in snapshot text")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_orchestrator(args)


if __name__ == "__main__":  # pragma: no cover
    main()


