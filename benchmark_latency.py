"""
SimLingo 全流程 Latency Benchmark
===================================
不需要 CARLA，直接載入模型跑 dummy forward pass，
量測各階段推論時間。

使用方式:
    python benchmark_latency.py
    python benchmark_latency.py --gpu 4 --runs 20
    python benchmark_latency.py --gpu 4 --runs 20 --warmup 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── 路徑設定 ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
CARLA_ROOT = Path("/mnt/SSD7/dow904/carla")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla"))
sys.path.insert(0, str(ROOT / "leaderboard"))
sys.path.insert(0, str(ROOT / "scenario_runner"))

import torch
import numpy as np
from omegaconf import OmegaConf
import hydra
from transformers import AutoProcessor

# ── 引入專案型別 ────────────────────────────────────────────────
from simlingo_training.utils.custom_types import (
    DrivingInput, LanguageLabel,
    DrivingExample, DrivingLabel,
)


# ────────────────────────────────────────────────────────────────
def make_dummy_language_label(phrase_ids: torch.Tensor, device: torch.device) -> LanguageLabel:
    """建立 dummy LanguageLabel，使用真實 tokenizer 產生的 token ids"""
    seq_len = phrase_ids.shape[1]
    return LanguageLabel(
        phrase_ids=phrase_ids.to(device),
        phrase_valid=torch.ones(1, seq_len, dtype=torch.bool, device=device),
        phrase_mask=torch.zeros(1, seq_len, dtype=torch.bool, device=device),
        placeholder_values=[],   # 空 list → 跳過 waypoint placeholder 替換
        language_string=["dummy"],
        loss_masking=torch.zeros(1, seq_len, dtype=torch.bool, device=device),
    )


def make_dummy_input(model, device: torch.device) -> DrivingInput:
    """
    建立與真實推論相同 shape 的 dummy DrivingInput。
    使用 model 的 tokenizer 建立含正確 <IMG_CONTEXT> 數量的 prompt tokens。

    InternVL2-1B: image_size=448, patch_size=14, downsample_ratio=0.5
      → num_image_token = (448//14)^2 * 0.5^2 = 256
      → num_patches_all = 2 (thumbnail + full)
      → total IMG_CONTEXT = 512 tokens
    """
    B, T, num_patches, C, H, W = 1, 1, 2, 3, 448, 448

    camera_images = torch.randint(
        0, 256, (B, T, num_patches, C, H, W),
        dtype=torch.bfloat16, device=device
    )
    image_sizes = torch.tensor([[H, W]], dtype=torch.long, device=device)
    camera_intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).float()
    camera_extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).float()
    vehicle_speed = torch.tensor([[8.0]], dtype=torch.float32, device=device)
    target_point  = torch.tensor([[10.0, 0.0]], dtype=torch.float32, device=device)

    # ── 用真實 tokenizer 建立含 <IMG_CONTEXT> 的 prompt ──────────
    # img_context_token_id 在第一次 forward 才設定，直接從 processor 取
    proc = model.vision_model.image_encoder.processor
    tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # InternVL2-1B: num_image_token=256, num_patches=2 → 512 context tokens
    num_image_token = 256
    num_patches_all = 2
    img_context_ids = torch.full(
        (1, num_image_token * num_patches_all),
        fill_value=img_context_token_id,
        dtype=torch.int64,
    )
    # 前後加一些普通文字 token (用 bos_token_id 代替)
    pad_id = tokenizer.pad_token_id or 0
    prefix = torch.full((1, 10), pad_id, dtype=torch.int64)
    suffix = torch.full((1, 10), pad_id, dtype=torch.int64)
    phrase_ids = torch.cat([prefix, img_context_ids, suffix], dim=1)  # [1, 532]

    prompt           = make_dummy_language_label(phrase_ids, device)
    prompt_inference = make_dummy_language_label(phrase_ids, device)

    return DrivingInput(
        camera_images=camera_images,
        image_sizes=image_sizes,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        vehicle_speed=vehicle_speed,
        target_point=target_point,
        prompt=prompt,
        prompt_inference=prompt_inference,
    )


# ────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    """使用與 agent_simlingo 完全相同的方式載入模型"""
    hydra_cfg_path = ROOT / ".hydra" / "config.yaml"
    print(f"[cfg]  讀取 hydra config: {hydra_cfg_path}")
    cfg = OmegaConf.load(hydra_cfg_path)
    cfg.model.vision_model.use_global_img = cfg.data_module.get("use_global_img", False)

    print(f"[model] 載入 processor ({cfg.model.vision_model.variant})...")
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True
    )
    cache_dir = str(ROOT / "pretrained" / cfg.model.vision_model.variant.split("/")[1])

    print(f"[model] 建立模型 ({cfg.model._target_})...")
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False,
    ).to(device)
    torch.set_default_dtype(default_dtype)

    print(f"[model] 載入 checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[model] 載入完成，移至 {device}")
    return model, cfg


# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_benchmark(model, device: torch.device, num_runs: int, warmup: int):
    dummy_input = make_dummy_input(model, device)

    # 計時用容器
    t_total   = []
    t_vision  = []
    t_llm     = []
    t_decoder = []

    print(f"\n[bench] warmup {warmup} runs ...")
    for _ in range(warmup):
        model(dummy_input)
        torch.cuda.synchronize()

    print(f"[bench] 正式量測 {num_runs} runs ...\n")
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        model(dummy_input)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_ms = round((t1 - t0) * 1000, 1)
        t_total.append(total_ms)

        # 從 model 的 timing 屬性讀取各階段時間
        t_vision.append(getattr(model, "_timing_vision_ms", None))
        t_llm.append(getattr(model, "_timing_llm_ms", None))
        t_decoder.append(getattr(model, "_timing_decoder_ms", None))

        print(f"  run {i+1:3d}/{num_runs} │ total={total_ms:7.1f} ms"
              f" │ vision={t_vision[-1] or '?':>7}ms"
              f" │ llm={t_llm[-1] or '?':>7}ms"
              f" │ decoder={t_decoder[-1] or '?':>7}ms")

    return t_total, t_vision, t_llm, t_decoder


# ────────────────────────────────────────────────────────────────
def summarize(label: str, values: list):
    clean = [v for v in values if v is not None]
    if not clean:
        print(f"  {label:<20s}: N/A")
        return
    arr = np.array(clean)
    print(f"  {label:<20s}: "
          f"mean={arr.mean():7.1f} ms  "
          f"std={arr.std():6.1f} ms  "
          f"min={arr.min():7.1f} ms  "
          f"max={arr.max():7.1f} ms  "
          f"p50={np.percentile(arr,50):7.1f} ms  "
          f"p95={np.percentile(arr,95):7.1f} ms")


# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",      type=int, default=4,    help="CUDA device index")
    parser.add_argument("--runs",     type=int, default=20,   help="正式量測次數")
    parser.add_argument("--warmup",   type=int, default=3,    help="warmup 次數")
    parser.add_argument("--model",    type=str,
        default=str(ROOT / "outputs/simlingo/checkpoints/pytorch_model.pt"),
        help="模型 checkpoint 路徑")
    args = parser.parse_args()

    os.chdir(ROOT)  # hydra 需要在 repo root 執行

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[info] 使用裝置: {device}")
    print(f"[info] GPU: {torch.cuda.get_device_name(device)}")

    model, _ = load_model(args.model, device)

    t_total, t_vision, t_llm, t_decoder = run_benchmark(
        model, device, args.runs, args.warmup
    )

    # 計算「其他」時間（preprocessing / adaptor 等）
    t_other = []
    for tot, vis, llm, dec in zip(t_total, t_vision, t_llm, t_decoder):
        if all(v is not None for v in [vis, llm, dec]):
            t_other.append(tot - vis - llm - dec)

    print("\n" + "=" * 75)
    print(" SimLingo Latency 報告")
    print("=" * 75)
    summarize("Total",          t_total)
    summarize("Vision Encoder", t_vision)
    summarize("LLM",            t_llm)
    summarize("Waypoint Decoder", t_decoder)
    summarize("Other (adaptor)", t_other)
    print("=" * 75)
    print(f"  runs={args.runs}  warmup={args.warmup}  device={device}")
    print("=" * 75)


if __name__ == "__main__":
    main()
