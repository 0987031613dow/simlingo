#!/usr/bin/env python3
"""
SimLingo 評估影片生成工具

將評估輸出的影格（images/ 目錄）轉換為 MP4 影片。
需要在評估時設定 SAVE_VIDEO=1 才會有影格資料。

使用方式:
    python generate_eval_videos.py --input_dir eval_results/batch_xxx/viz --output_dir eval_results/batch_xxx/videos

    # 指定 FPS
    python generate_eval_videos.py --input_dir viz/ --output_dir videos/ --fps 10

    # 只處理特定路線
    python generate_eval_videos.py --input_dir viz/ --output_dir videos/ --filter bench2drive_001
"""

import os
import sys
import glob
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_image_dirs(base_dir, filter_str=None):
    """遞迴搜尋所有含影格的 images/ 目錄"""
    pattern = os.path.join(base_dir, "**/images")
    dirs = glob.glob(pattern, recursive=True)
    dirs = sorted([d for d in dirs if os.path.isdir(d)])

    if filter_str:
        dirs = [d for d in dirs if filter_str in d]

    return dirs


def find_metric_file(images_dir):
    """尋找對應的 metric_info.json"""
    # images/ 的上層是 debug_viz 下的時間戳目錄
    parent = Path(images_dir).parent
    metric_file = parent / "metric" / "metric_info.json"
    if metric_file.exists():
        return str(metric_file)
    return None


def load_metrics(metric_file):
    """載入 metric_info.json，回傳 step -> metric dict"""
    if metric_file is None:
        return {}
    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        return {}


def overlay_text(frame, text, position, font_scale=0.6, color=(255, 255, 255),
                 thickness=1, bg_color=(0, 0, 0)):
    """在影格上疊加文字（含黑色背景）"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # 繪製背景矩形
    cv2.rectangle(frame,
                  (x - 2, y - text_h - baseline - 2),
                  (x + text_w + 2, y + baseline + 2),
                  bg_color, -1)
    # 繪製文字
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def create_video(images_dir, output_path, fps=10, metrics=None):
    """
    從 images/ 目錄建立 MP4 影片。

    Args:
        images_dir:  含 PNG 影格的目錄（step.png 命名）
        output_path: 輸出 MP4 路徑
        fps:         影片每秒幀數
        metrics:     metric_info.json 的內容（可選）

    Returns:
        True 若成功，False 若失敗
    """
    # 取得影格列表（依步驟數值排序）
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.png")),
        key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else 0
    )

    if not image_files:
        print(f"  警告: 在 {images_dir} 找不到 PNG 影格")
        return False

    # 讀取第一張取得尺寸
    first = cv2.imread(image_files[0])
    if first is None:
        print(f"  錯誤: 無法讀取 {image_files[0]}")
        return False

    h, w = first.shape[:2]

    # 建立輸出目錄
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        # 嘗試 avc1 codec
        writer = cv2.VideoWriter(output_path.replace('.mp4', '_h264.mp4'),
                                 cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    print(f"  影格數: {len(image_files)}, 解析度: {w}x{h}")

    for img_path in tqdm(image_files, desc="  寫入影片", unit="frame", leave=False):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # 疊加 metric 資訊
        step_str = Path(img_path).stem
        if metrics and step_str in metrics:
            m = metrics[step_str]
            y_offset = 30
            info_lines = []
            if 'speed' in m:
                info_lines.append(f"speed: {float(m['speed']):.1f} m/s")
            if 'steer' in m:
                info_lines.append(f"steer: {float(m['steer']):.3f}")
            if 'throttle' in m:
                info_lines.append(f"throttle: {float(m['throttle']):.3f}")
            if 'brake' in m:
                info_lines.append(f"brake: {float(m['brake']):.3f}")
            info_lines.append(f"step: {step_str}")

            for line in info_lines:
                overlay_text(frame, line, (10, y_offset))
                y_offset += 25

        writer.write(frame)

    writer.release()
    return True


def path_to_video_name(images_dir, base_dir):
    """將 images/ 路徑轉換為有意義的影片檔名"""
    rel = os.path.relpath(images_dir, base_dir)
    # 移除 /images 後綴
    parts = Path(rel).parts
    # 去除最後的 "images" 部分
    parts = [p for p in parts if p != 'images']
    name = "__".join(parts)
    # 清理非法字元
    name = name.replace('/', '__').replace('\\', '__')
    return name + ".mp4"


def main():
    parser = argparse.ArgumentParser(
        description="從 SimLingo 評估輸出生成 MP4 影片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input_dir', required=True,
                        help='評估輸出的 viz/ 目錄（含各路線的子目錄）')
    parser.add_argument('--output_dir', required=True,
                        help='影片輸出目錄')
    parser.add_argument('--fps', type=int, default=10,
                        help='影片 FPS（預設: 10）')
    parser.add_argument('--filter', type=str, default=None,
                        help='只處理路徑含此字串的路線（e.g. bench2drive_001）')
    parser.add_argument('--no-metrics', action='store_true',
                        help='不在影片上疊加 metric 資訊')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: 找不到輸入目錄: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 搜尋所有 images/ 目錄
    print(f"搜尋影格目錄: {args.input_dir}")
    image_dirs = find_image_dirs(args.input_dir, filter_str=args.filter)

    if not image_dirs:
        print(f"找不到任何 images/ 目錄！")
        print(f"請確認評估時有設定 SAVE_VIDEO=1")
        sys.exit(1)

    print(f"找到 {len(image_dirs)} 個影格目錄")
    print()

    success = 0
    failed = 0

    for images_dir in image_dirs:
        video_name = path_to_video_name(images_dir, args.input_dir)
        output_path = os.path.join(args.output_dir, video_name)

        print(f"[{success + failed + 1}/{len(image_dirs)}] 處理: {os.path.relpath(images_dir, args.input_dir)}")
        print(f"  → {output_path}")

        # 跳過已存在的影片
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 1024:  # 大於 1KB 才視為有效
                print(f"  已存在，跳過")
                success += 1
                continue

        # 載入 metrics
        metrics = {}
        if not args.no_metrics:
            metric_file = find_metric_file(images_dir)
            if metric_file:
                metrics = load_metrics(metric_file)
                print(f"  metric: {metric_file}")

        result = create_video(images_dir, output_path, fps=args.fps, metrics=metrics)

        if result:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  完成: {output_path} ({size_mb:.1f} MB)")
            success += 1
        else:
            print(f"  失敗!")
            failed += 1

        print()

    print("=" * 60)
    print(f"完成! 成功: {success}, 失敗: {failed}")
    print(f"影片目錄: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
