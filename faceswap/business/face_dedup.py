"""
face_dedup.py - src 数据集去冗余 (3DDFA)

用 3DDFA ResNet-50 ONNX 推理, 输出 257 维 3DMM 参数:
  id(80) + exp(64) + alb(80) + angle(3) + sh(27) + trans(3)

去重逻辑:
  1. 3DDFA 推理 → angle + exp + sh
  2. 角度网格化 → 同角度分组
  3. 网格内按 exp + sh 聚类 → 同状态
  4. 每簇保留 keep_per_cluster 张 → 其余移到 backup (含 json 元数据)
  5. 覆盖度分析 + 建议
"""

import sys
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import cv2
import onnxruntime as ort

V3_ONNX = Path(__file__).parent.parent / "plugin" / "3DDFA-V3" / "assets" / "net_recon_resnet50.onnx"

INPUT_SIZE = 224
ALPHA_DIM = 257
ID_SLICE = slice(0, 80)
EXP_SLICE = slice(80, 144)
ALB_SLICE = slice(144, 224)
ANGLE_SLICE = slice(224, 227)
SH_SLICE = slice(227, 254)
TRANS_SLICE = slice(254, 257)


def load_onnx_session(onnx_path: str, use_gpu: bool = True) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
        else ["CPUExecutionProvider"]
    )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_opts, providers=providers)


def preprocess_image(img_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img


def batch_inference(session: ort.InferenceSession, images: list, batch_size: int = 32):
    input_name = session.get_inputs()[0].name
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        blob = np.stack(batch, axis=0).astype(np.float32)
        out = session.run(None, {input_name: blob})[0]
        results.append(out)
    return np.concatenate(results, axis=0)


def split_alpha(alpha: np.ndarray) -> dict:
    a = alpha
    return {
        "id": a[ID_SLICE],
        "exp": a[EXP_SLICE],
        "alb": a[ALB_SLICE],
        "angle": a[ANGLE_SLICE],
        "sh": a[SH_SLICE],
        "trans": a[TRANS_SLICE],
    }


def angle_to_degrees(angle_rad: np.ndarray) -> tuple:
    pitch, yaw, roll = np.degrees(angle_rad)
    return float(pitch), float(yaw), float(roll)


def compute_sharpness(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _move_with_metadata(src_img: Path, dst_img: Path):
    """移动图片及其 JSON 元数据 sidecar."""
    shutil.move(str(src_img), str(dst_img))
    src_json = src_img.with_suffix(".json")
    if src_json.exists():
        dst_json = dst_img.with_suffix(".json")
        if dst_json.exists():
            dst_json = dst_img.parent / f"dup_{dst_img.stem}.json"
        shutil.move(str(src_json), str(dst_json))


def dedup_states(
    states: list,
    yaw_grid: float = 5.0,
    pitch_grid: float = 5.0,
    roll_grid: float = 5.0,
    exp_thresh: float = 1.0,
    sh_thresh: float = 0.3,
    keep_per_cluster: int = 2,
    protect_yaw: float = 45.0,
):
    bins = defaultdict(list)
    protected = []

    for s in states:
        if abs(s["yaw"]) > protect_yaw:
            protected.append(s)
            continue
        yb = int(round(s["yaw"] / yaw_grid))
        pb = int(round(s["pitch"] / pitch_grid))
        rb = int(round(s["roll"] / roll_grid))
        bins[(yb, pb, rb)].append(s)

    keep = list(protected)
    remove = []

    for key, group in bins.items():
        if len(group) == 1:
            keep.append(group[0])
            continue

        clusters = []
        for s in group:
            placed = False
            for cluster in clusters:
                ref = cluster[0]
                exp_dist = float(np.linalg.norm(s["exp"] - ref["exp"]))
                sh_dist = float(np.linalg.norm(s["sh"] - ref["sh"]))
                if exp_dist < exp_thresh and sh_dist < sh_thresh:
                    cluster.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        for cluster in clusters:
            cluster.sort(key=lambda x: -x["sharpness"])
            keep.extend(cluster[:keep_per_cluster])
            remove.extend(cluster[keep_per_cluster:])

    return keep, remove


def analyze_coverage(keep: list, yaw_grid: float, pitch_grid: float) -> dict:
    yaw_range = (-75, 75)
    pitch_range = (-75, 75)

    covered = set()
    yaw_hist = defaultdict(int)
    pitch_hist = defaultdict(int)

    for s in keep:
        yb = int(round(s["yaw"] / yaw_grid))
        pb = int(round(s["pitch"] / pitch_grid))
        covered.add((yb, pb))
        yaw_hist[yb] += 1
        pitch_hist[pb] += 1

    missing = []
    y_bins = range(int(yaw_range[0] / yaw_grid), int(yaw_range[1] / yaw_grid) + 1)
    p_bins = range(int(pitch_range[0] / pitch_grid), int(pitch_range[1] / pitch_grid) + 1)
    for yb in y_bins:
        for pb in p_bins:
            if (yb, pb) not in covered:
                missing.append((yb * yaw_grid, pb * pitch_grid))

    exp_arr = np.array([s["exp"] for s in keep]) if keep else np.zeros((0, 64))
    sh_arr = np.array([s["sh"] for s in keep]) if keep else np.zeros((0, 27))

    return {
        "covered": covered,
        "missing": missing,
        "yaw_hist": dict(yaw_hist),
        "pitch_hist": dict(pitch_hist),
        "exp_mean": exp_arr.mean(axis=0) if len(exp_arr) > 0 else np.zeros(64),
        "exp_std": exp_arr.std(axis=0) if len(exp_arr) > 0 else np.zeros(64),
        "sh_mean": sh_arr.mean(axis=0) if len(sh_arr) > 0 else np.zeros(27),
        "sh_std": sh_arr.std(axis=0) if len(sh_arr) > 0 else np.zeros(27),
    }


def generate_report(
    keep: list,
    remove: list,
    protected: list,
    total: int,
    coverage: dict,
    report_path: Path,
    yaw_grid: float,
    pitch_grid: float,
):
    lines = []
    lines.append("=" * 70)
    lines.append("src 去冗余报告 (3DDFA)")
    lines.append("=" * 70)
    lines.append(f"总图片数:    {total}")
    lines.append(f"保留:        {len(keep)} ({len(keep)/total*100:.1f}%)")
    lines.append(f"移除:        {len(remove)} ({len(remove)/total*100:.1f}%)")
    lines.append(f"大角度保护:  {len(protected)} (|yaw|>45°)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("角度覆盖 (yaw)")
    lines.append("-" * 70)
    yh = coverage["yaw_hist"]
    for yb in sorted(yh.keys()):
        yr = f"{yb*yaw_grid:.0f}~{(yb+1)*yaw_grid:.0f}"
        bar = "#" * min(yh[yb], 50)
        lines.append(f"  yaw {yr:>8s}°: {yh[yb]:>4d} {bar}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("角度覆盖 (pitch)")
    lines.append("-" * 70)
    ph = coverage["pitch_hist"]
    for pb in sorted(ph.keys()):
        pr = f"{pb*pitch_grid:.0f}~{(pb+1)*pitch_grid:.0f}"
        bar = "#" * min(ph[pb], 50)
        lines.append(f"  pitch {pr:>8s}°: {ph[pb]:>4d} {bar}")
    lines.append("")

    missing = coverage["missing"]
    lines.append("-" * 70)
    lines.append(f"角度覆盖缺口 ({len(missing)} 个网格缺失)")
    lines.append("-" * 70)
    if missing:
        for yaw_c, pitch_c in missing[:40]:
            lines.append(f"  yaw≈{yaw_c:>5.0f}°  pitch≈{pitch_c:>5.0f}°")
        if len(missing) > 40:
            lines.append(f"  ... 还有 {len(missing) - 40} 个")
        lines.append("")
        lines.append("建议: 用 LivePortrait 自驱动生成缺失角度的补充帧")
    else:
        lines.append("  角度覆盖完整")
    lines.append("")

    lines.append("-" * 70)
    lines.append("表情覆盖 (64维 blendshape)")
    lines.append("-" * 70)
    exp_mean = coverage["exp_mean"]
    exp_std = coverage["exp_std"]
    lines.append(f"  均值范围: [{exp_mean.min():.3f}, {exp_mean.max():.3f}]")
    lines.append(f"  标准差范围: [{exp_std.min():.3f}, {exp_std.max():.3f}]")
    lines.append(f"  活跃维度(std>0.5): {int((exp_std > 0.5).sum())}/64")
    lines.append("")

    lines.append("-" * 70)
    lines.append("光照覆盖 (27维 SH)")
    lines.append("-" * 70)
    sh_mean = coverage["sh_mean"]
    sh_std = coverage["sh_std"]
    lines.append(f"  均值范围: [{sh_mean.min():.3f}, {sh_mean.max():.3f}]")
    lines.append(f"  标准差范围: [{sh_std.min():.3f}, {sh_std.max():.3f}]")
    lines.append(f"  活跃维度(std>0.1): {int((sh_std > 0.1).sum())}/27")
    lines.append("")

    if remove:
        lines.append("-" * 70)
        lines.append(f"移除的冗余帧 ({len(remove)} 张)")
        lines.append("-" * 70)
        for s in remove[:100]:
            lines.append(
                f"  {s['path'].name}  "
                f"yaw={s['yaw']:>6.1f} pitch={s['pitch']:>6.1f} "
                f"roll={s['roll']:>6.1f}"
            )
        if len(remove) > 100:
            lines.append(f"  ... 还有 {len(remove) - 100} 张")

    report_text = "\n".join(lines)
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


def analyze_states(
    input_dir,
    onnx_path=None,
    batch_size=32,
    use_cpu=False,
    progress_callback=None,
):
    """
    推理完整 3DMM 状态, 返回 [(path, yaw, pitch, roll, exp, sh), ...].
    exp: 64维表情向量, sh: 27维光照SH系数。
    用于 GUI 角度/表情/光照可视化分析。
    """
    input_dir = Path(input_dir)
    if onnx_path is None:
        onnx_path = str(V3_ONNX)

    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_files.extend(input_dir.rglob(ext))
    img_files = sorted(set(img_files))

    if not img_files:
        return []

    total = len(img_files)
    if progress_callback:
        progress_callback(0, total, "加载 ONNX 模型...")

    session = load_onnx_session(onnx_path, use_gpu=not use_cpu)

    results = []
    for i in range(0, total, batch_size):
        batch_paths = img_files[i:i + batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            img = preprocess_image(p)
            if img is not None:
                batch_imgs.append(img)
                valid_paths.append(p)

        if not batch_imgs:
            continue

        alphas = batch_inference(session, batch_imgs, batch_size=len(batch_imgs))

        for j, alpha in enumerate(alphas):
            parts = split_alpha(alpha)
            pitch, yaw, roll = angle_to_degrees(parts["angle"])
            results.append((valid_paths[j], yaw, pitch, roll, parts["exp"], parts["sh"]))

        done = min(i + batch_size, total)
        if progress_callback:
            progress_callback(done, total, f"推理 {done}/{total}")

    return results


def run_dedup(
    input_dir,
    onnx_path=None,
    backup_dir=None,
    dry_run=False,
    batch_size=32,
    keep_per_cluster=2,
    yaw_grid=5.0,
    pitch_grid=5.0,
    roll_grid=5.0,
    exp_thresh=1.0,
    sh_thresh=0.3,
    protect_yaw=45.0,
    use_cpu=False,
    progress_callback=None,
):
    """
    运行去重, 返回 (keep, remove, report_text).

    progress_callback(current, total, message) 可选, 用于 GUI 进度.
    """
    input_dir = Path(input_dir)
    if onnx_path is None:
        onnx_path = str(V3_ONNX)

    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_files.extend(input_dir.rglob(ext))
    img_files = sorted(set(img_files))

    if not img_files:
        return [], [], "未找到图片"

    total = len(img_files)
    if progress_callback:
        progress_callback(0, total, f"加载 ONNX 模型...")

    session = load_onnx_session(onnx_path, use_gpu=not use_cpu)

    all_states = []
    failed = 0

    for i in range(0, total, batch_size):
        batch_paths = img_files[i:i + batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            img = preprocess_image(p)
            if img is not None:
                batch_imgs.append(img)
                valid_paths.append(p)
            else:
                failed += 1

        if not batch_imgs:
            continue

        alphas = batch_inference(session, batch_imgs, batch_size=len(batch_imgs))

        for j, alpha in enumerate(alphas):
            parts = split_alpha(alpha)
            pitch, yaw, roll = angle_to_degrees(parts["angle"])
            sharpness = compute_sharpness(valid_paths[j])
            all_states.append({
                "path": valid_paths[j],
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "exp": parts["exp"],
                "sh": parts["sh"],
                "sharpness": sharpness,
            })

        done = min(i + batch_size, total)
        if progress_callback:
            progress_callback(done, total, f"推理 {done}/{total}")

    keep, remove = dedup_states(
        all_states,
        yaw_grid=yaw_grid,
        pitch_grid=pitch_grid,
        roll_grid=roll_grid,
        exp_thresh=exp_thresh,
        sh_thresh=sh_thresh,
        keep_per_cluster=keep_per_cluster,
        protect_yaw=protect_yaw,
    )

    protected_count = sum(1 for s in keep if abs(s["yaw"]) > protect_yaw)

    default_backup = input_dir.parent / (input_dir.name + "_dedup_backup")

    if not dry_run and remove:
        bd = Path(backup_dir) if backup_dir else default_backup
        bd.mkdir(parents=True, exist_ok=True)
        moved = 0
        for s in remove:
            src_img = s["path"]
            dst_img = bd / src_img.name
            if dst_img.exists():
                dst_img = bd / f"dup_{src_img.stem}{src_img.suffix}"
            try:
                _move_with_metadata(src_img, dst_img)
                moved += 1
            except Exception:
                pass

    report_dir = Path(backup_dir) if backup_dir else default_backup
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.txt"

    coverage = analyze_coverage(keep, yaw_grid, pitch_grid)
    report_text = generate_report(
        keep, remove,
        [s for s in keep if abs(s["yaw"]) > protect_yaw],
        len(all_states), coverage, report_path,
        yaw_grid, pitch_grid,
    )

    return keep, remove, report_text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="src 数据集去冗余 (3DDFA) - 角度/表情/光照全覆盖",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", help="src aligned 图片目录")
    parser.add_argument("--onnx_path", default=str(V3_ONNX), help="3DDFA ONNX 路径")
    parser.add_argument("--backup_dir", default=None, help="冗余帧备份目录")
    parser.add_argument("--dry_run", action="store_true", help="仅分析不移动文件")
    parser.add_argument("--batch_size", type=int, default=32, help="推理批大小")
    parser.add_argument("--keep_per_cluster", type=int, default=2, help="每簇保留几张")
    parser.add_argument("--yaw_grid", type=float, default=5.0, help="yaw 网格(度)")
    parser.add_argument("--pitch_grid", type=float, default=5.0, help="pitch 网格(度)")
    parser.add_argument("--roll_grid", type=float, default=5.0, help="roll 网格(度)")
    parser.add_argument("--exp_thresh", type=float, default=1.0, help="表情相似阈值")
    parser.add_argument("--sh_thresh", type=float, default=0.3, help="光照相似阈值")
    parser.add_argument("--protect_yaw", type=float, default=45.0, help="大角度保护阈值")
    parser.add_argument("--use_cpu", action="store_true", help="用 CPU 推理")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 目录不存在 {input_dir}")
        sys.exit(1)

    if not Path(args.onnx_path).exists():
        print(f"错误: ONNX 不存在 {args.onnx_path}")
        sys.exit(1)

    def _progress(current, total, msg):
        print(f"  {msg}")

    keep, remove, report_text = run_dedup(
        input_dir,
        onnx_path=args.onnx_path,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        keep_per_cluster=args.keep_per_cluster,
        yaw_grid=args.yaw_grid,
        pitch_grid=args.pitch_grid,
        roll_grid=args.roll_grid,
        exp_thresh=args.exp_thresh,
        sh_thresh=args.sh_thresh,
        protect_yaw=args.protect_yaw,
        use_cpu=args.use_cpu,
        progress_callback=_progress,
    )

    print(f"\n{'='*50}")
    print(f"结果: 保留 {len(keep)}, 移除 {len(remove)}")
    if remove:
        print(f"精简率: {len(remove)/(len(keep)+len(remove))*100:.1f}%")
    if args.dry_run:
        print("[dry_run] 未实际移动文件")
    print(f"\n报告已保存")


if __name__ == "__main__":
    main()
