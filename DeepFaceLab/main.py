import argparse
import sys
from pathlib import Path

from DeepFaceLab.shared.torch_config import configure_torch
configure_torch("gpu_train")

import torch

from DeepFaceLab.setting import (
    WORKSPACE_DIR, MODEL_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_ALIGNED_DEBUG_DIR,
    DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
    FaceType,
)
from DeepFaceLab.shared.logger import setup_logger, get_logger

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(prog="DeepFaceLab", description="DeepFaceLab PyTorch Implementation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # clear workspace
    subparsers.add_parser("clear-workspace", help="Clear workspace and rebuild directory structure")

    # video extraction
    p = subparsers.add_parser("extract-video-src", help="Extract frames from source video")
    p.add_argument("--input", type=Path, help="Source video path (default: workspace/data_src.mp4)")
    p.add_argument("--output-dir", type=Path, default=DATA_SRC_DIR, help="Output directory")
    p.add_argument("--fps", type=float, default=None, help="Target FPS")
    p.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"])

    p = subparsers.add_parser("extract-video-dst", help="Extract frames from destination video")
    p.add_argument("--input", type=Path, help="Destination video path")
    p.add_argument("--output-dir", type=Path, default=DATA_DST_DIR)

    p = subparsers.add_parser("cut-video", help="Cut video")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=str, required=True, help="Start time (HH:MM:SS)")
    p.add_argument("--end", type=str, required=True, help="End time (HH:MM:SS)")

    subparsers.add_parser("denoise-dst", help="Denoise destination frames")

    # face extraction
    p = subparsers.add_parser("extract-faces-src", help="Extract source faces")
    p.add_argument("--input-dir", type=Path, default=DATA_SRC_DIR)
    p.add_argument("--output-dir", type=Path, default=DATA_SRC_ALIGNED_DIR)
    p.add_argument("--face-type", type=str, default="whole_face", choices=["half", "mid_full", "full", "whole_face", "head"])
    p.add_argument("--max-faces", type=int, default=0)
    p.add_argument("--det-thresh", type=float, default=0.5)
    p.add_argument("--output-size", type=int, default=512)

    p = subparsers.add_parser("extract-faces-dst", help="Extract destination faces")
    p.add_argument("--input-dir", type=Path, default=DATA_DST_DIR)
    p.add_argument("--output-dir", type=Path, default=DATA_DST_ALIGNED_DIR)
    p.add_argument("--face-type", type=str, default="whole_face")
    p.add_argument("--max-faces", type=int, default=0)
    p.add_argument("--det-thresh", type=float, default=0.5)
    p.add_argument("--output-size", type=int, default=512)
    p.add_argument("--debug", action="store_true")

    # sorting
    p = subparsers.add_parser("sort-src", help="Sort source faces")
    p.add_argument("--input-dir", type=Path, default=DATA_SRC_ALIGNED_DIR)
    p.add_argument("--algorithm", type=str, default="blur", choices=["blur", "hist", "yaw", "pitch", "brightness", "hue", "oneface", "final"])

    p = subparsers.add_parser("sort-dst", help="Sort destination faces")
    p.add_argument("--input-dir", type=Path, default=DATA_DST_ALIGNED_DIR)
    p.add_argument("--algorithm", type=str, default="blur")

    # face tools
    subparsers.add_parser("enhance-src", help="Enhance source faces")
    subparsers.add_parser("enhance-dst", help="Enhance destination faces")

    p = subparsers.add_parser("pack-src", help="Pack source faceset")
    p.add_argument("--input-dir", type=Path, default=DATA_SRC_ALIGNED_DIR)
    p = subparsers.add_parser("pack-dst", help="Pack destination faceset")
    p.add_argument("--input-dir", type=Path, default=DATA_DST_ALIGNED_DIR)

    p = subparsers.add_parser("unpack-src", help="Unpack source faceset")
    p.add_argument("--input-dir", type=Path, default=DATA_SRC_ALIGNED_DIR)
    p = subparsers.add_parser("unpack-dst", help="Unpack destination faceset")
    p.add_argument("--input-dir", type=Path, default=DATA_DST_ALIGNED_DIR)

    subparsers.add_parser("metadata-save-src", help="Save source faceset metadata")
    subparsers.add_parser("metadata-restore-src", help="Restore source faceset metadata")
    subparsers.add_parser("metadata-save-dst", help="Save destination faceset metadata")
    subparsers.add_parser("metadata-restore-dst", help="Restore destination faceset metadata")

    p = subparsers.add_parser("resize-src", help="Resize source faceset")
    p.add_argument("--size", type=int, default=256)
    p = subparsers.add_parser("resize-dst", help="Resize destination faceset")
    p.add_argument("--size", type=int, default=256)

    subparsers.add_parser("recover-filename-src", help="Recover original filenames for source faces")
    subparsers.add_parser("landmarks-debug-src", help="Add landmarks debug images for source faces")
    subparsers.add_parser("landmarks-debug-dst", help="Add landmarks debug images for destination faces")

    # xseg
    subparsers.add_parser("xseg-edit-src", help="Open XSeg editor for source faces")
    subparsers.add_parser("xseg-edit-dst", help="Open XSeg editor for destination faces")
    subparsers.add_parser("xseg-fetch-src", help="Fetch annotated source faces")
    subparsers.add_parser("xseg-fetch-dst", help="Fetch annotated destination faces")
    subparsers.add_parser("xseg-remove-src", help="Remove XSeg annotations from source faces")
    subparsers.add_parser("xseg-remove-dst", help="Remove XSeg annotations from destination faces")

    p = subparsers.add_parser("xseg-train", help="Train XSeg model")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)

    subparsers.add_parser("xseg-apply-src", help="Apply trained XSeg mask to source faces")
    subparsers.add_parser("xseg-apply-dst", help="Apply trained XSeg mask to destination faces")
    subparsers.add_parser("xseg-remove-trained-src", help="Remove trained XSeg mask from source faces")
    subparsers.add_parser("xseg-remove-trained-dst", help="Remove trained XSeg mask from destination faces")
    subparsers.add_parser("xseg-generic-apply-src", help="Apply generic XSeg mask to source faces")
    subparsers.add_parser("xseg-generic-apply-dst", help="Apply generic XSeg mask to destination faces")

    # training
    p = subparsers.add_parser("train-saehd", help="Train SAEHD model")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--architecture", type=str, default="df", choices=["df", "liae"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--use-amp", action="store_true", default=True)

    p = subparsers.add_parser("train-quick96", help="Train Quick96 model")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)

    p = subparsers.add_parser("train-amp", help="Train AMP model")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--src-src-mode", action="store_true")

    subparsers.add_parser("export-dfm-saehd", help="Export SAEHD as DFM")
    subparsers.add_parser("export-dfm-amp", help="Export AMP as DFM")

    # insightface training
    subparsers.add_parser("insightface-train-init", help="Initialize insightface training directories")

    p = subparsers.add_parser("insightface-prepare-scrfd", help="Prepare SCRFD training data")
    p.add_argument("--input", type=Path, help="Input data directory (WIDERFace)")
    p.add_argument("--format", type=str, default="manual_annotated", choices=["manual_annotated", "widerface"])
    p.add_argument("--val-ratio", type=float, default=0.1)

    p = subparsers.add_parser("insightface-prepare-synthetics", help="Prepare 2d106 training data")
    p.add_argument("--input", type=Path, help="Input data directory (FaceSynthetics)")
    p.add_argument("--format", type=str, default="manual_annotated", choices=["manual_annotated", "facesynthetics"])
    p.add_argument("--input-size", type=int, default=256)

    p = subparsers.add_parser("insightface-train-scrfd", help="Start SCRFD training")
    p.add_argument("--config", type=str, default="dfl_scrfd_10g_bnkps")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--resume-from", type=str, default=None)

    p = subparsers.add_parser("insightface-train-synthetics", help="Start 2d106 training")
    p.add_argument("--backbone", type=str, default="resnet50d")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=80)

    p = subparsers.add_parser("insightface-export-scrfd", help="Export SCRFD ONNX model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--deploy", action="store_true")

    p = subparsers.add_parser("insightface-export-synthetics", help="Export 2d106 ONNX model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--deploy", action="store_true")

    p = subparsers.add_parser("insightface-deploy", help="Deploy ONNX model to antelopev2")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--model-name", type=str, required=True, choices=["scrfd_10g_bnkps.onnx", "2d106det.onnx"])
    p.add_argument("--no-backup", action="store_true")

    # merge
    p = subparsers.add_parser("merge-saehd", help="Merge with SAEHD model")
    p.add_argument("--mask-mode", type=str, default="xseg", choices=["dst", "xseg", "learned"])

    subparsers.add_parser("merge-quick96", help="Merge with Quick96 model")
    subparsers.add_parser("merge-amp", help="Merge with AMP model")
    subparsers.add_parser("merge-inswapper", help="Merge with INSwapper")

    # video output
    p = subparsers.add_parser("merged-to-avi", help="Convert merged frames to AVI")
    p.add_argument("--lossless", action="store_true")
    p = subparsers.add_parser("merged-to-mp4", help="Convert merged frames to MP4")
    p.add_argument("--lossless", action="store_true")
    p = subparsers.add_parser("merged-to-mov", help="Convert merged frames to MOV")
    p.add_argument("--lossless", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    try:
        _dispatch(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def _face_type_from_str(s: str) -> FaceType:
    return {
        "half": FaceType.HALF,
        "mid_full": FaceType.MID_FULL,
        "full": FaceType.FULL,
        "whole_face": FaceType.WHOLE_FACE,
        "head": FaceType.HEAD,
    }[s.lower()]


def _dispatch(args):
    from DeepFaceLab.business.workspace_manager import WorkspaceManager
    from DeepFaceLab.business.video_processor import VideoProcessor
    from DeepFaceLab.business.face_extractor import FaceExtractor, ExtractConfig
    from DeepFaceLab.business.face_sorter import FaceSorter, SortAlgorithm
    from DeepFaceLab.business.face_tool import FaceTool
    from DeepFaceLab.business.xseg_editor import XSegEditor
    from DeepFaceLab.business.xseg_trainer import XSegTrainer
    from DeepFaceLab.business.model_trainer import ModelTrainer, ModelType
    from DeepFaceLab.business.model_merger import ModelMerger, MergeConfig, MaskMode
    from DeepFaceLab.business.video_output import VideoOutput, OutputFormat
    from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter

    ws = WorkspaceManager()
    cmd = args.command

    if cmd == "clear-workspace":
        ws.clear_workspace()

    elif cmd in ("extract-video-src", "extract-video-dst"):
        vp = VideoProcessor()
        if cmd == "extract-video-src":
            video = args.input or ws.find_src_video()
            vp.extract_frames_src(video, args.output_dir, fps=args.fps, output_format=args.format)
        else:
            video = args.input or ws.find_dst_video()
            vp.extract_frames_dst(video, args.output_dir)

    elif cmd == "cut-video":
        vp = VideoProcessor()
        vp.cut_video(args.input, args.output, args.start, args.end)

    elif cmd == "denoise-dst":
        vp = VideoProcessor()
        vp.denoise_frames(DATA_DST_DIR)

    elif cmd in ("extract-faces-src", "extract-faces-dst"):
        output_size = args.output_size
        face_type_str = args.face_type
        det_thresh = args.det_thresh
        max_faces = args.max_faces
        jpg_quality = 100
        debug = getattr(args, "debug", False)

        gpu_idx = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_list = ", ".join(f"[{i}] {torch.cuda.get_device_name(i)}" for i in range(gpu_count))
            print(f"\n  [CPU] : CPU")
            print(f"  {gpu_list}")
            try:
                gpu_input = input(f"\n  Which GPU index to choose? [{gpu_idx}] : ").strip()
            except EOFError:
                gpu_input = ""
            if gpu_input:
                try:
                    gpu_idx = int(gpu_input)
                except ValueError:
                    pass

        try:
            ft_input = input(f"  Face type ( half/mid_full/full/wf/head ) [{face_type_str}] : ").strip().lower()
        except EOFError:
            ft_input = ""
        if ft_input:
            ft_map = {"h": "half", "mf": "mid_full", "f": "full", "wf": "whole_face", "head": "head",
                       "half": "half", "mid_full": "mid_full", "full": "full", "whole_face": "whole_face"}
            face_type_str = ft_map.get(ft_input, face_type_str)

        ft = _face_type_from_str(face_type_str)
        default_size = 768 if ft == FaceType.HEAD else 512
        output_size = default_size

        try:
            mf_input = input(f"  Max number of faces from image (0=unlimited) [{max_faces}] : ").strip()
        except EOFError:
            mf_input = ""
        if mf_input:
            try:
                max_faces = int(mf_input)
            except ValueError:
                pass

        try:
            sz_input = input(f"  Image size (128-2048, must be multiple of 128) [{output_size}] : ").strip()
        except EOFError:
            sz_input = ""
        if sz_input:
            try:
                val = int(sz_input)
                val = max(128, min(2048, (val // 128) * 128))
                if val < 128:
                    val = 128
                output_size = val
            except ValueError:
                pass

        try:
            q_input = input(f"  Jpeg quality (1-100) [{jpg_quality}] : ").strip()
        except EOFError:
            q_input = ""
        if q_input:
            try:
                jpg_quality = max(1, min(100, int(q_input)))
            except ValueError:
                pass

        try:
            db_input = input(f"  Write debug images to aligned_debug? (y/n) [{'y' if debug else 'n'}] : ").strip().lower()
        except EOFError:
            db_input = ""
        if db_input in ("y", "yes"):
            debug = True
        elif db_input in ("n", "no"):
            debug = False

        print(f"\n  Extracting faces on GPU {gpu_idx}...")

        adapter = InsightFaceAdapter(det_thresh=det_thresh, ctx_id=gpu_idx)
        extractor = FaceExtractor(adapter)
        ft = _face_type_from_str(face_type_str)
        config = ExtractConfig(
            face_type=ft, max_faces=max_faces,
            det_thresh=det_thresh, output_size=output_size,
            jpg_quality=jpg_quality, debug_output=debug,
        )
        if cmd == "extract-faces-src":
            extractor.extract_src_faces(args.input_dir, args.output_dir, config)
        else:
            extractor.extract_dst_faces(args.input_dir, args.output_dir, config)

    elif cmd in ("sort-src", "sort-dst"):
        sorter = FaceSorter()
        sorter.sort_aligned(args.input_dir, SortAlgorithm(args.algorithm))

    elif cmd in ("enhance-src", "enhance-dst"):
        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        tool.enhance(d)

    elif cmd in ("pack-src", "pack-dst"):
        tool = FaceTool()
        tool.pack(args.input_dir)

    elif cmd in ("unpack-src", "unpack-dst"):
        tool = FaceTool()
        tool.unpack(args.input_dir / "faceset.pak", args.input_dir)

    elif cmd in ("metadata-save-src", "metadata-save-dst"):
        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        tool.metadata_save(d)

    elif cmd in ("metadata-restore-src", "metadata-restore-dst"):
        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        tool.metadata_restore(d)

    elif cmd in ("resize-src", "resize-dst"):
        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        tool.resize(d, args.size)

    elif cmd == "recover-filename-src":
        FaceTool().recover_original_filename(DATA_SRC_ALIGNED_DIR)

    elif cmd in ("landmarks-debug-src", "landmarks-debug-dst"):
        tool = FaceTool()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        tool.add_landmarks_debug_images(d)

    elif cmd in ("xseg-edit-src", "xseg-edit-dst"):
        editor = XSegEditor()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        editor.open(d)

    elif cmd in ("xseg-fetch-src", "xseg-fetch-dst"):
        editor = XSegEditor()
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        editor.fetch_annotated(d)

    elif cmd in ("xseg-remove-src", "xseg-remove-dst"):
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        XSegEditor().remove_annotations(d)

    elif cmd == "xseg-train":
        trainer = XSegTrainer()
        trainer.train(DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                      resolution=args.resolution, batch_size=args.batch_size, epochs=args.epochs)

    elif cmd in ("xseg-apply-src", "xseg-apply-dst"):
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        XSegTrainer().apply_trained_mask(d, MODEL_DIR)

    elif cmd in ("xseg-remove-trained-src", "xseg-remove-trained-dst"):
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        XSegTrainer().remove_trained_mask(d)

    elif cmd in ("xseg-generic-apply-src", "xseg-generic-apply-dst"):
        d = DATA_SRC_ALIGNED_DIR if "src" in cmd else DATA_DST_ALIGNED_DIR
        from DeepFaceLab.setting import _PROJECT_ROOT
        generic_dir = _PROJECT_ROOT / "_internal" / "model_generic_xseg"
        XSegTrainer().apply_generic_mask(d, generic_dir)

    elif cmd == "train-saehd":
        trainer = ModelTrainer()
        cfg = trainer.configure_interactive(ModelType.SAEHD, MODEL_DIR)
        trainer.train(ModelType.SAEHD, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                      resolution=cfg.get("resolution", 128),
                      architecture=cfg.get("architecture", "df"),
                      batch_size=cfg.get("batch_size", 4),
                      learning_rate=cfg.get("learning_rate", 1e-4),
                      use_amp=cfg.get("use_amp", True),
                      face_type=cfg.get("face_type", "whole_face"),
                      ae_dims=cfg.get("ae_dims", 256),
                      e_dims=cfg.get("e_dims", 64),
                      d_dims=cfg.get("d_dims", 64),
                      random_warp=cfg.get("random_warp", True),
                      gan_power=cfg.get("gan_power", 0.0),
                      random_hsv_power=cfg.get("random_hsv_power", 0.0))

    elif cmd == "train-quick96":
        trainer = ModelTrainer()
        cfg = trainer.configure_interactive(ModelType.QUICK96, MODEL_DIR)
        trainer.train(ModelType.QUICK96, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                      batch_size=cfg.get("batch_size", 4),
                      learning_rate=cfg.get("learning_rate", 1e-4),
                      use_amp=cfg.get("use_amp", True))

    elif cmd == "train-amp":
        trainer = ModelTrainer()
        cfg = trainer.configure_interactive(ModelType.AMP, MODEL_DIR)
        trainer.train(ModelType.AMP, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, MODEL_DIR,
                      resolution=cfg.get("resolution", 128),
                      batch_size=cfg.get("batch_size", 4),
                      learning_rate=cfg.get("learning_rate", 1e-4),
                      use_amp=cfg.get("use_amp", True),
                      src_src_mode=cfg.get("src_src_mode", False))

    elif cmd == "export-dfm-saehd":
        ModelTrainer().export_dfm(ModelType.SAEHD, MODEL_DIR)

    elif cmd == "export-dfm-amp":
        ModelTrainer().export_dfm(ModelType.AMP, MODEL_DIR)

    elif cmd in ("merge-saehd", "merge-quick96", "merge-amp"):
        adapter = InsightFaceAdapter()
        merger = ModelMerger(adapter)
        model_type = cmd.replace("merge-", "").upper()
        mask_mode = MaskMode(getattr(args, "mask_mode", "xseg"))
        config = MergeConfig(mask_mode=mask_mode)
        merger.merge_auto(MODEL_DIR, model_type, DATA_DST_DIR, DATA_DST_MERGED_DIR,
                          DATA_DST_MERGED_MASK_DIR, DATA_DST_ALIGNED_DIR, config)

    elif cmd == "merge-inswapper":
        adapter = InsightFaceAdapter()
        merger = ModelMerger(adapter)
        merger.merge_inswapper(DATA_DST_DIR, DATA_DST_MERGED_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_DIR)

    elif cmd in ("merged-to-avi", "merged-to-mp4", "merged-to-mov"):
        vp = VideoProcessor()
        vo = VideoOutput(vp)
        fmt_map = {"merged-to-avi": OutputFormat.AVI, "merged-to-mp4": OutputFormat.MP4, "merged-to-mov": OutputFormat.MOV}
        ref_video = ws.find_dst_video()
        output_path = WORKSPACE_DIR / f"result.{fmt_map[cmd].value}"
        vo.merged_to_video(DATA_DST_MERGED_DIR, output_path, fmt_map[cmd],
                           reference_video=ref_video, lossless=getattr(args, "lossless", False))

    elif cmd == "insightface-train-init":
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer
        trainer = InsightFaceTrainer()
        trainer.init_train_dirs()
        print("insightface训练目录已初始化")

    elif cmd == "insightface-prepare-scrfd":
        from DeepFaceLab.business.scrfd_data_preparer import SCRFDDataPreparer
        preparer = SCRFDDataPreparer()
        if args.format == "manual_annotated":
            stats = preparer.prepare_from_manual_annotated(val_ratio=args.val_ratio)
        else:
            if not args.input:
                print("错误: WIDERFace格式需要指定 --input 目录")
                sys.exit(1)
            stats = preparer.prepare_from_widerface(args.input, val_ratio=args.val_ratio)
        print(f"SCRFD数据准备完成: {stats}")

    elif cmd == "insightface-prepare-synthetics":
        from DeepFaceLab.business.synthetics_data_preparer import SyntheticsDataPreparer
        preparer = SyntheticsDataPreparer()
        if args.format == "manual_annotated":
            stats = preparer.prepare_from_manual_annotated()
        else:
            if not args.input:
                print("错误: FaceSynthetics格式需要指定 --input 目录")
                sys.exit(1)
            stats = preparer.prepare_from_facesynthetics(args.input)
        print(f"2d106数据准备完成: {stats}")

    elif cmd == "insightface-train-scrfd":
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer, SCRFDTrainConfig
        trainer = InsightFaceTrainer()
        config = SCRFDTrainConfig(
            config_name=args.config,
            gpus=list(range(args.gpus)),
            resume_from=args.resume_from,
        )
        trainer.train_scrfd(config)
        print("SCRFD训练已启动")

    elif cmd == "insightface-train-synthetics":
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer, SyntheticsTrainConfig
        trainer = InsightFaceTrainer()
        config = SyntheticsTrainConfig(
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            max_epochs=args.max_epochs,
        )
        trainer.train_synthetics(config)
        print("2d106训练已启动")

    elif cmd == "insightface-export-scrfd":
        from DeepFaceLab.business.insightface_exporter import InsightFaceExporter
        exporter = InsightFaceExporter()
        result = exporter.export_scrfd_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=(args.input_size, args.input_size),
            deploy=args.deploy,
        )
        print(f"ONNX导出完成: {result.onnx_path}")
        print(f"一致性验证: {'通过' if result.is_consistent else '未通过'} (误差: {result.max_abs_error:.6f})")

    elif cmd == "insightface-export-synthetics":
        from DeepFaceLab.business.insightface_exporter import InsightFaceExporter
        exporter = InsightFaceExporter()
        result = exporter.export_synthetics_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=args.input_size,
            deploy=args.deploy,
        )
        print(f"ONNX导出完成: {result.onnx_path}")
        print(f"一致性验证: {'通过' if result.is_consistent else '未通过'} (误差: {result.max_abs_error:.6f})")

    elif cmd == "insightface-deploy":
        from DeepFaceLab.business.insightface_exporter import InsightFaceExporter
        exporter = InsightFaceExporter()
        exporter.deploy_to_antelopev2(args.onnx, args.model_name, backup=not args.no_backup)
        print(f"ONNX模型已部署: {args.model_name}")

    else:
        logger.error(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
