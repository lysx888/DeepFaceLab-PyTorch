import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import orjson

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.models.face_enhancer import FaceEnhancer
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("face_tool")


class FaceTool:
    def __init__(self) -> None:
        pass

    def enhance(self, aligned_dir: Path, device: str = "auto") -> int:
        enhancer = FaceEnhancer(device=device)
        return enhancer.process_folder(aligned_dir)

    def pack(self, aligned_dir: Path, output_path: Optional[Path] = None,
             progress_callback=None) -> Path:
        """将对齐目录打包为 faceset.pak (zip 容器: images/* + meta/*.json).

        - ZIP_STORED 容器打包: 只封装不压缩, CPU 占用低, 速度接近复制;
        - 每张图附带 sidecar 元数据 json, 解包后完整还原 (对齐图 + 元数据)。
        - progress_callback(current, total, msg) 可选进度回调。
        """
        aligned_dir = Path(aligned_dir)
        if output_path is None:
            output_path = aligned_dir / "faceset.pak"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import zipfile
        imgs = FileManager.find_images(aligned_dir)
        total = len(imgs)
        tmp_path = output_path.with_name(output_path.name + ".tmp")

        def _notify(cur, msg):
            if progress_callback:
                progress_callback(cur, total, msg)

        with zipfile.ZipFile(str(tmp_path), "w", zipfile.ZIP_STORED) as zf:
            for i, img_path in enumerate(imgs, 1):
                zf.write(str(img_path), f"images/{img_path.name}")
                meta = MetadataManager.load(img_path)
                if meta is not None:
                    zf.writestr(f"meta/{img_path.name}.json", orjson.dumps(meta.to_dict()))
                _notify(i, f"打包 {i}/{total}: {img_path.name}")
        if tmp_path.exists():
            tmp_path.replace(output_path)
        _notify(total, f"完成: {output_path.name}")
        return output_path

    def unpack(self, pack_path: Path, aligned_dir: Path, progress_callback=None) -> int:
        """解压 faceset.pak 到指定目录, 支持 zip 容器 (新格式) 与单 JSON (旧格式).

        progress_callback(current, total, msg) 可选进度回调。
        """
        pack_path = Path(pack_path)
        aligned_dir = Path(aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        # 新格式: zip 容器
        if pack_path.is_file():
            try:
                with open(str(pack_path), "rb") as f:
                    head = f.read(4)
            except OSError:
                head = b""
            if head == b"PK\x03\x04":
                import zipfile
                with zipfile.ZipFile(str(pack_path)) as zf:
                    names = zf.namelist()
                    img_names = [n for n in names if n.startswith("images/")]
                    total = len(img_names)
                    for i, n in enumerate(img_names, 1):
                        fname = Path(n).name
                        img_path = aligned_dir / fname
                        with zf.open(n) as src, open(str(img_path), "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        meta_name = f"meta/{fname}.json"
                        if meta_name in names:
                            meta = FaceMetadata.from_dict(orjson.loads(zf.read(meta_name)))
                            MetadataManager.save(img_path, meta)
                        if progress_callback:
                            progress_callback(i, total, f"解压 {i}/{total}: {fname}")
                if progress_callback:
                    progress_callback(total, total, f"完成: {total} 张")
                return total

        # 旧格式: 单 JSON {filename: {image: base64, metadata}}
        import base64
        with open(str(pack_path), "rb") as f:
            pack_data = orjson.loads(f.read())

        total = len(pack_data)
        count = 0
        for i, (filename, entry) in enumerate(pack_data.items(), 1):
            img_bytes = base64.b64decode(entry["image"])
            img_path = aligned_dir / filename
            FileManager.atomic_write(img_path, img_bytes)

            if "metadata" in entry:
                meta = FaceMetadata.from_dict(entry["metadata"])
                MetadataManager.save(img_path, meta)
            count += 1
            if progress_callback:
                progress_callback(i, total, f"解压 {i}/{total}: {filename}")
        if progress_callback:
            progress_callback(total, total, f"完成: {total} 张")
        return count

    def resize(self, aligned_dir: Path, target_size: int) -> int:
        aligned_dir = Path(aligned_dir)
        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            from faceswap.shared.file_manager import imwrite_auto
            imwrite_auto(img_path, resized)

            if meta is not None:
                s = scale
                meta.landmarks_106 = (meta.landmarks_106.astype(np.float64) * s).astype(np.int64)
                # source_landmarks_106 / source_kps_5 / source_rect 属于原图坐标系,
                # 原图未缩放, 不可随对齐图一起缩放, 否则调试图预览/手动标注回显会漂移。
                # image_to_face_mat 是原图->对齐图的正向仿射变换, 缩放对齐图等价于
                # 在输出侧乘对角缩放: M' = S·M, 即全部 6 个元素都乘 s。
                # 只乘平移会导致 M'·source != s·landmarks_106, 与对齐图坐标自相矛盾。
                if meta.image_to_face_mat is not None:
                    mat = meta.image_to_face_mat.astype(np.float64).copy()
                    mat *= s
                    meta.image_to_face_mat = mat
                if meta.seg_ie_polys is not None:
                    for poly in meta.seg_ie_polys:
                        if "pts" in poly:
                            poly["pts"] = [[pt[0] * s, pt[1] * s] for pt in poly["pts"]]
                if meta.xseg_mask is not None:
                    old_mask = meta.get_xseg_mask_array()
                    if old_mask is not None:
                        new_mask = cv2.resize(old_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                        meta.xseg_mask = FaceMetadata.encode_xseg_mask(new_mask)
                meta.output_size = target_size
                MetadataManager.save(img_path, meta)
            count += 1

        return count

    def recover_original_filename(self, aligned_dir: Path, index_progress_callback=None) -> int:
        aligned_dir = Path(aligned_dir)
        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue

            original = meta.source_filename
            if not original:
                continue

            new_path = aligned_dir / original
            if new_path.exists() and new_path != img_path:
                _logger.warning(f"Skip recover: target already exists {new_path.name} (source: {img_path.name})")
                continue

            old_json = MetadataManager.sidecar_path(img_path)
            if img_path != new_path:
                img_path.rename(new_path)
                if old_json.exists():
                    old_json.rename(MetadataManager.sidecar_path(new_path))
                count += 1

        MetadataManager.build_source_index(aligned_dir, progress_callback=index_progress_callback)
        return count

    def add_landmarks_debug_images(self, aligned_dir: Path, output_dir: Optional[Path] = None) -> int:
        aligned_dir = Path(aligned_dir)
        out_dir = Path(output_dir) if output_dir else aligned_dir / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue

            from faceswap.core.landmarks106 import fill_hull_mask_106
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lm = meta.landmarks_106.astype(np.float32)
            for pt_idx in range(lm.shape[0]):
                x, y = int(lm[pt_idx, 0]), int(lm[pt_idx, 1])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            hull_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            fill_hull_mask_106(hull_mask, lm)
            img[hull_mask > 0] = (img[hull_mask > 0] * 0.7).astype(np.uint8)

            debug_path = out_dir / f"debug_{img_path.name}"
            cv2.imwrite(str(debug_path), img)
            count += 1

        return count

    def view_aligned_result(self, aligned_dir: Path) -> None:
        aligned_dir = Path(aligned_dir)
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(aligned_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(aligned_dir)])
        else:
            subprocess.Popen(["xdg-open", str(aligned_dir)])

    def metadata_save(self, aligned_dir: Path) -> Path:
        output_path = aligned_dir / "meta.dat"
        MetadataManager.pack_metadata(aligned_dir, output_path)
        return output_path

    def metadata_restore(self, aligned_dir: Path) -> int:
        pack_path = aligned_dir / "meta.dat"
        if not pack_path.exists():
            raise FileNotFoundError(f"Metadata pack not found: {pack_path}")
        MetadataManager.unpack_metadata(pack_path, aligned_dir)
        return 0
