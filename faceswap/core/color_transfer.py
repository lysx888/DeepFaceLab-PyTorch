import cv2
import numpy as np
from numpy import linalg as npla

from faceswap.shared.logger import get_logger

_logger = get_logger("color_transfer")


def _safe_mean_std(arr: np.ndarray) -> tuple[float, float]:
    m = arr.mean()
    s = arr.std()
    if s < 1e-6:
        s = 1e-6
    return float(m), float(s)


def _masked_mean_std(arr: np.ndarray, mask: np.ndarray = None,
                     cutoff: float = 0.5) -> tuple[float, float]:
    if mask is None:
        return _safe_mean_std(arr)
    valid = mask.reshape(arr.shape[:2]) >= cutoff
    n = valid.sum()
    if n < 1:
        _logger.warning("_masked_mean_std: no valid pixels in mask, falling back to full image stats")
        return _safe_mean_std(arr)
    pixels = arr[valid]
    m = float(pixels.mean())
    s = float(pixels.std())
    if s < 1e-6:
        s = 1e-6
    return m, s


def reinhard_color_transfer(target: np.ndarray, source: np.ndarray,
                            target_mask: np.ndarray = None,
                            source_mask: np.ndarray = None,
                            mask_cutoff: float = 0.5) -> np.ndarray:
    if target.dtype != np.float32:
        target = np.asarray(target, dtype=np.float32)
    if source.dtype != np.float32:
        source = np.asarray(source, dtype=np.float32)
    target = np.ascontiguousarray(target)
    source = np.ascontiguousarray(source)

    if target_mask is not None and target_mask.dtype != np.float32:
        target_mask = np.asarray(target_mask, dtype=np.float32)
    if source_mask is not None and source_mask.dtype != np.float32:
        source_mask = np.asarray(source_mask, dtype=np.float32)

    if target.size == 0 or source.size == 0:
        _logger.warning("reinhard_color_transfer: empty input, returning target unchanged")
        return target

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    t_l_mean, t_l_std = _masked_mean_std(target_lab[..., 0], target_mask, mask_cutoff)
    t_a_mean, t_a_std = _masked_mean_std(target_lab[..., 1], target_mask, mask_cutoff)
    t_b_mean, t_b_std = _masked_mean_std(target_lab[..., 2], target_mask, mask_cutoff)

    s_l_mean, s_l_std = _masked_mean_std(source_lab[..., 0], source_mask, mask_cutoff)
    s_a_mean, s_a_std = _masked_mean_std(source_lab[..., 1], source_mask, mask_cutoff)
    s_b_mean, s_b_std = _masked_mean_std(source_lab[..., 2], source_mask, mask_cutoff)

    target_l = (target_lab[..., 0] - t_l_mean) * (s_l_std / t_l_std) + s_l_mean
    target_a = (target_lab[..., 1] - t_a_mean) * (s_a_std / t_a_std) + s_a_mean
    target_b = (target_lab[..., 2] - t_b_mean) * (s_b_std / t_b_std) + s_b_mean

    np.clip(target_l, 0, 100, out=target_l)
    np.clip(target_a, -127, 127, out=target_a)
    np.clip(target_b, -127, 127, out=target_b)

    out = cv2.cvtColor(np.stack([target_l, target_a, target_b], -1).astype(np.float32),
                       cv2.COLOR_LAB2BGR)
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def linear_color_transfer(target_img: np.ndarray, source_img: np.ndarray,
                          mode: str = 'pca', eps: float = 1e-5) -> np.ndarray:
    if target_img.size == 0 or source_img.size == 0:
        _logger.warning("linear_color_transfer: empty input, returning target unchanged")
        return target_img
    try:
        mu_t = target_img.mean(0).mean(0)
        t = target_img - mu_t
        t = t.transpose(2, 0, 1).reshape(t.shape[-1], -1)
        Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])

        mu_s = source_img.mean(0).mean(0)
        s = source_img - mu_s
        s = s.transpose(2, 0, 1).reshape(s.shape[-1], -1)
        Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])

        if mode == 'chol':
            chol_t = np.linalg.cholesky(Ct)
            chol_s = np.linalg.cholesky(Cs)
            ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
        elif mode == 'pca':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t.clip(eps, None)))).dot(eve_t.T)
            eva_s, eve_s = np.linalg.eigh(Cs)
            Qs = eve_s.dot(np.sqrt(np.diag(eva_s.clip(eps, None)))).dot(eve_s.T)
            ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
        elif mode == 'sym':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t.clip(eps, None)))).dot(eve_t.T)
            Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
            eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
            QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt.clip(eps, None)))).dot(eve_QtCsQt.T)
            ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
        else:
            raise ValueError(f"Unknown LCT mode: {mode}")
    except np.linalg.LinAlgError as e:
        _logger.warning(f"linear_color_transfer: matrix computation failed ({e}), returning target unchanged")
        return target_img

    matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)


def color_transfer_mkl(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    if x0.size == 0 or x1.size == 0:
        _logger.warning("color_transfer_mkl: empty input, returning x0 unchanged")
        return x0
    try:
        eps = np.finfo(float).eps
        h, w, c = x0.shape

        x0_flat = x0.reshape(h * w, c)
        x1_flat = x1.reshape(h * w, c)

        a = np.cov(x0_flat.T)
        b = np.cov(x1_flat.T)

        Da2, Ua = np.linalg.eig(a)
        Da = np.diag(np.sqrt(Da2.clip(eps, None)))

        C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

        Dc2, Uc = np.linalg.eig(C)
        Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

        Da_inv = np.diag(1.0 / (np.diag(Da) + eps))

        t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

        mx0 = np.mean(x0_flat, axis=0)
        mx1 = np.mean(x1_flat, axis=0)

        result = np.dot(x0_flat - mx0, t) + mx1
    except np.linalg.LinAlgError as e:
        _logger.warning(f"color_transfer_mkl: matrix computation failed ({e}), returning x0 unchanged")
        return x0
    return np.clip(result.reshape(h, w, c).astype(x0.dtype), 0, 1)


def color_transfer_idt(i0: np.ndarray, i1: np.ndarray,
                       bins: int = 256, n_rot: int = 20) -> np.ndarray:
    if i0.size == 0 or i1.size == 0:
        _logger.warning("color_transfer_idt: empty input, returning i0 unchanged")
        return i0
    try:
        import scipy.stats

        relaxation = 1.0 / n_rot
        h, w, c = i0.shape

        d0 = i0.reshape(h * w, c).T.copy()
        d1 = i1.reshape(h * w, c).T.copy()

        for _ in range(n_rot):
            r = scipy.stats.special_ortho_group.rvs(c).astype(np.float32)

            d0r = np.dot(r, d0)
            d1r = np.dot(r, d1)
            d_r = np.empty_like(d0)

            for j in range(c):
                lo = min(d0r[j].min(), d1r[j].min())
                hi = max(d0r[j].max(), d1r[j].max())

                p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
                p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

                cp0r = p0r.cumsum().astype(np.float32)
                cp0r /= cp0r[-1]
                cp1r = p1r.cumsum().astype(np.float32)
                cp1r /= cp1r[-1]

                f = np.interp(cp0r, cp1r, edges[1:])
                d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

            d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

        return np.clip(d0.T.reshape(h, w, c).astype(i0.dtype), 0, 1)
    except Exception as e:
        _logger.warning(f"color_transfer_idt failed ({e}), returning i0 unchanged")
        return i0


def color_transfer_sot(src: np.ndarray, trg: np.ndarray,
                       steps: int = 10, batch_size: int = 5,
                       reg_sigmaXY: float = 16.0,
                       reg_sigmaV: float = 5.0) -> np.ndarray:
    if src.size == 0 or trg.size == 0:
        _logger.warning("color_transfer_sot: empty input, returning src unchanged")
        return src
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg must be float")

    try:
        h, w, c = src.shape
        new_src = src.copy()

        advect = np.empty((h * w, c), dtype=src.dtype)
        for _ in range(steps):
            advect.fill(0)
            for _ in range(batch_size):
                dir_v = np.random.normal(size=c).astype(src.dtype)
                dir_v /= npla.norm(dir_v)

                projsource = np.sum(new_src * dir_v, axis=-1).reshape(h * w)
                projtarget = np.sum(trg * dir_v, axis=-1).reshape(h * w)

                idSource = np.argsort(projsource)
                idTarget = np.argsort(projtarget)

                a = projtarget[idTarget] - projsource[idSource]
                for i_c in range(c):
                    advect[idSource, i_c] += a * dir_v[i_c]
            new_src += advect.reshape(h, w, c) / batch_size

        if reg_sigmaXY != 0.0:
            src_diff = new_src - src
            src_diff_filt = cv2.bilateralFilter(src_diff, 0, reg_sigmaV, reg_sigmaXY)
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
        return new_src
    except Exception as e:
        _logger.warning(f"color_transfer_sot failed ({e}), returning src unchanged")
        return src


def color_transfer_hist_match(target: np.ndarray, source: np.ndarray,
                              target_mask: np.ndarray = None,
                              source_mask: np.ndarray = None,
                              mask_cutoff: float = 0.5) -> np.ndarray:
    if target.size == 0 or source.size == 0:
        _logger.warning("color_transfer_hist_match: empty input, returning target unchanged")
        return target
    if target.dtype != np.float32:
        target = np.asarray(target, dtype=np.float32)
    if source.dtype != np.float32:
        source = np.asarray(source, dtype=np.float32)

    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    result_lab = target_lab.copy()
    for ch in range(3):
        t_ch = target_lab[..., ch].ravel()
        s_ch = source_lab[..., ch].ravel()

        if target_mask is not None:
            t_valid = target_mask.reshape(target_lab.shape[:2]).ravel() >= mask_cutoff
            t_indices = np.where(t_valid)[0]
            t_valid_ch = t_ch[t_indices]
            t_order = np.argsort(t_valid_ch)
        else:
            t_indices = np.arange(len(t_ch))
            t_valid_ch = t_ch
            t_order = np.argsort(t_valid_ch)

        if source_mask is not None:
            s_valid = source_mask.reshape(source_lab.shape[:2]).ravel() >= mask_cutoff
            s_valid_ch = s_ch[s_valid]
        else:
            s_valid_ch = s_ch

        s_order = np.argsort(s_valid_ch)
        n = len(t_valid_ch)
        matched_valid = np.empty(n, dtype=np.float32)
        if len(s_valid_ch) >= n:
            matched_valid[t_order] = s_valid_ch[s_order[:n]]
        else:
            matched_valid[t_order] = s_valid_ch[s_order[:len(s_valid_ch)] % len(s_valid_ch)]

        result_lab[..., ch].ravel()[t_indices] = matched_valid

    out = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def color_transfer_lab_match(target: np.ndarray, source: np.ndarray,
                             target_mask: np.ndarray = None,
                             source_mask: np.ndarray = None,
                             mask_cutoff: float = 0.5) -> np.ndarray:
    if target.size == 0 or source.size == 0:
        _logger.warning("color_transfer_lab_match: empty input, returning target unchanged")
        return target
    if target.dtype != np.float32:
        target = np.asarray(target, dtype=np.float32)
    if source.dtype != np.float32:
        source = np.asarray(source, dtype=np.float32)

    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    result_lab = target_lab.copy()

    for ch in (1, 2):
        t_mean, t_std = _masked_mean_std(target_lab[..., ch], target_mask, mask_cutoff)
        s_mean, s_std = _masked_mean_std(source_lab[..., ch], source_mask, mask_cutoff)
        result_lab[..., ch] = (target_lab[..., ch] - t_mean) * (s_std / t_std) + s_mean

    np.clip(result_lab[..., 1], -127, 127, out=result_lab[..., 1])
    np.clip(result_lab[..., 2], -127, 127, out=result_lab[..., 2])

    out = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def color_transfer_rct_partial(target: np.ndarray, source: np.ndarray,
                               blend: float = 0.5,
                               target_mask: np.ndarray = None,
                               source_mask: np.ndarray = None,
                               mask_cutoff: float = 0.5) -> np.ndarray:
    transferred = reinhard_color_transfer(target, source,
                                          target_mask=target_mask,
                                          source_mask=source_mask,
                                          mask_cutoff=mask_cutoff)
    return (1.0 - blend) * target + blend * transferred


def color_transfer(ct_mode: str, img_src: np.ndarray, img_trg: np.ndarray,
                   src_mask: np.ndarray = None,
                   trg_mask: np.ndarray = None,
                   mask_cutoff: float = 0.5) -> np.ndarray:
    mk = dict(target_mask=trg_mask, source_mask=src_mask, mask_cutoff=mask_cutoff)
    if ct_mode == 'lct':
        return linear_color_transfer(img_src, img_trg)
    elif ct_mode == 'rct':
        return reinhard_color_transfer(img_src, img_trg, **mk)
    elif ct_mode == 'rct-p':
        return color_transfer_rct_partial(img_src, img_trg, blend=0.5, **mk)
    elif ct_mode == 'mkl':
        return color_transfer_mkl(img_src, img_trg)
    elif ct_mode == 'idt':
        return color_transfer_idt(img_src, img_trg)
    elif ct_mode == 'sot':
        out = color_transfer_sot(img_src, img_trg)
        return np.clip(out, 0.0, 1.0)
    elif ct_mode == 'hist-match':
        return color_transfer_hist_match(img_src, img_trg, **mk)
    elif ct_mode == 'lab-match':
        return color_transfer_lab_match(img_src, img_trg, **mk)
    else:
        raise ValueError(f"Unknown ct_mode: {ct_mode}")
