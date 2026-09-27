"""
run_native_fastmri_qnp.py

Run the ORIGINAL upstream QNP_WaveletTV algorithm directly on native
fastMRI Cartesian multicoil k-space instead of converting the scan to
an image and re-simulating a radial/spiral acquisition.

What changes
------------
- The upstream QNP optimization function is NOT modified.
- Native fastMRI multicoil k-space is used directly.
- Sensitivity maps are estimated from the acquired Cartesian data with ESPIRiT.
- No extra synthetic Gaussian noise is added.
- Optional retrospective Cartesian undersampling can be applied with
  --acceleration > 1.
- The original QNP Wavelet+TV solver is then run with a Cartesian SENSE
  forward model.

This is a new acquisition/model condition. It is not the same experiment as
the original upstream radial/spiral demo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr


EPS = 1e-12


# -----------------------------------------------------------------------------
# Vendored runtime subset of the original Apache-2.0 CQNPCS_MRIReco QNP code.
# Source: hongtao-argmin/CQNPCS_MRIReco (optalgTao.py + TVutilities.py).
# The QNP_WaveletTV update logic is retained unchanged; only unrelated upstream
# algorithms are omitted so this project can run from a single local folder.
# -----------------------------------------------------------------------------

from tqdm import tqdm


def _tv_lforward(P_1, P_2, TV_bound="Dirchlet", isComplex=False):
    m2, n2 = P_1.shape
    m1, n1 = P_2.shape
    if TV_bound == "Neumann":
        if n2 != n1 + 1:
            print("dimensions are not consistent\n")
        if m1 != m2 + 1:
            print("dimension are not consistent\n")
        m = m2 + 1
        n = n2
        X = np.zeros((m, n), dtype=np.complex128 if isComplex else float)
        X[0:m-1, :] = P_1
        X[:, 0:n-1] = X[:, 0:n-1] + P_2
        X[1:m, :] = X[1:m, :] - P_1
        X[:, 1:n] = X[:, 1:n] - P_2
    elif TV_bound == "Dirchlet":
        m = m2
        n = n2
        X = np.zeros((m, n), dtype=np.complex128 if isComplex else float)
        X[0:m-1, :] = P_1[0:m-1, :]
        X[:, 0:n-1] = X[:, 0:n-1] + P_2[:, 0:n-1]
        X[1:m, :] = X[1:m, :] - P_1[0:m-1, :]
        X[:, 1:n] = X[:, 1:n] - P_2[:, 0:n-1]
        X[0:m-1, n-1] = X[0:m-1, n-1] + P_2[0:m-1, n-1]
        X[m-1, 0:n-1] = X[m-1, 0:n-1] + P_1[m-1, 0:n-1]
        X[m-1, n-1] = X[m-1, n-1] + P_1[m-1, n-1] + P_2[m-1, n-1]
    elif TV_bound == "Periodic":
        m = m2
        n = n2
        X = np.zeros((m, n), dtype=np.complex128 if isComplex else float)
        X[0:m-1, :] = P_1[0:m-1, :]
        X[:, 0:n-1] = X[:, 0:n-1] + P_2[:, 0:n-1]
        X[1:m, :] = X[1:m, :] - P_1[0:m-1, :]
        X[:, 1:n] = X[:, 1:n] - P_2[:, 0:n-1]
        X[0:m-1, n-1] = X[0:m-1, n-1] + P_2[0:m-1, n-1]
        X[0:m-1, 0] = X[0:m-1, 0] - P_2[0:m-1, n-1]
        X[m-1, 0:n-1] = X[m-1, 0:n-1] + P_1[m-1, 0:n-1]
        X[0, 0:n-1] = X[0, 0:n-1] - P_1[m-1, 0:n-1]
        X[m-1, n-1] = X[m-1, n-1] + P_1[m-1, n-1] + P_2[m-1, n-1]
        X[0, n-1] = X[0, n-1] - P_1[m-1, n-1]
        X[m-1, 0] = X[m-1, 0] - P_2[m-1, n-1]
    else:
        raise ValueError(f"Unsupported TV boundary: {TV_bound}")
    return X


def _tv_ltrans(X, m, n, TV_bound):
    if TV_bound == "Neumann":
        P_1 = X[0:m-1, :] - X[1:m, :]
        P_2 = X[:, 0:n-1] - X[:, 1:n]
    elif TV_bound == "Dirchlet":
        P_1 = np.vstack((X[0:m-1, :] - X[1:m, :], X[m-1, :]))
        P_2 = np.column_stack((X[:, 0:n-1] - X[:, 1:n], X[:, n-1]))
    elif TV_bound == "Periodic":
        P_1 = X - np.vstack((X[1:m, :], X[0, :]))
        P_2 = X - np.column_stack((X[:, 1:n-1], X[:, 0]))
    else:
        raise ValueError(f"Unsupported TV boundary: {TV_bound}")
    return P_1, P_2


def _power_iter_handle(A, m, n, tol=1e-6):
    b_k = np.random.randn(m, n)
    Ab_k = A(b_k)
    norm_b_k = np.linalg.norm(Ab_k)
    while True:
        b_k_1 = Ab_k / norm_b_k
        if np.linalg.norm(b_k - b_k_1) <= tol:
            break
        b_k = b_k_1
        Ab_k = A(b_k_1)
        norm_b_k = np.linalg.norm(Ab_k)
    b = b_k_1
    L = np.vdot(b, Ab_k / np.vdot(b, b))
    return np.real(L)


def _obj_tv(X, TV_bound, TV_type):
    m, n = X.shape
    P_1, P_2 = _tv_ltrans(X, m, n, TV_bound)
    if TV_type == "iso":
        if TV_bound == "Neumann":
            D = np.zeros((m, n))
            D[0:m-1, :] = np.abs(P_1) ** 2
            D[:, 0:n-1] = D[:, 0:n-1] + np.abs(P_2) ** 2
            f_TV = np.sum(np.sqrt(D))
        else:
            f_TV = np.sum(np.sqrt(np.abs(P_1) ** 2 + np.abs(P_2) ** 2))
    elif TV_type == "l1":
        f_TV = np.sum(np.abs(P_1)) + np.sum(np.abs(P_2))
    else:
        raise ValueError(f"Unsupported TV type: {TV_type}")
    return f_TV


def _reg_wavelet_tv(v, W, z_1, z_2, z_3, lamda, beta, L_T, Binv=None,
                    MaxIter=100, TV_type="l1", TV_bound="Dirchlet", tol=1e-6):
    m, n = v.shape
    x_1_old = np.zeros(z_1.shape, dtype=np.complex128)
    x_2_old = np.zeros((m, n), dtype=np.complex128)
    x_3_old = np.zeros((m, n), dtype=np.complex128)
    if Binv is None:
        Binv = lambda x: x
    t_k_1 = 1
    temp_old = np.zeros((m, n), dtype=np.complex128)
    for k in range(MaxIter):
        temp = Binv(lamda * W.H * z_1 + beta * _tv_lforward(z_2, z_3, TV_bound, isComplex=True))
        re = np.linalg.norm(temp_old - temp) / (np.linalg.norm(temp) + 1e-10)
        if k > 0 and (re < tol or k == MaxIter - 1):
            break
        temp_w = temp - v
        temp_old = temp
        grad_1 = W * (lamda * temp_w)
        grad_2, grad_3 = _tv_ltrans(beta * temp_w, m, n, TV_bound)
        temp_1 = z_1 - grad_1 / L_T
        temp_2 = z_2 - grad_2 / L_T
        temp_3 = z_3 - grad_3 / L_T
        x_1_new = temp_1 / np.maximum(np.abs(temp_1), 1)
        if TV_type == "l1":
            x_2_new = temp_2 / np.maximum(np.abs(temp_2), 1)
            x_3_new = temp_3 / np.maximum(np.abs(temp_3), 1)
        elif TV_type == "iso":
            if TV_bound == "Neumann":
                temp_proj = np.abs(np.vstack((temp_2, np.zeros((1, n), dtype=np.complex128)))) ** 2 + np.abs(np.column_stack((temp_3, np.zeros((m, 1), dtype=np.complex128)))) ** 2
                temp_proj = np.sqrt(np.maximum(temp_proj, 1))
                x_2_new = temp_2 / temp_proj[0:m-1, :]
                x_3_new = temp_3 / temp_proj[:, 0:n-1]
            else:
                temp_proj = np.abs(temp_2) ** 2 + np.abs(temp_3) ** 2
                temp_proj = np.sqrt(np.maximum(temp_proj, 1))
                x_2_new = temp_2 / temp_proj
                x_3_new = temp_3 / temp_proj
        t_k = t_k_1
        t_k_1 = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        z_1 = x_1_new + ((t_k - 1) / t_k_1) * (x_1_new - x_1_old)
        z_2 = x_2_new + ((t_k - 1) / t_k_1) * (x_2_new - x_2_old)
        z_3 = x_3_new + ((t_k - 1) / t_k_1) * (x_3_new - x_3_old)
        x_1_old = x_1_new.copy()
        x_2_old = x_2_new.copy()
        x_3_old = x_3_new.copy()
    x = v - Binv(lamda * W.H * x_1_old + beta * _tv_lforward(x_2_old, x_3_old, TV_bound, isComplex=True))
    return x, x_1_old, x_3_old, x_3_old


def QNP_WaveletTV(num_iters, A, b, lamda=0, beta=0, gamma=1.6, a_k=1,
                  Maxsub_Iter=100, W=None, TV_bound="Dirchlet", save=None,
                  SaveIter=True, verbose=True, original=None, TV_type="l1"):
    AHb = A.H(b)
    x = AHb.copy()
    size_m, size_n = x.shape
    m_w, n_w = W.H.ishape
    z_1 = np.zeros((m_w, n_w), dtype=np.complex128)
    z_2 = np.zeros((size_m, size_n), dtype=np.complex128)
    z_3 = np.zeros((size_m, size_n), dtype=np.complex128)
    lst_time, lst_cost, lst_psnr, lst_ssim = [], [], [], []
    if verbose:
        pbar = tqdm(total=num_iters, desc="QNP Wavelet+l1TV", leave=True)
    x_old = x.copy()
    L_W = _power_iter_handle(W.H * W, size_m, size_n, tol=1e-2)
    L_T_HT = ((lamda * L_W) ** 2 + 8 * beta ** 2)
    lst_cost.append(0.5 * np.linalg.norm(A * x - b) ** 2 + lamda * np.sum(np.abs(W * x)) + beta * _obj_tv(x, TV_bound, TV_type))
    lst_time.append(0)
    if original is not None:
        # Upstream only uses these for progress/history; final web metrics are
        # computed independently with a fixed reference data range.
        from skimage.metrics import structural_similarity as _ssim
        def _psnr(o, c):
            mse = np.mean((np.abs(o - c)) ** 2)
            if mse == 0:
                return 100.0
            max_pixel = 1.0 if np.max(np.abs(o)) < 1.01 else 255.0
            return 20 * np.log10(max_pixel / np.sqrt(mse))
        lst_psnr.append(_psnr(np.abs(original), np.abs(x)))
        lst_ssim.append(_ssim(np.abs(original), np.abs(x), data_range=np.abs(x).max() - np.abs(x).min()))
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = A.H * A * x - AHb
        if k == 0:
            temp_grad = x - gr
            x, z_1, z_2, z_3 = _reg_wavelet_tv(temp_grad, W, z_1, z_2, z_3, lamda, beta, L_T_HT, MaxIter=Maxsub_Iter, TV_type=TV_type, TV_bound=TV_bound)
            gr_old = gr.copy()
        else:
            y_k = gr - gr_old
            s_k = x - x_old
            x_old = x.copy()
            gr_old = gr.copy()
            y_k_dot = np.real(np.vdot(y_k, y_k))
            s_k_dot = np.real(np.vdot(s_k, s_k))
            tau_BB = y_k_dot / np.real(np.vdot(s_k, y_k))
            if tau_BB < 0:
                temp_grad = x_old - gr
                L_Binv = 1
                Bx_inv = lambda xx: xx
            else:
                H_0 = gamma * tau_BB
                D = H_0
                temp_1 = y_k - D * s_k
                temp_2 = np.real(np.vdot(temp_1, s_k))
                if np.abs(temp_2) <= 1e-8 * np.sqrt(s_k_dot) * np.linalg.norm(temp_1):
                    Bx_inv = lambda xx: (a_k / D) * xx
                    L_Binv = a_k / D
                else:
                    u = temp_1.reshape((size_m * size_n, 1), order="F") / np.sqrt(np.abs(temp_2))
                    u_inv = u / D
                    u_sign = np.array((np.sign(temp_2),))
                    u_u_dot = np.vdot(u_inv, u)
                    u_sign_scale = a_k * u_sign / (1 + u_sign * u_u_dot)
                    Bx_inv = lambda xx: ((a_k / D) * xx - np.reshape((u_sign_scale * (u_inv.conj().T @ xx.reshape((size_m * size_n, 1), order="F"))) * u_inv, (size_m, size_n), order="F"))
                    if u_sign > 0:
                        L_Binv = a_k / D
                    else:
                        L_Binv = a_k * np.real(1 / (D - u.conj().T @ u))
                temp_grad = x_old - Bx_inv(gr_old)
            x, z_1, z_2, z_3 = _reg_wavelet_tv(temp_grad, W, z_1, z_2, z_3, lamda, beta, L_Binv * L_T_HT, Binv=Bx_inv, MaxIter=Maxsub_Iter, TV_type=TV_type, TV_bound=TV_bound)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        lst_cost.append(0.5 * np.linalg.norm(A * x - b) ** 2 + lamda * np.sum(np.abs(W * x)) + beta * _obj_tv(x, TV_bound, TV_type))
        if original is not None:
            lst_psnr.append(_psnr(np.abs(original), np.abs(x)))
            lst_ssim.append(_ssim(np.abs(original), np.abs(x), data_range=np.abs(x).max() - np.abs(x).min()))
        if save is not None:
            np.save(f"{save}/time.npy", np.cumsum(lst_time))
            np.save(f"{save}/cost.npy", lst_cost)
            if original is not None:
                np.save(f"{save}/psnr.npy", lst_psnr)
                np.save(f"{save}/ssim.npy", lst_ssim)
            if SaveIter:
                np.save(f"{save}/iter_{k:03d}.npy", x)
        if verbose:
            pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
        pbar.close()
        print(np.cumsum(lst_time)[-1])
        print(lst_cost[-1])
    return x, lst_cost, lst_psnr, lst_ssim, np.cumsum(lst_time)



def centered_mask_1d(
    native_lines: np.ndarray,
    acceleration: int,
    acs_lines: int,
) -> np.ndarray:
    """Create deterministic phase-encode sampling on lines that exist natively."""
    native_lines = np.asarray(native_lines, dtype=bool)
    n = native_lines.size

    if acceleration <= 1:
        return native_lines.copy()

    acceleration = int(max(acceleration, 1))
    acs_lines = int(max(acs_lines, 1))
    acs_lines = min(acs_lines, n)

    center = n // 2
    mask = np.zeros(n, dtype=bool)

    start = max(0, center - acs_lines // 2)
    stop = min(n, start + acs_lines)
    mask[start:stop] = True

    idx = np.arange(n)
    mask |= ((idx - center) % acceleration) == 0

    # Never invent samples that are zero/unavailable in the source acquisition.
    mask &= native_lines

    return mask


def detect_native_lines(kspace: np.ndarray, phase_axis: int) -> np.ndarray:
    """Detect phase-encode lines carrying measured signal."""
    if phase_axis not in (-1, -2):
        raise ValueError("phase_axis must be -1 or -2.")

    spatial_axis = phase_axis % kspace.ndim
    reduce_axes = tuple(
        axis
        for axis in range(kspace.ndim)
        if axis != spatial_axis
    )

    energy = np.sum(
        np.abs(kspace) ** 2,
        axis=reduce_axes,
    )

    peak = float(np.max(energy))
    if peak <= 0:
        raise ValueError("Input k-space appears to contain no signal.")

    return energy > (peak * 1e-12)


def make_2d_mask(
    height: int,
    width: int,
    line_mask: np.ndarray,
    phase_axis: int,
) -> np.ndarray:
    if phase_axis == -1:
        if line_mask.size != width:
            raise ValueError("Line mask does not match width.")
        return np.broadcast_to(
            line_mask[None, :],
            (height, width),
        ).astype(np.float64)

    if line_mask.size != height:
        raise ValueError("Line mask does not match height.")

    return np.broadcast_to(
        line_mask[:, None],
        (height, width),
    ).astype(np.float64)


def sense_combine_full_kspace(
    full_kspace: np.ndarray,
    mps: np.ndarray,
    scale: float,
) -> np.ndarray:
    coil_images = sp.ifft(
        full_kspace / scale,
        axes=(-2, -1),
    )

    denom = (
        np.sum(
            np.abs(mps) ** 2,
            axis=0,
        )
        + EPS
    )

    return (
        np.sum(
            np.conj(mps) * coil_images,
            axis=0,
        )
        / denom
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Path to a multicoil fastMRI .h5 file.",
    )

    ap.add_argument(
        "--slice",
        type=int,
        default=8,
        help="Slice index.",
    )

    ap.add_argument(
        "--acceleration",
        type=int,
        default=1,
        help=(
            "Retrospective Cartesian acceleration. "
            "Use 1 to reconstruct all native acquired k-space lines."
        ),
    )

    ap.add_argument(
        "--acs_lines",
        type=int,
        default=24,
        help="Number of fully sampled central calibration lines for R>1.",
    )

    ap.add_argument(
        "--phase_axis",
        type=int,
        choices=[-1, -2],
        default=-1,
        help=(
            "Phase-encode axis inside one coil k-space image. "
            "For the supplied fastMRI brain sample, -1 is appropriate."
        ),
    )

    ap.add_argument(
        "--calib_width",
        type=int,
        default=24,
        help="ESPIRiT calibration width.",
    )

    ap.add_argument(
        "--iters",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--inner_iters",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--wavelet_lambda",
        type=float,
        default=1e-4,
        help="QNP wavelet regularization weight.",
    )

    ap.add_argument(
        "--tv_beta",
        type=float,
        default=5e-4,
        help="QNP TV regularization weight.",
    )

    ap.add_argument(
        "--gamma",
        type=float,
        default=1.7,
    )

    ap.add_argument(
        "--output",
        default="results/native_fastmri_qnp",
    )

    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    input_path = Path(args.input)

    with h5py.File(input_path, "r") as hf:
        if "kspace" not in hf:
            raise KeyError(
                "The HDF5 file does not contain a 'kspace' dataset."
            )

        all_kspace = hf["kspace"]

        if all_kspace.ndim != 4:
            raise ValueError(
                "Expected fastMRI multicoil kspace with shape "
                "(slices, coils, height, width), "
                f"got {all_kspace.shape}"
            )

        n_slices = all_kspace.shape[0]

        if args.slice < 0 or args.slice >= n_slices:
            raise IndexError(
                f"Slice {args.slice} is out of range for {n_slices} slices."
            )

        full_kspace = np.asarray(
            all_kspace[args.slice]
        ).astype(np.complex128)

        rss_reference = None
        if "reconstruction_rss" in hf:
            rss_reference = np.asarray(
                hf["reconstruction_rss"][args.slice]
            ).astype(np.float64)

        attrs = {
            str(k): (
                v.decode("utf-8", errors="ignore")
                if isinstance(v, bytes)
                else str(v)
            )
            for k, v in hf.attrs.items()
        }

    if full_kspace.ndim != 3:
        raise ValueError(
            f"Expected one slice as (coils, height, width), got {full_kspace.shape}"
        )

    coils, height, width = full_kspace.shape

    print("=== NATIVE fastMRI CARTESIAN QNP ===")
    print(f"Input H5                     : {input_path}")
    print(f"Slice                        : {args.slice}")
    print(f"Native k-space               : {full_kspace.shape}")
    print(f"Coils                        : {coils}")
    print(f"Image grid                   : {(height, width)}")
    print(f"Phase axis                   : {args.phase_axis}")
    print(f"Requested acceleration       : {args.acceleration}")
    print("Extra synthetic noise         : NONE")

    native_lines = detect_native_lines(
        full_kspace,
        args.phase_axis,
    )

    line_mask = centered_mask_1d(
        native_lines,
        args.acceleration,
        args.acs_lines,
    )

    sampling_mask = make_2d_mask(
        height,
        width,
        line_mask,
        args.phase_axis,
    )

    acquired_native = int(
        np.count_nonzero(native_lines)
    )
    acquired_used = int(
        np.count_nonzero(line_mask)
    )

    effective_acceleration = (
        acquired_native
        / max(acquired_used, 1)
    )

    print(
        f"Native acquired PE lines     : {acquired_native}/{native_lines.size}"
    )
    print(
        f"Lines used by QNP            : {acquired_used}/{native_lines.size}"
    )
    print(
        f"Effective acceleration       : {effective_acceleration:.3f}x"
    )

    masked_kspace = (
        full_kspace
        * sampling_mask[None, :, :]
    )

    calib_width = int(
        min(
            args.calib_width,
            height,
            width,
        )
    )

    print(
        f"ESPIRiT calibration width    : {calib_width}"
    )

    # Estimate sensitivity maps only from the measurements available to QNP.
    mps = mr.app.EspiritCalib(
        masked_kspace,
        calib_width=calib_width,
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        max_iter=100,
        show_pbar=True,
    ).run()

    mps = np.asarray(
        mps
    ).astype(np.complex128)

    if mps.shape != full_kspace.shape:
        raise ValueError(
            f"ESPIRiT returned {mps.shape}, expected {full_kspace.shape}"
        )

    # Cartesian SENSE model: P F S
    A0 = mr.linop.Sense(
        mps,
        coord=None,
        weights=sampling_mask,
    )

    b0 = masked_kspace.astype(
        np.complex128
    )

    # Normalize image scale before applying QNP's operator-Lipschitz scaling.
    adjoint_initial = A0.H * b0

    scale = float(
        np.percentile(
            np.abs(adjoint_initial),
            99.5,
        )
    )

    if not np.isfinite(scale) or scale <= EPS:
        scale = float(
            np.max(
                np.abs(adjoint_initial)
            )
        )

    if not np.isfinite(scale) or scale <= EPS:
        raise ValueError(
            "Could not determine a stable data normalization scale."
        )

    b0 = b0 / scale

    print(
        f"Data normalization scale     : {scale:.8g}"
    )

    LL = float(
        np.real(
            sp.app.MaxEig(
                A0.H * A0,
                dtype=np.complex128,
            ).run()
        )
    )

    A = np.sqrt(
        1.0 / LL
    ) * A0

    b = np.sqrt(
        1.0 / LL
    ) * b0

    print(
        f"Operator LL                  : {LL:.8g}"
    )

    # Reference is created from the full native k-space only for evaluation.
    # It is not used to construct the undersampled measurement or sensitivity
    # maps when acceleration > 1.
    gt_complex = sense_combine_full_kspace(
        full_kspace,
        mps,
        scale,
    )

    W = sp.linop.Wavelet(
        A.ishape
    )

    lamda = float(
        args.wavelet_lambda
    )
    beta = float(
        args.tv_beta
    )

    gamma = float(
        args.gamma
    )

    print(
        f"Wavelet lambda               : {lamda:.8g}"
    )
    print(
        f"TV beta                      : {beta:.8g}"
    )
    print(
        f"QNP iterations               : {args.iters}"
    )
    print(
        f"QNP inner iterations         : {args.inner_iters}"
    )

    out = (
        root
        / args.output
        / "cartesian"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        out / "measurement.npy",
        b,
    )

    np.save(
        out / "measurement_pre_LL.npy",
        b0,
    )

    np.save(
        out / "mps.npy",
        mps,
    )

    np.save(
        out / "sampling_mask.npy",
        sampling_mask,
    )

    np.save(
        out / "GT_complex.npy",
        gt_complex,
    )

    np.save(
        out / "GT_magnitude.npy",
        np.abs(gt_complex).astype(np.float32),
    )

    if rss_reference is not None:
        np.save(
            out / "fastmri_reconstruction_rss.npy",
            rss_reference.astype(np.float32),
        )

    save_dir = (
        out
        / "QNPDual_TV_Wavelet"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    t0 = time.perf_counter()

    x, cost, psnr_set, ssim_set, cpu = QNP_WaveletTV(
        args.iters,
        A,
        b,
        lamda=lamda,
        beta=beta,
        gamma=gamma,
        a_k=1,
        Maxsub_Iter=args.inner_iters,
        W=W,
        save=str(save_dir),
        SaveIter=True,
        verbose=True,
        original=gt_complex,
        TV_bound="Dirchlet",
        TV_type="l1",
    )

    wall = time.perf_counter() - t0

    np.save(
        out / "QNP_reconstruction_complex.npy",
        x,
    )

    np.save(
        out / "QNP_reconstruction_magnitude.npy",
        np.abs(x).astype(np.float32),
    )

    operator_meta = {
        "operator_type":
            "native_cartesian_sense",

        "source_h5":
            str(input_path),

        "slice_index":
            int(args.slice),

        "native_kspace_shape":
            list(full_kspace.shape),

        "image_shape":
            list(x.shape),

        "phase_axis":
            int(args.phase_axis),

        "requested_acceleration":
            int(args.acceleration),

        "effective_acceleration":
            float(effective_acceleration),

        "acs_lines":
            int(args.acs_lines),

        "native_acquired_phase_lines":
            int(acquired_native),

        "used_phase_lines":
            int(acquired_used),

        "LL":
            float(LL),

        "data_scale":
            float(scale),

        "noise_level":
            None,

        "synthetic_noise_added":
            False,

        "mps_file":
            "mps.npy",

        "sampling_mask_file":
            "sampling_mask.npy",

        "measurement_file":
            "measurement.npy",

        "gt_complex_file":
            "GT_complex.npy",

        "qnp_source":
            "vendored Apache-2.0 CQNPCS_MRIReco::QNP_WaveletTV",

        "qnp_core_modified":
            False,

        "sensitivity_estimation":
            "sigpy.mri.app.EspiritCalib",

        "wavelet_lambda":
            lamda,

        "beta_tv":
            beta,

        "gamma":
            gamma,

        "h5_attributes":
            attrs,
    }

    (
        out
        / "operator_meta.json"
    ).write_text(
        json.dumps(
            operator_meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = {
        "method":
            "original QNP_WaveletTV with native fastMRI Cartesian SENSE forward model",

        "source_h5":
            str(input_path),

        "slice_index":
            int(args.slice),

        "acceleration":
            int(args.acceleration),

        "effective_acceleration":
            float(effective_acceleration),

        "iterations":
            int(args.iters),

        "inner_iterations":
            int(args.inner_iters),

        "lambda_wavelet":
            lamda,

        "beta_tv":
            beta,

        "gamma":
            gamma,

        "operator_LL":
            float(LL),

        "final_psnr_db":
            (
                float(psnr_set[-1])
                if len(psnr_set)
                else None
            ),

        "final_ssim":
            (
                float(ssim_set[-1])
                if len(ssim_set)
                else None
            ),

        "qnp_cpu_time_seconds":
            (
                float(cpu[-1])
                if len(cpu)
                else None
            ),

        "wall_time_seconds":
            float(wall),

        "synthetic_noise_added":
            False,

        "qnp_core_modified":
            False,
    }

    (
        out
        / "metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== NATIVE CARTESIAN QNP RESULT ==="
    )

    if result["final_psnr_db"] is not None:
        print(
            f"PSNR : {result['final_psnr_db']:.4f} dB"
        )

    if result["final_ssim"] is not None:
        print(
            f"SSIM : {result['final_ssim']:.6f}"
        )

    if result["qnp_cpu_time_seconds"] is not None:
        print(
            f"QNP CPU time : {result['qnp_cpu_time_seconds']:.4f} s"
        )

    print(
        f"Wall time    : {wall:.4f} s"
    )

    print(
        f"Saved to     : {out.resolve()}"
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "This run uses native Cartesian fastMRI k-space directly."
    )
    print(
        "It is a different acquisition/model condition from the original "
        "simulated radial/spiral QNP demo."
    )


if __name__ == "__main__":
    main()
