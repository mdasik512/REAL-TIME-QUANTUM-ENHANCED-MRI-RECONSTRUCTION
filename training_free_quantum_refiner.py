"""
training_free_quantum_refiner_V11.py

FROZEN TRAINING-FREE QUANTUM-GATED TV ENHANCEMENT
=================================================

This version freezes the best quantum-gated candidate found on the development
case by quantum_psnr_capacity_diagnostic.py:

    TV weight = 0.008
    blend     = 0.40

The quantum circuit does NOT learn parameters.  It produces a spatial gating
map from six local image/physics features.  The frozen final candidate is

    x_q = x_qnp + 0.40 * qgate(x) * (TV_0.008(x_qnp) - x_qnp)

Ground truth, if supplied, is used ONLY after inference for evaluation.
It is never used to choose parameters, gates, or the final image.

IMPORTANT:
The same development experiment showed that the matched classical TV blend
outperformed the quantum-gated candidate on that one validation case.  This
file therefore also saves the matched classical ablation for honest comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pennylane as qml
import sigpy as sp
import sigpy.mri as mr
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle

EPS = 1e-12
N_QUBITS = 6

# Frozen from the development/validation diagnostic.
FROZEN_TV_WEIGHT = 0.008
FROZEN_BLEND = 0.40


def nmse(gt, pred):
    return float(
        np.sum(np.abs(gt - pred) ** 2)
        / (np.sum(np.abs(gt) ** 2) + EPS)
    )


def metrics(gt, pred):
    g = np.abs(gt)
    p = np.abs(pred)
    dr = float(g.max() - g.min())
    if dr <= 0:
        dr = 1.0

    return {
        "PSNR_dB": float(
            peak_signal_noise_ratio(g, p, data_range=dr)
        ),
        "SSIM": float(
            structural_similarity(g, p, data_range=dr)
        ),
        "NMSE_complex": nmse(gt, pred),
    }


def complex_gaussian(x, sigma):
    return (
        gaussian_filter(np.real(x), sigma=sigma, mode="reflect")
        + 1j
        * gaussian_filter(np.imag(x), sigma=sigma, mode="reflect")
    )


def complex_coherence(a, b):
    return float(
        np.clip(
            abs(np.vdot(a, b))
            / (np.linalg.norm(a) * np.linalg.norm(b) + EPS),
            0.0,
            1.0,
        )
    )


def grad_activity(x):
    m = np.abs(x)
    gx = np.diff(m, axis=1, append=m[:, -1:])
    gy = np.diff(m, axis=0, append=m[-1:, :])
    return float(
        np.mean(np.sqrt(gx * gx + gy * gy))
        / (np.mean(m) + EPS)
    )


def patch_starts(n, patch, stride):
    if n <= patch:
        return [0]

    out = list(range(0, n - patch + 1, stride))
    if out[-1] != n - patch:
        out.append(n - patch)
    return out


def patch_weight(patch):
    w = np.hanning(patch)
    w = 0.12 + 0.88 * w
    return np.outer(w, w).astype(np.float64)


def local_features(xp, rp):
    mag = np.abs(xp)

    # 0: normalized physics residual
    f0 = float(
        np.linalg.norm(rp)
        / (np.linalg.norm(xp) + EPS)
    )

    # 1: image gradient activity
    f1 = grad_activity(xp)

    # 2: local coefficient of variation
    f2 = float(
        np.std(mag)
        / (np.mean(mag) + EPS)
    )

    # 3-4: residual frequency-energy fractions
    rf = rp - complex_gaussian(rp, 0.70)
    rm = (
        complex_gaussian(rp, 0.70)
        - complex_gaussian(rp, 1.60)
    )
    rl = complex_gaussian(rp, 1.60)

    ef = float(np.mean(np.abs(rf) ** 2))
    em = float(np.mean(np.abs(rm) ** 2))
    el = float(np.mean(np.abs(rl) ** 2))

    et = ef + em + el + EPS
    f3 = ef / et
    f4 = em / et

    # 5: image-high-frequency / residual-high-frequency coherence
    xh = xp - complex_gaussian(xp, 1.0)
    rh = rp - complex_gaussian(rp, 1.0)
    f5 = complex_coherence(xh, rh)

    return np.nan_to_num(
        np.asarray(
            [f0, f1, f2, f3, f4, f5],
            dtype=np.float64,
        ),
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )


def robust_normalize(F):
    lo = np.percentile(F, 5.0, axis=0)
    hi = np.percentile(F, 95.0, axis=0)

    out = np.empty_like(F, dtype=np.float64)

    for j in range(F.shape[1]):
        span = hi[j] - lo[j]

        if span < 1e-10:
            out[:, j] = 0.5
        else:
            out[:, j] = np.clip(
                (F[:, j] - lo[j]) / span,
                0.0,
                1.0,
            )

    return out


def build_fixed_qnode(layers):
    dev = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    layers = max(1, int(layers))

    @qml.qnode(dev)
    def circuit(features):
        for layer in range(layers):
            shift = layer % N_QUBITS

            for wire in range(N_QUBITS):
                a = features[
                    (wire + shift) % N_QUBITS
                ]
                b = features[
                    (
                        N_QUBITS
                        - 1
                        - wire
                        + shift
                    )
                    % N_QUBITS
                ]

                qml.RY(a, wires=wire)
                qml.RZ(
                    0.55 * b,
                    wires=wire,
                )

            for wire in range(N_QUBITS - 1):
                qml.CNOT(
                    wires=[wire, wire + 1]
                )

            qml.CNOT(
                wires=[
                    N_QUBITS - 1,
                    0,
                ]
            )

        return [
            qml.expval(
                qml.PauliZ(w)
            )
            for w in range(N_QUBITS)
        ]

    return circuit


def build_quantum_gate(
    x,
    residual,
    patch,
    stride,
    circuit,
):
    h, w = x.shape

    ys = patch_starts(
        h,
        patch,
        stride,
    )
    xs = patch_starts(
        w,
        patch,
        stride,
    )

    win = patch_weight(patch)

    features = []
    records = []
    coherences = []

    for y in ys:
        for xx in xs:
            xp = x[
                y:y + patch,
                xx:xx + patch,
            ]
            rp = residual[
                y:y + patch,
                xx:xx + patch,
            ]

            features.append(
                local_features(xp, rp)
            )

            coherences.append(
                complex_coherence(
                    xp
                    - complex_gaussian(
                        xp,
                        1.0,
                    ),
                    rp
                    - complex_gaussian(
                        rp,
                        1.0,
                    ),
                )
            )

            records.append((y, xx))

    F = robust_normalize(
        np.stack(
            features,
            axis=0,
        )
    )

    gate_acc = np.zeros(
        (h, w),
        dtype=np.float64,
    )
    coh_acc = np.zeros(
        (h, w),
        dtype=np.float64,
    )
    wt = np.zeros(
        (h, w),
        dtype=np.float64,
    )

    q_rows = []

    for i, (y, xx) in enumerate(records):
        q = np.asarray(
            circuit(
                np.pi * F[i]
            ),
            dtype=np.float64,
        )

        u = 0.5 * (
            np.clip(
                q,
                -1.0,
                1.0,
            )
            + 1.0
        )

        # Same fixed quantum-confidence mapping
        # used by the capacity diagnostic.
        q_conf = float(
            np.clip(
                0.22 * u[0]
                + 0.18 * u[1]
                + 0.18 * u[2]
                + 0.16 * u[3]
                + 0.14 * u[4]
                + 0.12 * u[5],
                0.0,
                1.0,
            )
        )

        coh = float(
            coherences[i]
        )

        conf = np.clip(
            q_conf
            * (
                0.35
                + 0.65 * coh
            ),
            0.0,
            1.0,
        )

        sl = np.s_[
            y:y + patch,
            xx:xx + patch,
        ]

        gate_acc[sl] += (
            conf * win
        )
        coh_acc[sl] += (
            coh * win
        )
        wt[sl] += win

        q_rows.append(q)

    gate = gate_acc / (wt + EPS)
    coherence = (
        coh_acc / (wt + EPS)
    )

    q_rows = np.stack(
        q_rows,
        axis=0,
    )

    return (
        gate,
        coherence,
        q_rows,
    )


def normalize_gate(qgate):
    g5, g95 = np.percentile(
        qgate,
        [5.0, 95.0],
    )

    gate = np.clip(
        (qgate - g5)
        / (g95 - g5 + EPS),
        0.0,
        1.0,
    )

    # Same bounded dynamic range as
    # the development diagnostic.
    return (
        0.10
        + 0.90 * gate
    )


def tv_denoise_complex(
    x,
    weight,
):
    r = denoise_tv_chambolle(
        np.real(x),
        weight=float(weight),
        max_num_iter=25,
        channel_axis=None,
    )

    i = denoise_tv_chambolle(
        np.imag(x),
        weight=float(weight),
        max_num_iter=25,
        channel_axis=None,
    )

    return r + 1j * i


def build_native_operator(
    meta_path,
    measurement,
):
    meta_path = Path(
        meta_path
    ).resolve()

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        meta.get("operator_type")
        != "native_cartesian_sense"
    ):
        raise ValueError(
            "The quantum stage currently supports "
            "native Cartesian SENSE only."
        )

    base = meta_path.parent

    mps = np.load(
        base
        / meta.get(
            "mps_file",
            "mps.npy",
        )
    ).astype(np.complex128)

    mask = np.load(
        base
        / meta.get(
            "sampling_mask_file",
            "sampling_mask.npy",
        )
    ).astype(np.float64)

    A0 = mr.linop.Sense(
        mps,
        coord=None,
        weights=mask,
    )

    LL = meta.get(
        "LL",
        None,
    )

    if LL is None:
        print(
            "Computing MaxEig..."
        )

        LL = float(
            np.real(
                sp.app.MaxEig(
                    A0.H * A0,
                    dtype=np.complex128,
                ).run()
            )
        )

    LL = float(LL)

    A = np.sqrt(
        1.0 / LL
    ) * A0

    b = np.asarray(
        measurement,
        dtype=np.complex128,
    )

    if tuple(b.shape) != tuple(A.oshape):
        raise ValueError(
            f"Measurement shape {b.shape} "
            f"does not match operator "
            f"{A.oshape}"
        )

    return A, LL


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--qnp",
        required=True,
    )
    ap.add_argument(
        "--measurement",
        required=True,
    )
    ap.add_argument(
        "--operator_meta",
        required=True,
    )

    # Compatibility with existing pipeline.
    ap.add_argument(
        "--trajectory",
        default="cartesian",
    )
    ap.add_argument(
        "--upstream",
        default="upstream",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--q_layers",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--patch_size",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=16,
    )

    # Optional only for final evaluation.
    ap.add_argument(
        "--gt",
        default=None,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    if args.trajectory != "cartesian":
        raise ValueError(
            "The fixed quantum design is for "
            "native Cartesian experiments."
        )

    qnp = np.load(
        args.qnp
    ).astype(np.complex128)

    b = np.load(
        args.measurement
    ).astype(np.complex128)

    A, LL = build_native_operator(
        args.operator_meta,
        b,
    )

    # Physics residual is used ONLY to
    # construct local quantum features.
    residual = A.H * (
        b - A * qnp
    )

    circuit = build_fixed_qnode(
        args.q_layers
    )

    qgate_raw, coherence, qrows = (
        build_quantum_gate(
            qnp,
            residual,
            args.patch_size,
            args.stride,
            circuit,
        )
    )

    qgate = normalize_gate(
        qgate_raw
    )

    tv_base = tv_denoise_complex(
        qnp,
        FROZEN_TV_WEIGHT,
    )

    delta = tv_base - qnp

    # FROZEN quantum-gated enhancement.
    quantum_final = (
        qnp
        + FROZEN_BLEND
        * qgate
        * delta
    )

    # Matched classical ablation:
    # same TV base, same global blend,
    # but no quantum gate.
    classical_ablation = (
        qnp
        + FROZEN_BLEND
        * delta
    )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        out
        / "quantum_refined_final_complex.npy",
        quantum_final,
    )

    np.save(
        out
        / "classical_tv_ablation_complex.npy",
        classical_ablation,
    )

    np.save(
        out
        / "quantum_gate.npy",
        qgate,
    )

    np.save(
        out
        / "coherence_map.npy",
        coherence,
    )

    np.save(
        out
        / "quantum_outputs.npy",
        qrows,
    )

    report = {
        "version": "V11",
        "method":
            "frozen training-free "
            "quantum-gated TV enhancement",
        "trainable_quantum_parameters": 0,
        "dataset_training": False,
        "gt_used_for_selection": False,
        "quantum_device":
            "PennyLane default.qubit simulator",
        "q_layers": int(
            args.q_layers
        ),
        "patch_size": int(
            args.patch_size
        ),
        "stride": int(
            args.stride
        ),
        "operator_LL": float(LL),
        "frozen_tv_weight":
            FROZEN_TV_WEIGHT,
        "frozen_blend":
            FROZEN_BLEND,
        "mean_quantum_gate": float(
            np.mean(qgate)
        ),
        "gate_min": float(
            np.min(qgate)
        ),
        "gate_max": float(
            np.max(qgate)
        ),
        "mean_quantum_outputs":
            np.mean(
                qrows,
                axis=0,
            ).tolist(),
        "mean_coherence": float(
            np.mean(coherence)
        ),
        "evaluation": None,
    }

    print(
        "=== FROZEN TRAINING-FREE "
        "QUANTUM-GATED TV ENHANCEMENT ==="
    )
    print(
        "Trainable quantum parameters : 0"
    )
    print(
        "Dataset training             : NONE"
    )
    print(
        "GT used for selection        : NEVER"
    )
    print(
        "Quantum role                 : "
        "spatial TV enhancement gate"
    )
    print(
        "Frozen TV weight             : "
        f"{FROZEN_TV_WEIGHT}"
    )
    print(
        "Frozen blend                 : "
        f"{FROZEN_BLEND}"
    )
    print(
        "Qubits                       : "
        f"{N_QUBITS}"
    )
    print(
        "Fixed circuit layers         : "
        f"{args.q_layers}"
    )
    print(
        "Patch / stride               : "
        f"{args.patch_size} / "
        f"{args.stride}"
    )
    print(
        "Operator LL                  : "
        f"{LL:.8f}"
    )
    print(
        "Mean quantum gate            : "
        f"{np.mean(qgate):.6f}"
    )
    print(
        "Gate range                   : "
        f"{np.min(qgate):.4f} - "
        f"{np.max(qgate):.4f}"
    )

    if args.gt is not None:
        gt = np.load(
            args.gt
        ).astype(np.complex128)

        if gt.shape != qnp.shape:
            raise ValueError(
                f"GT {gt.shape} and "
                f"QNP {qnp.shape} differ."
            )

        m_qnp = metrics(
            gt,
            qnp,
        )
        m_quantum = metrics(
            gt,
            quantum_final,
        )
        m_classical = metrics(
            gt,
            classical_ablation,
        )

        report[
            "evaluation"
        ] = {
            "QNP": m_qnp,
            "Quantum_frozen": m_quantum,
            "Classical_matched_ablation":
                m_classical,
            "quantum_minus_qnp_psnr_db":
                float(
                    m_quantum["PSNR_dB"]
                    - m_qnp["PSNR_dB"]
                ),
            "quantum_minus_classical_psnr_db":
                float(
                    m_quantum["PSNR_dB"]
                    - m_classical["PSNR_dB"]
                ),
        }

        print(
            "\nGT supplied ONLY for "
            "post-inference evaluation."
        )
        print(
            json.dumps(
                report["evaluation"],
                indent=2,
            )
        )

    (
        out / "v11_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== DONE ==="
    )
    print(
        "Final output                 : "
        "quantum_refined_final_complex.npy"
    )
    print(
        "Matched classical ablation   : "
        "classical_tv_ablation_complex.npy"
    )
    print(
        "Saved to                     : "
        f"{out.resolve()}"
    )


if __name__ == "__main__":
    main()
