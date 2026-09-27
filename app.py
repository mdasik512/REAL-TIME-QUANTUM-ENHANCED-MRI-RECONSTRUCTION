from __future__ import annotations

"""
QNP + Training-Free Quantum MRI — single-file local web application
==================================================================

This file contains BOTH:
  1) the Flask API/backend controller + metrics/image packaging logic, and
  2) the complete browser frontend (HTML/CSS/JavaScript) inline.

It intentionally has NO React/Vite/npm/CDN dependency. Run one Python server and
open http://127.0.0.1:8080.

Scientific reconstruction remains in the local project files beside this file:
  - run_native_fastmri_qnp.py (includes the vendored Apache-2.0 QNP runtime core)
  - training_free_quantum_refiner.py

The app never downloads reconstruction code at runtime.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from werkzeug.utils import secure_filename


# -----------------------------------------------------------------------------
# Paths / application configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
WEB_JOBS_DIR = PROJECT_ROOT / "results" / "web_jobs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
WEB_JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
RUN_SEMAPHORE = threading.Semaphore(1)
ALLOWED_EXTENSIONS = {".h5", ".hdf5"}
EPS = 1e-12


# -----------------------------------------------------------------------------
# Common result metrics + image generation (integrated from web_utils.py)
# -----------------------------------------------------------------------------

def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_mag = np.abs(gt)
    pred_mag = np.abs(pred)
    data_range = float(gt_mag.max() - gt_mag.min())
    if data_range <= 0:
        data_range = 1.0

    nmse = float(
        np.sum(np.abs(gt - pred) ** 2)
        / (np.sum(np.abs(gt) ** 2) + EPS)
    )

    return {
        "PSNR_dB": float(
            peak_signal_noise_ratio(gt_mag, pred_mag, data_range=data_range)
        ),
        "SSIM": float(
            structural_similarity(gt_mag, pred_mag, data_range=data_range)
        ),
        "NMSE_complex": nmse,
    }


def _save_gray(path: Path, image: np.ndarray):
    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)


def _save_error(path: Path, image: np.ndarray, vmax: float):
    plt.imsave(path, image, cmap="magma", vmin=0.0, vmax=max(vmax, EPS))


def _asset_url(job_id: str, name: str) -> str:
    return f"/api/jobs/{job_id}/assets/{name}"


def generate_web_assets(
    cartesian_dir: Path,
    quantum_dir: Path,
    web_dir: Path,
    run_parameters: dict,
    input_info: dict,
) -> dict:
    web_dir.mkdir(parents=True, exist_ok=True)

    gt = np.load(cartesian_dir / "GT_complex.npy").astype(np.complex128)
    qnp = np.load(cartesian_dir / "QNP_reconstruction_complex.npy").astype(np.complex128)
    quantum = np.load(
        quantum_dir / "quantum_refined_final_complex.npy"
    ).astype(np.complex128)

    classical_path = quantum_dir / "classical_tv_ablation_complex.npy"
    classical = (
        np.load(classical_path).astype(np.complex128)
        if classical_path.exists()
        else None
    )

    if not (gt.shape == qnp.shape == quantum.shape):
        raise ValueError(
            f"Image shape mismatch: GT={gt.shape}, QNP={qnp.shape}, Quantum={quantum.shape}"
        )

    scale = float(np.percentile(np.abs(gt), 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(gt)) + EPS)

    gt_disp = np.clip(np.abs(gt) / scale, 0, 1)
    qnp_disp = np.clip(np.abs(qnp) / scale, 0, 1)
    quantum_disp = np.clip(np.abs(quantum) / scale, 0, 1)
    classical_disp = (
        np.clip(np.abs(classical) / scale, 0, 1)
        if classical is not None
        else None
    )

    qnp_error = np.abs(gt_disp - qnp_disp)
    quantum_error = np.abs(gt_disp - quantum_disp)
    error_vmax = float(
        max(
            np.percentile(qnp_error, 99.5),
            np.percentile(quantum_error, 99.5),
            EPS,
        )
    )

    # Exact stable filenames used by the frontend.
    _save_gray(web_dir / "ground_truth.png", gt_disp)
    _save_gray(web_dir / "qnp.png", qnp_disp)
    _save_gray(web_dir / "quantum.png", quantum_disp)
    if classical_disp is not None:
        _save_gray(web_dir / "classical.png", classical_disp)
    _save_error(web_dir / "qnp_error.png", qnp_error, error_vmax)
    _save_error(web_dir / "quantum_error.png", quantum_error, error_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (axes[0, 0], gt_disp, "Reference / Ground Truth", "gray", 1.0),
        (axes[0, 1], qnp_disp, "QNP", "gray", 1.0),
        (axes[0, 2], quantum_disp, "QNP + Quantum", "gray", 1.0),
        (axes[1, 0], qnp_error, "QNP Error", "magma", error_vmax),
        (axes[1, 1], quantum_error, "Quantum Error", "magma", error_vmax),
    ]
    for ax, im, title, cmap, vmax in panels:
        ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
        ax.set_title(title)
        ax.axis("off")

    if classical_disp is not None:
        axes[1, 2].imshow(
            classical_disp, cmap="gray", vmin=0, vmax=1, interpolation="none"
        )
        axes[1, 2].set_title("Matched Classical TV")
        axes[1, 2].axis("off")
    else:
        axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(web_dir / "comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "QNP": _metrics(gt, qnp),
        "Quantum": _metrics(gt, quantum),
    }
    if classical is not None:
        metrics["Classical"] = _metrics(gt, classical)

    metrics["gains"] = {
        "quantum_minus_qnp_psnr_db": float(
            metrics["Quantum"]["PSNR_dB"] - metrics["QNP"]["PSNR_dB"]
        ),
        "quantum_minus_qnp_ssim": float(
            metrics["Quantum"]["SSIM"] - metrics["QNP"]["SSIM"]
        ),
        "quantum_minus_qnp_nmse": float(
            metrics["Quantum"]["NMSE_complex"] - metrics["QNP"]["NMSE_complex"]
        ),
    }
    if "Classical" in metrics:
        metrics["gains"]["quantum_minus_classical_psnr_db"] = float(
            metrics["Quantum"]["PSNR_dB"] - metrics["Classical"]["PSNR_dB"]
        )

    report_path = quantum_dir / "v11_report.json"
    v11_report = None
    if report_path.exists():
        try:
            v11_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    job_id = web_dir.parent.name
    images = {
        "ground_truth": _asset_url(job_id, "ground_truth.png"),
        "qnp": _asset_url(job_id, "qnp.png"),
        "quantum": _asset_url(job_id, "quantum.png"),
        "qnp_error": _asset_url(job_id, "qnp_error.png"),
        "quantum_error": _asset_url(job_id, "quantum_error.png"),
        "comparison": _asset_url(job_id, "comparison.png"),
    }
    if classical_disp is not None:
        images["classical"] = _asset_url(job_id, "classical.png")

    summary = {
        "metrics": metrics,
        "images": images,
        "display_normalization": {
            "rule": "All magnitude images use the same GT/reference 99.5th percentile scale.",
            "scale": scale,
            "error_vmax_99_5_percentile": error_vmax,
        },
        "run_parameters": run_parameters,
        "input_info": input_info,
        "v11_report": v11_report,
        "research_note": (
            "Ground truth/reference is used only for post-inference evaluation. "
            "The quantum parameters are fixed; the matched classical TV output "
            "is retained as an ablation/control."
        ),
    }

    (web_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def make_result_zip(
    zip_path: Path,
    cartesian_dir: Path,
    quantum_dir: Path,
    web_dir: Path,
    job_status_path: Path,
):
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = [
        (web_dir / "ground_truth.png", "ground_truth.png"),
        (web_dir / "qnp.png", "qnp.png"),
        (web_dir / "quantum.png", "quantum.png"),
        (web_dir / "classical.png", "classical.png"),
        (web_dir / "qnp_error.png", "qnp_error.png"),
        (web_dir / "quantum_error.png", "quantum_error.png"),
        (web_dir / "comparison.png", "comparison.png"),
        (web_dir / "metrics.json", "metrics.json"),
        (quantum_dir / "v11_report.json", "v11_report.json"),
        (quantum_dir / "quantum_gate.npy", "quantum_gate.npy"),
        (quantum_dir / "quantum_outputs.npy", "quantum_outputs.npy"),
        (quantum_dir / "classical_tv_ablation_complex.npy", "classical_tv_ablation_complex.npy"),
        (quantum_dir / "quantum_refined_final_complex.npy", "quantum_refined_final_complex.npy"),
        (cartesian_dir / "QNP_reconstruction_complex.npy", "QNP_reconstruction_complex.npy"),
        (cartesian_dir / "GT_complex.npy", "GT_complex.npy"),
        (job_status_path, "job_status.json"),
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in wanted:
            if source.exists():
                zf.write(source, arcname=arcname)


# -----------------------------------------------------------------------------
# Job / upload helpers
# -----------------------------------------------------------------------------

def _valid_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", value or ""))


def _upload_dir(file_id: str) -> Path:
    if not _valid_id(file_id):
        raise ValueError("Invalid file id.")
    return UPLOADS_DIR / file_id


def _job_dir(job_id: str) -> Path:
    if not _valid_id(job_id):
        raise ValueError("Invalid job id.")
    return WEB_JOBS_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job_status.json"


STATUS_FILE_LOCK = threading.Lock()


def _write_json_atomic(path: Path, payload: dict):
    """Persist job state without letting a transient Windows file lock kill reconstruction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2)
    last_error = None

    with STATUS_FILE_LOCK:
        for attempt in range(12):
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                tmp.write_text(data, encoding="utf-8")
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

        # The in-memory JOBS state remains authoritative while the server is running.
        # A locked status file should never terminate the scientific reconstruction.
        try:
            path.write_text(data, encoding="utf-8")
        except PermissionError:
            print(
                f"[warning] Could not persist {path.name} because Windows temporarily locked it: {last_error}",
                file=sys.stderr,
                flush=True,
            )


def _set_job(job_id: str, **changes):
    with JOBS_LOCK:
        state = JOBS.setdefault(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "stage": "Queued",
                "progress": 0,
                "logs": [],
            },
        )
        state.update(changes)
        snapshot = json.loads(json.dumps(state))
    _write_json_atomic(_status_path(job_id), snapshot)
    return snapshot


def _append_log(job_id: str, line: str):
    line = line.rstrip()
    if not line:
        return
    with JOBS_LOCK:
        state = JOBS.setdefault(job_id, {"job_id": job_id, "logs": []})
        state.setdefault("logs", []).append(line)
        state["logs"] = state["logs"][-300:]
        snapshot = json.loads(json.dumps(state))
    _write_json_atomic(_status_path(job_id), snapshot)


def _get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        state = JOBS.get(job_id)
        if state is not None:
            return json.loads(json.dumps(state))

    p = _status_path(job_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _get_upload_meta(file_id: str) -> dict:
    d = _upload_dir(file_id)
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Uploaded file was not found.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    file_path = d / meta["stored_name"]
    if not file_path.exists():
        raise FileNotFoundError("Uploaded MRI file is missing.")
    meta["path"] = str(file_path)
    return meta


def _inspect_h5(path: Path, slice_index: int = 0) -> dict:
    with h5py.File(path, "r") as hf:
        if "kspace" not in hf:
            raise ValueError(
                "This H5 file does not contain a 'kspace' dataset. "
                "Raw reconstruction mode expects a fastMRI-style multicoil H5 file."
            )

        ks = hf["kspace"]
        shape = tuple(int(v) for v in ks.shape)

        if ks.ndim == 4:
            slices, coils, readout, pe = shape
            if not 0 <= slice_index < slices:
                raise ValueError(f"Slice must be between 0 and {slices - 1}.")
            sample = np.asarray(ks[slice_index])
        elif ks.ndim == 3:
            slices = 1
            coils, readout, pe = shape
            if slice_index != 0:
                raise ValueError("This H5 contains one k-space volume; use slice 0.")
            sample = np.asarray(ks)
        else:
            raise ValueError(
                f"Unsupported kspace shape {shape}. Expected "
                "[slices, coils, readout, phase] or [coils, readout, phase]."
            )

        acquired = np.any(np.abs(sample) > 0, axis=(0, 1))
        acquired_lines = int(np.count_nonzero(acquired))

        attrs = {}
        for key in ("acquisition", "max", "norm"):
            if key in hf.attrs:
                try:
                    value = hf.attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    elif hasattr(value, "item"):
                        value = value.item()
                    attrs[key] = value
                except Exception:
                    pass

        return {
            "kspace_shape": list(shape),
            "slices": int(slices),
            "coils": int(coils),
            "readout": int(readout),
            "phase_encoding": int(pe),
            "slice": int(slice_index),
            "acquired_pe_lines": acquired_lines,
            "total_pe_lines": int(pe),
            "attrs": attrs,
        }


def _run_command(job_id: str, cmd: list[str], cwd: Path):
    _append_log(job_id, "$ " + subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_log(job_id, line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Command failed with exit code {rc}: {subprocess.list2cmdline(cmd)}"
        )


def _worker(job_id: str, params: dict):
    with RUN_SEMAPHORE:
        total_start = time.perf_counter()
        qnp_seconds = None
        quantum_seconds = None

        try:
            _set_job(job_id, status="running", stage="Validating upload", progress=3)

            upload = _get_upload_meta(params["file_id"])
            input_path = Path(upload["path"]).resolve()
            info = _inspect_h5(input_path, int(params["slice"]))

            job_root = _job_dir(job_id)
            recon_root = job_root / "reconstruction"
            recon_root.mkdir(parents=True, exist_ok=True)

            qnp_script = PROJECT_ROOT / "run_native_fastmri_qnp.py"
            quantum_script = PROJECT_ROOT / "training_free_quantum_refiner.py"

            missing = [
                str(p.name)
                for p in (qnp_script, quantum_script)
                if not p.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing project component(s): " + ", ".join(missing)
                )

            _set_job(
                job_id,
                stage="Running native Cartesian QNP",
                progress=10,
                input_info=info,
            )

            qnp_cmd = [
                sys.executable,
                str(qnp_script),
                "--input",
                str(input_path),
                "--slice",
                str(params["slice"]),
                "--acceleration",
                str(params["acceleration"]),
                "--acs_lines",
                str(params["acs_lines"]),
                "--phase_axis",
                "-1",
                "--iters",
                str(params["qnp_iters"]),
                "--inner_iters",
                str(params["inner_iters"]),
                "--wavelet_lambda",
                str(params["wavelet_lambda"]),
                "--tv_beta",
                str(params["tv_beta"]),
                "--output",
                str(recon_root),
            ]

            qnp_start = time.perf_counter()
            _run_command(job_id, qnp_cmd, PROJECT_ROOT)
            qnp_seconds = time.perf_counter() - qnp_start

            cart = recon_root / "cartesian"
            required_qnp = [
                cart / "QNP_reconstruction_complex.npy",
                cart / "GT_complex.npy",
                cart / "measurement.npy",
                cart / "operator_meta.json",
            ]
            missing_outputs = [p.name for p in required_qnp if not p.exists()]
            if missing_outputs:
                raise FileNotFoundError(
                    "QNP completed but required output(s) are missing: "
                    + ", ".join(missing_outputs)
                )

            _set_job(
                job_id,
                stage="Running training-free quantum enhancement",
                progress=62,
            )

            quantum_out = cart / "training_free_quantum"
            quantum_cmd = [
                sys.executable,
                str(quantum_script),
                "--qnp",
                str(cart / "QNP_reconstruction_complex.npy"),
                "--measurement",
                str(cart / "measurement.npy"),
                "--operator_meta",
                str(cart / "operator_meta.json"),
                "--trajectory",
                "cartesian",
                "--upstream",
                str(PROJECT_ROOT),
                "--steps",
                "30",
                "--q_layers",
                str(params["q_layers"]),
                "--gt",
                str(cart / "GT_complex.npy"),
                "--out",
                str(quantum_out),
            ]

            quantum_start = time.perf_counter()
            _run_command(job_id, quantum_cmd, PROJECT_ROOT)
            quantum_seconds = time.perf_counter() - quantum_start

            _set_job(
                job_id,
                stage="Generating images and common metrics",
                progress=88,
            )

            web_dir = job_root / "web"
            summary = generate_web_assets(
                cartesian_dir=cart,
                quantum_dir=quantum_out,
                web_dir=web_dir,
                run_parameters=params,
                input_info=info,
            )

            # Timing is written into result.json before status becomes completed.
            total_seconds = time.perf_counter() - total_start
            timing = {
                "total_seconds": round(total_seconds, 2),
                "qnp_seconds": round(qnp_seconds, 2) if qnp_seconds is not None else None,
                "quantum_seconds": (
                    round(quantum_seconds, 2) if quantum_seconds is not None else None
                ),
            }

            summary["job_id"] = job_id
            summary["download_url"] = f"/api/jobs/{job_id}/download"
            summary["input_info"] = info
            summary["parameters"] = params
            summary["timing"] = timing
            _write_json_atomic(job_root / "result.json", summary)

            zip_path = job_root / "result_bundle.zip"
            make_result_zip(
                zip_path=zip_path,
                cartesian_dir=cart,
                quantum_dir=quantum_out,
                web_dir=web_dir,
                job_status_path=_status_path(job_id),
            )

            _set_job(
                job_id,
                status="completed",
                stage="Completed",
                progress=100,
                result_url=f"/api/jobs/{job_id}/result",
                download_url=f"/api/jobs/{job_id}/download",
                timing=timing,
            )
        except Exception as exc:
            _append_log(job_id, traceback.format_exc())
            _set_job(
                job_id,
                status="failed",
                stage="Failed",
                error=str(exc),
                progress=100,
            )


# -----------------------------------------------------------------------------
# Single-file frontend
# -----------------------------------------------------------------------------

INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>QNP Spectral MRI Lab</title>
<style>
:root{
  --bg:#050914;--bg2:#080f1e;--panel:#0b1424;--panel2:#0e1a2d;--line:#1d2c43;
  --text:#eaf2ff;--muted:#8fa3bd;--cyan:#59e1ff;--teal:#41e8bc;--violet:#a991ff;
  --green:#61e6a5;--red:#ff7b96;--amber:#ffd27a;--shadow:0 20px 80px rgba(0,0,0,.34);
}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:radial-gradient(1200px 900px at 85% -20%,rgba(86,87,255,.12),transparent 55%),radial-gradient(900px 700px at -10% 30%,rgba(0,225,255,.08),transparent 50%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
button,input,select{font:inherit}.app{min-height:100vh;display:grid;grid-template-columns:260px 1fr}.sidebar{position:sticky;top:0;height:100vh;padding:28px 18px;border-right:1px solid var(--line);background:rgba(5,9,20,.86);backdrop-filter:blur(22px)}
.brand{display:flex;gap:12px;align-items:center;padding:0 8px 26px}.mark{width:38px;height:38px;border-radius:12px;position:relative;background:linear-gradient(145deg,rgba(89,225,255,.22),rgba(169,145,255,.18));border:1px solid rgba(89,225,255,.38);box-shadow:inset 0 0 28px rgba(89,225,255,.08)}.mark:before,.mark:after{content:"";position:absolute;border:1px solid var(--cyan);border-radius:50%;inset:9px}.mark:after{inset:14px;border-color:var(--violet)}
.brand strong{font-size:14px;letter-spacing:.12em}.brand small{display:block;color:var(--muted);font-size:10px;margin-top:4px;letter-spacing:.08em}.nav{display:grid;gap:8px}.nav button{width:100%;text-align:left;border:1px solid transparent;background:transparent;color:var(--muted);padding:12px 13px;border-radius:12px;cursor:pointer;transition:.2s}.nav button:hover,.nav button.active{color:var(--text);background:linear-gradient(90deg,rgba(89,225,255,.10),rgba(169,145,255,.05));border-color:rgba(89,225,255,.16)}
.side-foot{position:absolute;left:18px;right:18px;bottom:22px;padding:14px;border:1px solid var(--line);border-radius:14px;background:#07101e}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:7px}.main{padding:30px;max-width:1640px;width:100%;margin:0 auto}.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:26px}.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);font-weight:800}.title{font-size:clamp(28px,3vw,46px);line-height:1.02;margin:8px 0 10px;letter-spacing:-.04em}.subtitle{color:var(--muted);max-width:760px;line-height:1.6}.job-pill{padding:10px 13px;border:1px solid var(--line);background:rgba(11,20,36,.7);border-radius:999px;color:var(--muted);font-size:12px;white-space:nowrap}.job-pill b{color:var(--text)}
.grid{display:grid;gap:18px}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{position:relative;border:1px solid var(--line);background:linear-gradient(180deg,rgba(14,26,45,.88),rgba(8,15,29,.92));border-radius:18px;box-shadow:var(--shadow);overflow:hidden}.panel::before{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(120deg,rgba(89,225,255,.025),transparent 28%,transparent 75%,rgba(169,145,255,.03))}.pad{padding:20px}.metric .label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:800}.metric .value{font-size:28px;font-weight:800;margin:10px 0 7px;letter-spacing:-.03em}.metric .help{font-size:12px;color:var(--muted);line-height:1.4}
.hero{display:grid;grid-template-columns:1.4fr .9fr;gap:18px}.pipeline{padding:22px}.pipe{display:grid;grid-template-columns:repeat(5,1fr);align-items:center;gap:8px;margin-top:24px}.node{border:1px solid var(--line);background:#07101f;padding:15px 10px;border-radius:14px;text-align:center;font-size:12px;position:relative}.node b{display:block;color:var(--cyan);font-size:10px;letter-spacing:.13em;margin-bottom:6px}.node:not(:last-child):after{content:"";position:absolute;right:-13px;top:50%;width:18px;height:1px;background:linear-gradient(90deg,var(--cyan),transparent)}
.btn{border:1px solid var(--line);background:#0d192a;color:var(--text);padding:11px 15px;border-radius:12px;cursor:pointer;font-weight:750;transition:.18s}.btn:hover{transform:translateY(-1px);border-color:#34516f}.btn.primary{border-color:rgba(89,225,255,.4);background:linear-gradient(135deg,#0e3950,#142b56);box-shadow:0 10px 30px rgba(17,121,168,.18)}.btn.ghost{background:transparent}.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}.actions{display:flex;gap:10px;flex-wrap:wrap}
.drop{border:1px dashed #2d5570;border-radius:18px;padding:34px;text-align:center;background:linear-gradient(180deg,rgba(31,92,118,.08),rgba(10,18,32,.2));cursor:pointer}.drop.drag{border-color:var(--cyan);background:rgba(89,225,255,.08)}.drop h3{margin:8px 0}.drop p{color:var(--muted);font-size:13px}.drop input{display:none}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.field label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:7px}.field input,.field select{width:100%;border:1px solid var(--line);background:#07101e;color:var(--text);border-radius:11px;padding:11px 12px;outline:none}.field input:focus,.field select:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(89,225,255,.08)}details{border:1px solid var(--line);border-radius:14px;padding:14px;background:#07101e}summary{cursor:pointer;font-weight:750}
.progress{height:10px;border-radius:999px;background:#09111e;border:1px solid #16243a;overflow:hidden}.progress>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--teal),var(--violet));transition:width .45s ease}.stage-line{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}.stage-line b{font-size:15px}.stage-line span{color:var(--muted);font-size:12px}.log{margin-top:14px;background:#03070e;border:1px solid #142035;border-radius:14px;padding:14px;max-height:300px;overflow:auto;white-space:pre-wrap;font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;color:#9fb5ce}
.image-card .head{padding:13px 15px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.image-card .head b{font-size:13px}.image-card .head span{font-size:10px;color:var(--muted);letter-spacing:.1em}.image-wrap{aspect-ratio:1/1;background:#000;display:grid;place-items:center;position:relative;overflow:hidden}.image-wrap.wide{aspect-ratio:16/9}.image-wrap img{width:100%;height:100%;object-fit:contain;display:block;cursor:default}.img-error{display:none;position:absolute;inset:0;align-items:center;justify-content:center;text-align:center;padding:20px;color:var(--red);font-size:12px;background:#03060b}.img-error code{display:block;color:#8196af;margin-top:8px;word-break:break-all}.tag{display:inline-flex;gap:7px;align-items:center;padding:6px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted);font-size:11px}.tag.good{color:var(--green);border-color:rgba(97,230,165,.25);background:rgba(97,230,165,.06)}
.table{width:100%;border-collapse:collapse}.table th,.table td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:12px}.table th{color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-size:10px}.table tr{cursor:pointer}.table tbody tr:hover{background:rgba(89,225,255,.035)}.notice{padding:14px 16px;border:1px solid rgba(255,210,122,.25);background:rgba(255,210,122,.05);color:#e6c789;border-radius:14px;font-size:12px;line-height:1.55}.muted{color:var(--muted)}.hidden{display:none!important}.section-title{display:flex;justify-content:space-between;align-items:end;gap:15px;margin:28px 0 14px}.section-title h2{margin:0;font-size:18px}.section-title p{margin:4px 0 0;color:var(--muted);font-size:12px}

/* Observatory visual language: deliberately asymmetric, scientific, and local-only. */
body::before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.14;background-image:linear-gradient(rgba(89,225,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(89,225,255,.035) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 82%)}
.sidebar::after{content:"";position:absolute;right:-1px;top:88px;width:1px;height:220px;background:linear-gradient(180deg,transparent,var(--cyan),var(--violet),transparent);box-shadow:0 0 18px rgba(89,225,255,.28)}
.panel{border-radius:18px 6px 18px 6px}.panel::after{content:"";position:absolute;width:22px;height:22px;right:9px;top:9px;border-top:1px solid rgba(89,225,255,.18);border-right:1px solid rgba(89,225,255,.18);pointer-events:none}
.pipeline{min-height:292px;padding-right:285px}.hero-orbit{position:absolute;right:28px;top:29px;width:220px;height:220px;border-radius:50%;border:1px solid rgba(89,225,255,.18);background:radial-gradient(circle at center,rgba(65,232,188,.12) 0 4%,transparent 5% 28%,rgba(89,225,255,.04) 29% 30%,transparent 31% 100%);box-shadow:inset 0 0 55px rgba(89,225,255,.025),0 0 80px rgba(169,145,255,.05)}
.hero-orbit::before,.hero-orbit::after{content:"";position:absolute;border-radius:50%;inset:25px;border:1px dashed rgba(169,145,255,.34);animation:orbit 18s linear infinite}.hero-orbit::after{inset:52px;border-style:solid;border-color:rgba(65,232,188,.25);animation-duration:11s;animation-direction:reverse}.orbit-dot{position:absolute;width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 18px var(--cyan);left:104px;top:-4px;transform-origin:6px 114px;animation:spinDot 8s linear infinite}.orbit-core{position:absolute;inset:84px;border:1px solid rgba(89,225,255,.5);border-radius:50%;display:grid;place-items:center;font:800 10px ui-monospace,monospace;letter-spacing:.12em;color:var(--cyan);box-shadow:0 0 28px rgba(89,225,255,.15)}
@keyframes orbit{to{transform:rotate(360deg)}}@keyframes spinDot{to{transform:rotate(360deg)}}
.title{position:relative}.title::after{content:"";display:block;width:112px;height:2px;margin-top:13px;background:linear-gradient(90deg,var(--cyan),var(--teal),var(--violet),transparent);box-shadow:0 0 16px rgba(89,225,255,.3)}
.image-card{background:#02050a}.image-card .image-wrap::before,.image-card .image-wrap::after{content:"";position:absolute;z-index:2;pointer-events:none}.image-card .image-wrap::before{inset:14px;border:1px solid rgba(89,225,255,.07);clip-path:polygon(0 0,24px 0,24px 1px,1px 1px,1px 24px,0 24px,0 0,100% 0,100% 24px,99% 24px,99% 1px,calc(100% - 24px) 1px,calc(100% - 24px) 0,100% 0,100% 100%,calc(100% - 24px) 100%,calc(100% - 24px) 99%,99% 99%,99% calc(100% - 24px),100% calc(100% - 24px),100% 100%,0 100%,0 calc(100% - 24px),1px calc(100% - 24px),1px 99%,24px 99%,24px 100%,0 100%)}
.image-card .image-wrap::after{left:50%;top:16px;bottom:16px;width:1px;background:linear-gradient(transparent,rgba(89,225,255,.055),transparent)}
.btn{border-radius:11px 4px 11px 4px}.job-pill{border-radius:999px 6px 999px 6px}.tag{border-radius:999px 5px 999px 5px}
.spectral-strip{display:flex;gap:8px;flex-wrap:wrap;margin-top:15px}.spectral-strip span{font:700 10px ui-monospace,monospace;letter-spacing:.08em;color:#9ab0c8;border:1px solid #1b3048;padding:6px 8px;background:#07101d}.spectral-strip span:nth-child(2){color:var(--teal)}.spectral-strip span:nth-child(3){color:var(--violet)}
@media(max-width:1100px){.pipeline{padding-right:22px}.hero-orbit{display:none}.app{grid-template-columns:82px 1fr}.sidebar{padding:22px 12px}.brand strong,.brand small,.nav button span,.side-foot{display:none}.brand{justify-content:center}.nav button{text-align:center;padding:12px 6px}.g4{grid-template-columns:repeat(2,1fr)}.hero{grid-template-columns:1fr}.pipe{grid-template-columns:1fr}.node:not(:last-child):after{display:none}}
@media(max-width:760px){.main{padding:18px}.g3,.g2,.g4,.fields{grid-template-columns:1fr}.topbar{flex-direction:column}.app{grid-template-columns:1fr}.sidebar{position:relative;height:auto;border-right:none;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}.brand{padding:0}.nav{display:flex;overflow:auto}.nav button{width:auto}.main{padding-top:20px}.pipe{grid-template-columns:1fr}}

/* Spectral Cartography — rare scientific workstation theme */
:root{--bg:#02050a;--bg2:#071019;--panel:#08141d;--panel2:#0b1d27;--line:#1a3441;--text:#eefcff;--muted:#88aab4;--cyan:#7ddcff;--teal:#73ffd9;--violet:#aa8cff;--amber:#ffd479}
.main{max-width:none;padding:28px clamp(20px,2.2vw,44px) 48px}
.sidebar{background:linear-gradient(180deg,rgba(2,5,10,.96),rgba(5,13,18,.94))}
.panel{background:linear-gradient(155deg,rgba(8,20,29,.96),rgba(3,9,14,.98));border-color:#183745;box-shadow:0 24px 90px rgba(0,0,0,.42)}
.panel::before{background:linear-gradient(118deg,rgba(125,220,255,.035),transparent 34%,rgba(115,255,217,.018) 57%,transparent 72%,rgba(170,140,255,.035))}
.title{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;font-weight:850;letter-spacing:-.055em}
.eyebrow,.spectral-strip span,.metric .label,.protocol-stamp{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.slice-only{display:grid;grid-template-columns:minmax(180px,.7fr) minmax(260px,1.3fr);gap:14px;align-items:stretch}
.protocol-stamp{border:1px solid rgba(115,255,217,.22);background:linear-gradient(135deg,rgba(115,255,217,.055),rgba(125,220,255,.025));padding:13px 15px;display:grid;align-content:center;gap:4px;clip-path:polygon(0 0,calc(100% - 13px) 0,100% 13px,100% 100%,13px 100%,0 calc(100% - 13px))}
.protocol-stamp span{font-size:9px;letter-spacing:.15em;color:var(--teal);font-weight:800}.protocol-stamp b{font-size:18px;letter-spacing:-.02em}.protocol-stamp small{color:var(--muted);font-size:10px}
#page-results .section-title{margin-top:34px}
#mainImages{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}
.image-card{min-width:0;background:#000;border-color:#173642}
.image-card .head{background:linear-gradient(90deg,#06131b,#04080d);min-height:48px}
.image-card .head span{color:var(--teal);font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.image-wrap,.image-wrap.wide{aspect-ratio:auto;height:min(68vh,760px);min-height:420px;background:#000;display:flex;align-items:center;justify-content:center;padding:10px;overflow:hidden}
.image-wrap img{display:block;width:auto;height:auto;max-width:100%;max-height:100%;object-fit:contain;background:#000;margin:auto}
.image-card .image-wrap::after{display:none}
@media(max-width:1100px){#mainImages{grid-template-columns:1fr}.image-wrap,.image-wrap.wide{height:min(72vh,760px)}.slice-only{grid-template-columns:1fr}}
@media(max-width:760px){.image-wrap,.image-wrap.wide{height:64vh;min-height:340px}.main{padding:18px 12px 36px}}

</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div class="mark"></div><div><strong>QNP // SPECTRAL</strong><small>MRI RECONSTRUCTION LAB</small></div></div>
    <nav class="nav">
      <button data-page="dashboard" class="active">◫ <span>Dashboard</span></button>
      <button data-page="upload">＋ <span>New Reconstruction</span></button>
      <button data-page="results">◉ <span>Results</span></button>
      <button data-page="history">≡ <span>History</span></button>
    </nav>
    <div class="side-foot"><div><span class="dot"></span><b>Local system</b></div><div class="muted" style="font-size:11px;margin-top:6px">No cloud inference required</div></div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div><div class="eyebrow">Native Cartesian MRI reconstruction</div><h1 class="title" id="pageTitle">Spectral MRI Reconstruction Lab</h1><div class="subtitle" id="pageSub">A local scientific workspace for QNP reconstruction and training-free quantum-guided enhancement.</div></div>
      <div class="job-pill">JOB <b id="topJob">NONE</b></div>
    </div>

    <section id="page-dashboard">
      <div class="hero">
        <div class="panel pipeline">
          <div class="eyebrow">Scientific pipeline / local observatory</div>
          <h2 style="margin:8px 0 0">From multicoil k-space to evaluated reconstruction</h2>
          <div class="spectral-strip"><span>CARTESIAN SENSE</span><span>QNP INVERSE SOLVER</span><span>QUANTUM / TRAINING-FREE</span></div>
          <div class="hero-orbit" aria-hidden="true"><i class="orbit-dot"></i><div class="orbit-core">QNP·Q</div></div>
          <div class="pipe">
            <div class="node"><b>01</b>Raw k-space</div><div class="node"><b>02</b>ESPIRiT + SENSE</div><div class="node"><b>03</b>QNP</div><div class="node"><b>04</b>Quantum</div><div class="node"><b>05</b>Metrics</div>
          </div>
          <div class="actions" style="margin-top:22px"><button class="btn primary" onclick="go('upload')">Start new reconstruction</button><button class="btn ghost" onclick="restoreLastJob()">Restore last job</button></div>
        </div>
        <div class="panel pad">
          <div class="eyebrow">System state</div>
          <h3 style="margin:8px 0 18px">Local execution</h3>
          <div class="grid" style="gap:11px">
            <div class="stage-line"><b>Flask API</b><span id="healthApi" class="tag">checking</span></div>
            <div class="stage-line"><b>QNP engine</b><span id="healthQnp" class="tag">checking</span></div>
            <div class="stage-line"><b>Quantum stage</b><span id="healthQuantum" class="tag">checking</span></div>
            <div class="stage-line"><b>Vendored QNP core</b><span id="healthUpstream" class="tag">checking</span></div>
          </div>
        </div>
      </div>
      <div class="section-title"><div><h2>Fixed reconstruction protocol</h2><p>Only the MRI slice is selected by the user.</p></div></div>
      <div class="grid g4">
        <div class="panel pad metric"><div class="label">Acceleration</div><div class="value">2×</div><div class="help">Fixed Cartesian undersampling</div></div>
        <div class="panel pad metric"><div class="label">QNP iterations</div><div class="value">20 / 15</div><div class="help">Outer / inner iterations</div></div>
        <div class="panel pad metric"><div class="label">Calibration</div><div class="value">24 ACS</div><div class="help">Central calibration lines retained</div></div>
        <div class="panel pad metric"><div class="label">Processing</div><div class="value">Local</div><div class="help">QNP + training-free quantum stage</div></div>
      </div>
    </section>

    <section id="page-upload" class="hidden">
      <div class="grid g2">
        <div class="panel pad">
          <div class="eyebrow">Input</div><h2>Upload multicoil MRI</h2>
          <label id="drop" class="drop"><input id="fileInput" type="file" accept=".h5,.hdf5"/><div style="font-size:28px">⌁</div><h3>Drop fastMRI H5/HDF5 here</h3><p>Raw multicoil k-space with a <code>kspace</code> dataset</p></label>
          <div id="uploadState" class="muted" style="margin-top:13px;font-size:12px">No file selected.</div>
          <div id="fileFacts" class="grid g4 hidden" style="margin-top:16px"></div>
        </div>
        <div class="panel pad">
          <div class="eyebrow">Reconstruction</div><h2>Select MRI slice</h2>
          <div class="slice-only">
            <div class="field"><label>Slice index</label><input id="slice" type="number" min="0" value="8"></div>
            <div class="protocol-stamp"><span>FIXED PROTOCOL</span><b>2× · QNP 20 / 15</b><small>24 ACS · wavelet 0.0001 · TV 0.0005</small></div>
          </div>
          <div class="notice" style="margin-top:15px">The reconstruction protocol is locked for consistent demonstrations. Choose only the slice, then run.</div>
          <div class="actions" style="margin-top:17px"><button id="runBtn" class="btn primary" disabled onclick="startReconstruction()">Run reconstruction</button></div>
        </div>
      </div>
      <div id="progressPanel" class="panel pad hidden" style="margin-top:18px">
        <div class="stage-line"><div><div class="eyebrow">Live job</div><b id="stageText">Queued</b></div><span id="progressText">0%</span></div>
        <div class="progress"><i id="progressBar"></i></div>
        <details style="margin-top:16px"><summary>Technical reconstruction log</summary><div class="log" id="jobLog">Waiting for backend output…</div></details>
      </div>
    </section>

    <section id="page-results" class="hidden">
      <div id="noResult" class="panel pad"><h2>No completed result selected</h2><p class="muted">Run a reconstruction or choose one from History.</p></div>
      <div id="resultContent" class="hidden">
        <div class="actions" style="justify-content:flex-end;margin-bottom:14px"><button class="btn" onclick="refreshCurrent()">Refresh metadata</button><a id="downloadBtn" class="btn primary" href="#">Download result ZIP</a></div>
        <div class="grid g4">
          <div class="panel pad metric"><div class="label">Structural similarity</div><div class="value" id="mSSIM">—</div><div class="help">SSIM · higher is better</div></div>
          <div class="panel pad metric"><div class="label">PSNR</div><div class="value" id="mPSNR">—</div><div class="help" id="mGain">Higher is better</div></div>
          <div class="panel pad metric"><div class="label">NMSE</div><div class="value" id="mNMSE">—</div><div class="help">Complex NMSE · lower is better</div></div>
          <div class="panel pad metric"><div class="label">Total reconstruction time</div><div class="value" id="mTime">—</div><div class="help">Measured by backend</div></div>
        </div>
        <div class="section-title"><div><h2>Final reconstruction</h2><p>Only the complete ground-truth reference and final quantum-enhanced reconstruction are shown. Both images are fitted fully inside the page.</p></div></div>
        <div class="grid g2" id="mainImages"></div>
        <div class="section-title"><div><h2>Processing time breakdown</h2></div></div>
        <div class="grid g3">
          <div class="panel pad metric"><div class="label">Total</div><div class="value" id="tTotal">—</div></div>
          <div class="panel pad metric"><div class="label">QNP reconstruction</div><div class="value" id="tQNP">—</div></div>
          <div class="panel pad metric"><div class="label">Quantum stage</div><div class="value" id="tQuantum">—</div></div>
        </div>
        <div class="notice" style="margin-top:18px">This system evaluates MRI reconstruction quality. Medical interpretation should be performed by a qualified clinician.</div>
      </div>
    </section>

    <section id="page-history" class="hidden">
      <div class="panel pad"><div class="eyebrow">Local browser history</div><h2>Recent reconstruction jobs</h2><div style="overflow:auto"><table class="table"><thead><tr><th>Date</th><th>Job</th><th>Slice</th><th>Acceleration</th><th>Status</th></tr></thead><tbody id="historyBody"></tbody></table></div></div>
    </section>
  </main>
</div>


<script>
const pages = ['dashboard','upload','results','history'];
const titles = {
 dashboard:['Spectral MRI Reconstruction Lab','Native Cartesian QNP reconstruction with a fixed training-free quantum-guided enhancement stage.'],
 upload:['New MRI Reconstruction','Upload fastMRI multicoil k-space and choose the slice. The scientific protocol is fixed.'],
 results:['MRI Reconstruction Results','Ground truth and final reconstruction with quality metrics and timing.'],
 history:['Reconstruction History','Reopen recently completed local jobs.']
};
let uploaded=null, currentJob=localStorage.getItem('qnp-current-job')||null, pollTimer=null, currentResult=null;

function $(id){return document.getElementById(id)}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
function go(name){pages.forEach(p=>$('page-'+p).classList.toggle('hidden',p!==name));document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===name));$('pageTitle').textContent=titles[name][0];$('pageSub').textContent=titles[name][1];if(name==='history')renderHistory();if(name==='results'&&currentJob)refreshCurrent()}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>go(b.dataset.page));

async function api(path, options={}){
  const r=await fetch(path,{cache:'no-store',...options});
  let data={};try{data=await r.json()}catch{data={error:`HTTP ${r.status}`}}
  if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);
  return data;
}

async function checkHealth(){
  try{const h=await api('/api/health');setHealth('healthApi',true);setHealth('healthQnp',h.components.run_native_fastmri_qnp);setHealth('healthQuantum',h.components.training_free_quantum_refiner);setHealth('healthUpstream',h.components.qnp_core)}catch{setHealth('healthApi',false)}
}
function setHealth(id,ok){const e=$(id);e.textContent=ok?'READY':'MISSING';e.className='tag '+(ok?'good':'')}

const drop=$('drop'), fileInput=$('fileInput');
fileInput.onchange=()=>{if(fileInput.files[0])uploadFile(fileInput.files[0])};
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)uploadFile(f)});

async function uploadFile(file){
  $('uploadState').textContent='Uploading and validating '+file.name+' …';$('runBtn').disabled=true;
  const fd=new FormData();fd.append('file',file);
  try{uploaded=await api('/api/upload',{method:'POST',body:fd});$('uploadState').innerHTML=`<span class="tag good">VALID MRI</span> &nbsp; ${esc(uploaded.original_name)}`;const i=uploaded.info||uploaded.initial_info;renderFacts(i);$('slice').max=Math.max(0,(i.slices||1)-1);$('slice').value=Math.min(8,Math.max(0,(i.slices||1)-1));$('runBtn').disabled=false}catch(e){uploaded=null;$('uploadState').innerHTML=`<span style="color:var(--red)">${esc(e.message)}</span>`;$('fileFacts').classList.add('hidden')}
}
function renderFacts(i){const vals=[['Slices',i.slices],['Coils',i.coils],['Matrix',`${i.readout} × ${i.phase_encoding}`],['Acquired PE',`${i.acquired_pe_lines}/${i.total_pe_lines}`]];$('fileFacts').innerHTML=vals.map(v=>`<div class="panel pad metric"><div class="label">${v[0]}</div><div class="value" style="font-size:19px">${v[1]}</div></div>`).join('');$('fileFacts').classList.remove('hidden')}

async function startReconstruction(){
  if(!uploaded)return;
  const payload={file_id:uploaded.file_id,slice:+$('slice').value,acceleration:2,qnp_iters:20,inner_iters:15,acs_lines:24,wavelet_lambda:0.0001,tv_beta:0.0005,q_layers:3};
  try{const r=await api('/api/reconstruct',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});currentJob=r.job_id;localStorage.setItem('qnp-current-job',currentJob);$('topJob').textContent=currentJob.slice(0,8);addHistory({job_id:currentJob,date:new Date().toISOString(),slice:payload.slice,acceleration:payload.acceleration,status:'running'});$('progressPanel').classList.remove('hidden');pollJob(true)}catch(e){alert(e.message)}
}

function pollJob(immediate=false){
  if(!currentJob)return;clearTimeout(pollTimer);
  const tick=async()=>{try{const s=await api(`/api/jobs/${currentJob}`);renderStatus(s);updateHistoryStatus(currentJob,s.status);if(s.status==='completed'){await loadResult(currentJob);go('results');return}if(s.status==='failed')return}catch(e){$('stageText').textContent='Connection retry: '+e.message}pollTimer=setTimeout(tick,1500)};
  immediate?tick():(pollTimer=setTimeout(tick,1500));
}
function renderStatus(s){$('progressPanel').classList.remove('hidden');const clean=v=>String(v??'').replace(/V11/gi,'Quantum');$('stageText').textContent=clean(s.stage||s.status||'Running');$('progressText').textContent=(s.progress??0)+'%';$('progressBar').style.width=Math.max(0,Math.min(100,s.progress??0))+'%';const logs=clean(Array.isArray(s.logs)?s.logs.join('\n'):(s.log||''));$('jobLog').textContent=logs||'Waiting for backend output…';$('jobLog').scrollTop=$('jobLog').scrollHeight}

async function loadResult(jobId){
  // Status is already completed. Retry a few times only if filesystem visibility lags.
  let lastErr=null;
  for(let n=0;n<8;n++){try{const r=await api(`/api/jobs/${jobId}/result`);currentResult=r;renderResult(r);return r}catch(e){lastErr=e;await new Promise(res=>setTimeout(res,350))}}
  throw lastErr;
}
async function refreshCurrent(){if(!currentJob)return;try{const s=await api(`/api/jobs/${currentJob}`);$('topJob').textContent=currentJob.slice(0,8);if(s.status==='completed')await loadResult(currentJob);else{renderStatus(s);go('upload');pollJob()}}catch(e){console.error(e)}}
function restoreLastJob(){if(!currentJob){alert('No saved local job.');return}$('topJob').textContent=currentJob.slice(0,8);refreshCurrent()}

function ftime(v){v=Number(v);if(!Number.isFinite(v))return '—';const s=Math.max(0,Math.round(v)),h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;if(h)return `${h} hr ${m} min ${ss} sec`;if(m)return `${m} min ${ss} sec`;return `${ss} sec`}
function num(v,n=2){return Number.isFinite(Number(v))?Number(v).toFixed(n):'—'}
function renderResult(r){
  $('noResult').classList.add('hidden');$('resultContent').classList.remove('hidden');$('topJob').textContent=(r.job_id||currentJob||'').slice(0,8);$('downloadBtn').href=r.download_url||`/api/jobs/${currentJob}/download`;
  const q=(r.metrics&&r.metrics.Quantum)||{};
  $('mSSIM').textContent=Number.isFinite(Number(q.SSIM))?(Number(q.SSIM)*100).toFixed(1)+'%':'—';$('mPSNR').textContent=Number.isFinite(Number(q.PSNR_dB))?Number(q.PSNR_dB).toFixed(2)+' dB':'—';$('mNMSE').textContent=Number.isFinite(Number(q.NMSE_complex))?Number(q.NMSE_complex).toFixed(5):'—';
  $('mGain').textContent='Final reconstruction · higher is better';
  $('mTime').textContent=ftime(r.timing&&r.timing.total_seconds);$('tTotal').textContent=ftime(r.timing&&r.timing.total_seconds);$('tQNP').textContent=ftime(r.timing&&r.timing.qnp_seconds);$('tQuantum').textContent=ftime(r.timing&&r.timing.quantum_seconds);
  const im=r.images||{};$('mainImages').innerHTML=[['Ground Truth',im.ground_truth],['Final Reconstruction',im.quantum]].map(x=>imageCard(x[0],x[1],false)).join('');
  bindImages();addHistory({job_id:r.job_id||currentJob,date:new Date().toISOString(),slice:r.parameters?.slice??r.run_parameters?.slice,acceleration:r.parameters?.acceleration??r.run_parameters?.acceleration,status:'completed'});
}
function imageCard(label,url,wide){const u=url||'';return `<article class="panel image-card"><div class="head"><b>${esc(label)}</b><span>FULL IMAGE</span></div><div class="image-wrap ${wide?'wide':''}"><img src="${esc(u)}" alt="${esc(label)}"><div class="img-error">Image endpoint failed.<code>${esc(u)}</code></div></div></article>`}
function bindImages(){document.querySelectorAll('.image-wrap img').forEach(img=>{img.onload=()=>{img.nextElementSibling.style.display='none'};img.onerror=()=>{img.nextElementSibling.style.display='flex'}})}

function hist(){try{return JSON.parse(localStorage.getItem('qnp-history')||'[]')}catch{return[]}}
function saveHist(h){localStorage.setItem('qnp-history',JSON.stringify(h.slice(0,30)))}
function addHistory(x){let h=hist();const i=h.findIndex(v=>v.job_id===x.job_id);if(i>=0)h[i]={...h[i],...x};else h.unshift(x);saveHist(h)}
function updateHistoryStatus(id,status){let h=hist();const i=h.findIndex(v=>v.job_id===id);if(i>=0){h[i].status=status;saveHist(h)}}
function renderHistory(){const h=hist();$('historyBody').innerHTML=h.length?h.map(v=>`<tr data-id="${esc(v.job_id)}"><td>${esc(new Date(v.date).toLocaleString())}</td><td><code>${esc((v.job_id||'').slice(0,8))}</code></td><td>${esc(v.slice??'—')}</td><td>${esc(v.acceleration?`${v.acceleration}×`:'—')}</td><td>${esc(v.status||'—')}</td></tr>`).join(''):`<tr><td colspan="5" class="muted">No saved jobs yet.</td></tr>`;document.querySelectorAll('#historyBody tr[data-id]').forEach(tr=>tr.onclick=()=>{currentJob=tr.dataset.id;localStorage.setItem('qnp-current-job',currentJob);refreshCurrent();go('results')})}

checkHealth();if(currentJob){$('topJob').textContent=currentJob.slice(0,8);setTimeout(()=>refreshCurrent(),250)}
</script>
</body>
</html>'''


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/health")
def health():
    components = {
        "run_native_fastmri_qnp": (PROJECT_ROOT / "run_native_fastmri_qnp.py").exists(),
        "training_free_quantum_refiner": (PROJECT_ROOT / "training_free_quantum_refiner.py").exists(),
        "qnp_core": (PROJECT_ROOT / "run_native_fastmri_qnp.py").exists(),
    }
    return jsonify(
        {
            "ok": True,
            "project_root": str(PROJECT_ROOT),
            "python": sys.executable,
            "components": components,
        }
    )


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Choose an MRI H5 file first."}), 400

    f = request.files["file"]
    original_name = f.filename or "upload.h5"
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(
            {
                "error": (
                    "Raw reconstruction supports fastMRI-style .h5/.hdf5 "
                    "multicoil k-space files."
                )
            }
        ), 400

    file_id = uuid.uuid4().hex
    d = _upload_dir(file_id)
    d.mkdir(parents=True, exist_ok=True)

    safe = secure_filename(original_name) or f"upload{ext}"
    stored_name = f"input{ext}"
    path = d / stored_name
    f.save(path)

    try:
        info = _inspect_h5(path, 0)
    except Exception as exc:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        return jsonify({"error": str(exc)}), 400

    meta = {
        "file_id": file_id,
        "original_name": original_name,
        "safe_name": safe,
        "stored_name": stored_name,
        "size_bytes": int(path.stat().st_size),
        "initial_info": info,
    }
    _write_json_atomic(d / "metadata.json", meta)
    return jsonify({**meta, "info": info})


@app.get("/api/files/<file_id>/info")
def file_info(file_id: str):
    try:
        meta = _get_upload_meta(file_id)
        slice_index = int(request.args.get("slice", "0"))
        info = _inspect_h5(Path(meta["path"]), slice_index)
        return jsonify(
            {
                "file_id": file_id,
                "original_name": meta["original_name"],
                "size_bytes": meta["size_bytes"],
                "info": info,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/reconstruct")
def reconstruct():
    payload = request.get_json(silent=True) or {}

    try:
        file_id = str(payload["file_id"])
        upload = _get_upload_meta(file_id)
        base_info = upload["initial_info"]

        params = {
            "file_id": file_id,
            "slice": int(payload.get("slice", 0)),
            "acceleration": 2,
            "qnp_iters": 20,
            "inner_iters": 15,
            "wavelet_lambda": 0.0001,
            "tv_beta": 0.0005,
            "q_layers": 3,
            "acs_lines": 24,
        }

        if not 0 <= params["slice"] < int(base_info["slices"]):
            raise ValueError(
                f"Slice must be between 0 and {int(base_info['slices']) - 1}."
            )
        if params["acceleration"] < 1:
            raise ValueError("Acceleration must be >= 1.")
        if params["qnp_iters"] < 1 or params["inner_iters"] < 1:
            raise ValueError("Iteration counts must be positive.")
        if params["q_layers"] < 1:
            raise ValueError("Quantum layers must be positive.")

        job_id = uuid.uuid4().hex
        _set_job(
            job_id,
            status="queued",
            stage="Queued",
            progress=0,
            parameters=params,
        )

        thread = threading.Thread(
            target=_worker, args=(job_id, params), daemon=True
        )
        thread.start()

        return jsonify(
            {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    try:
        state = _get_job(job_id)
        if state is None:
            return jsonify({"error": "Job not found."}), 404
        return jsonify(state)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/jobs/<job_id>/result")
def job_result(job_id: str):
    try:
        state = _get_job(job_id)
        if state is None:
            return jsonify({"error": "Job not found."}), 404
        if state.get("status") != "completed":
            return jsonify({"error": "Job has not completed.", "status": state}), 409

        p = _job_dir(job_id) / "result.json"
        if not p.exists():
            return jsonify({"error": "Result summary is missing."}), 500
        response = jsonify(json.loads(p.read_text(encoding="utf-8")))
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/jobs/<job_id>/assets/<path:filename>")
def job_asset(job_id: str, filename: str):
    try:
        response = send_from_directory(
            _job_dir(job_id) / "web",
            filename,
            max_age=0,
            conditional=False,
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/jobs/<job_id>/download")
def job_download(job_id: str):
    try:
        job_dir = _job_dir(job_id)
        return send_from_directory(
            job_dir,
            "result_bundle.zip",
            as_attachment=True,
            download_name=f"qnp_quantum_{job_id[:8]}_results.zip",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "The uploaded file exceeds the 8 GB local-app limit."}), 413


if __name__ == "__main__":
    components = {
        "run_native_fastmri_qnp.py": PROJECT_ROOT / "run_native_fastmri_qnp.py",
        "training_free_quantum_refiner.py": PROJECT_ROOT / "training_free_quantum_refiner.py",
        "qnp runtime core": PROJECT_ROOT / "run_native_fastmri_qnp.py",
    }
    print()
    print("QNP + Training-Free Quantum MRI — SINGLE-FILE WEB APP")
    print("Project root:", PROJECT_ROOT)
    print("Python      :", sys.executable)
    print("Open        : http://127.0.0.1:8080")
    print()
    missing = [name for name, p in components.items() if not p.exists()]
    if missing:
        print("WARNING: reconstruction components missing:")
        for name in missing:
            print("  -", name)
        print("The UI will open, but reconstruction cannot run until they are present.")
        print()
    print("Press Ctrl+C to stop.")
    print()
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
