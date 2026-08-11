"""Validate the exact benchmark reproduction matrix and write a hash manifest."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from src.utils.artifacts import sha256_file, sha256_json, write_json_artifact
from src.utils.checkpoint import load_checkpoint
from src.utils.config_parser import load_config


MODEL_KEYS = ("fn", "fcn", "unet", "rnn", "lstm", "gru", "transformer", "gcn")
CONDITIONS = (
    ("mnist", "mnist", 512, 10_000),
    ("mnist", "mnist", 1024, 10_000),
    ("mnist", "mnist", 2048, 10_000),
    ("fmnist", "fashion_mnist", 512, 10_000),
    ("celeba", "celeba", 512, 20_260),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
METRIC_KEYS = {"l1", "mse", "psnr", "ssim"}


def _read_json(path):
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return raw, data


def _finite_metrics(path, metrics):
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError(f"{path}: metrics must contain exactly {sorted(METRIC_KEYS)}")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path}: metric {name} is not finite: {value!r}")


def _entry(path, raw, *, kind, dataset=None, bucket_size=None, n=None):
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": kind,
        "dataset": dataset,
        "bucket_size": bucket_size,
        "n": n,
    }


def _effective_gi_config(dataset, bucket_size):
    config = load_config("configs/experiments_mnist/GI.yaml")
    config["data"]["dataset"] = dataset
    config["data"]["val_dir"] = f"data/{dataset}"
    config["data"]["bucket_size"] = bucket_size
    return config


def _validate_checkpoint_lineage(path, payload, expected, checkpoint_root):
    checkpoint_path = Path(checkpoint_root) / expected["experiment_name"] / "best.pth"
    if not checkpoint_path.is_file():
        raise ValueError(f"{path}: missing checkpoint lineage file {checkpoint_path}")
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if payload.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError(f"{path}: checkpoint SHA-256 does not match {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError(f"{path}: checkpoint has no embedded effective config")
    config = checkpoint["config"]
    config_expected = {
        ("model", "name"): expected["model"],
        ("data", "dataset"): expected["dataset"],
        ("data", "bucket_size"): expected["bucket_size"],
        ("training", "experiment_name"): expected["experiment_name"],
        ("training", "seed"): expected["seed"],
        ("training", "epochs"): expected["configured_epochs"],
    }
    for (section, field), value in config_expected.items():
        actual = config.get(section, {}).get(field)
        if actual != value:
            raise ValueError(
                f"{path}: checkpoint config {section}.{field}={actual!r}, expected {value!r}"
            )
    if payload.get("effective_config_sha256") != sha256_json(config):
        raise ValueError(f"{path}: effective config SHA-256 does not match checkpoint config")
    if payload.get("checkpoint_epoch") != checkpoint.get("epoch"):
        raise ValueError(f"{path}: checkpoint_epoch does not match checkpoint payload")


def validate(
    result_root, *, seed, epochs, run_id, include_cs=False,
    checkpoint_root=Path("checkpoints"),
):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(run_id)):
        raise ValueError("run_id contains unsafe characters")
    root = Path(result_root)
    expected_paths = set()
    entries = []

    expected_models = {}
    for key in MODEL_KEYS:
        mnist_cfg = load_config(f"configs/experiments_mnist/{key}.yaml")
        celeba_cfg = load_config(f"configs/experiments_celeba/{key}.yaml")
        expected_models[(key, "mnist")] = mnist_cfg["model"]["name"]
        expected_models[(key, "fmnist")] = mnist_cfg["model"]["name"]
        expected_models[(key, "celeba")] = celeba_cfg["model"]["name"]

    for bucket_size in (512, 1024, 2048):
        path = root / "profile" / f"benchmark_mnist_{bucket_size}.json"
        expected_paths.add(path.resolve())
        raw, payload = _read_json(path)
        if not isinstance(payload, list) or len(payload) != len(MODEL_KEYS):
            raise ValueError(f"{path}: profile must contain exactly {len(MODEL_KEYS)} models")
        observed = set()
        for record in payload:
            if record.get("bucket_size") != bucket_size:
                raise ValueError(f"{path}: wrong bucket_size in profile record: {record}")
            model_name = record.get("model_name")
            observed.add(model_name)
            if not isinstance(record.get("num_params"), int) or record["num_params"] <= 0:
                raise ValueError(f"{path}: invalid num_params for {model_name}")
            latency = record.get("inference_time_ms")
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"{path}: invalid inference time for {model_name}")
            flops = record.get("flops_G")
            if flops is not None and (
                isinstance(flops, bool)
                or not isinstance(flops, (int, float))
                or not math.isfinite(flops)
                or flops < 0
            ):
                raise ValueError(f"{path}: invalid FLOPs for {model_name}")
        wanted = {expected_models[(key, "mnist")] for key in MODEL_KEYS}
        if observed != wanted:
            raise ValueError(f"{path}: model set mismatch: {sorted(observed)} != {sorted(wanted)}")
        entries.append(_entry(path.relative_to(root.parent), raw, kind="profile", bucket_size=bucket_size))

    for key in MODEL_KEYS:
        for label, dataset, bucket_size, expected_n in CONDITIONS:
            name = f"{key}_{label}_{bucket_size}_seed{seed}_{run_id}"
            path = root / "matrix" / f"{name}.json"
            expected_paths.add(path.resolve())
            raw, payload = _read_json(path)
            expected = {
                "schema_version": 1,
                "script": "scripts.evaluate",
                "model": expected_models[(key, label)],
                "dataset": dataset,
                "bucket_size": bucket_size,
                "split": "val",
                "experiment_name": name,
                "seed": seed,
                "configured_epochs": epochs,
                "n": expected_n,
            }
            for field, value in expected.items():
                if payload.get(field) != value:
                    raise ValueError(f"{path}: {field}={payload.get(field)!r}, expected {value!r}")
            checkpoint_epoch = payload.get("checkpoint_epoch")
            if not isinstance(checkpoint_epoch, int) or not 1 <= checkpoint_epoch <= epochs:
                raise ValueError(f"{path}: invalid checkpoint_epoch={checkpoint_epoch!r}")
            for field in ("checkpoint_sha256", "effective_config_sha256"):
                if not SHA256_RE.fullmatch(str(payload.get(field, ""))):
                    raise ValueError(f"{path}: invalid {field}")
            _validate_checkpoint_lineage(path, payload, expected, checkpoint_root)
            _finite_metrics(path, payload.get("metrics"))
            entries.append(_entry(
                path.relative_to(root.parent), raw, kind="learned", dataset=dataset,
                bucket_size=bucket_size, n=expected_n,
            ))

    for label, dataset, bucket_size, expected_n in CONDITIONS:
        path = root / "matrix" / f"physics_{label}_{bucket_size}.json"
        expected_paths.add(path.resolve())
        raw, payload = _read_json(path)
        expected = {
            "schema_version": 1,
            "script": "scripts.GI_rec_eval",
            "method": "Hadamard adjoint",
            "dataset": dataset,
            "bucket_size": bucket_size,
            "split": "val",
            "n": expected_n,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(f"{path}: {field}={payload.get(field)!r}, expected {value!r}")
        for field in ("source_config_sha256", "effective_config_sha256"):
            if not SHA256_RE.fullmatch(str(payload.get(field, ""))):
                raise ValueError(f"{path}: invalid {field}")
        source_config = Path("configs/experiments_mnist/GI.yaml")
        if payload.get("source_config_sha256") != sha256_file(source_config):
            raise ValueError(f"{path}: source config SHA-256 mismatch")
        if payload.get("effective_config_sha256") != sha256_json(
            _effective_gi_config(dataset, bucket_size)
        ):
            raise ValueError(f"{path}: effective config SHA-256 mismatch")
        _finite_metrics(path, payload.get("metrics"))
        entries.append(_entry(
            path.relative_to(root.parent), raw, kind="physics", dataset=dataset,
            bucket_size=bucket_size, n=expected_n,
        ))

    if include_cs:
        path = root / "baselines" / "cs_admm_mnist_512.json"
        expected_paths.add(path.resolve())
        raw, payload = _read_json(path)
        expected = {
            "schema_version": 1,
            "script": "scripts.CS_GI_rec_eval",
            "method": "admm",
            "dataset": "mnist",
            "bucket_size": 512,
            "split": "val",
            "n": 10_000,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(
                    f"{path}: {field}={payload.get(field)!r}, expected {value!r}"
                )
        for field in ("source_config_sha256", "effective_config_sha256"):
            if not SHA256_RE.fullmatch(str(payload.get(field, ""))):
                raise ValueError(f"{path}: invalid {field}")
        source_config = Path("configs/experiments_mnist/GI.yaml")
        if payload.get("source_config_sha256") != sha256_file(source_config):
            raise ValueError(f"{path}: source config SHA-256 mismatch")
        if payload.get("effective_config_sha256") != sha256_json(
            _effective_gi_config("mnist", 512)
        ):
            raise ValueError(f"{path}: effective config SHA-256 mismatch")
        _finite_metrics(path, payload.get("metrics"))
        entries.append(_entry(
            path.relative_to(root.parent), raw, kind="optional_cs", dataset="mnist",
            bucket_size=512, n=10_000,
        ))

    actual_paths = {
        path.resolve() for path in root.rglob("*.json")
        if path.resolve() != (root / "manifest.json").resolve()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"result JSON inventory mismatch; missing={missing}, extra={extra}")

    expected_count = 48 + int(include_cs)
    if len(entries) != expected_count:
        raise AssertionError(f"internal expected artifact count mismatch: {len(entries)}")
    return {
        "schema_version": 2,
        "run_id": run_id,
        "seed": seed,
        "configured_epochs": epochs,
        "expected_artifact_count": expected_count,
        "artifacts": entries,
        "python": sys.version,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--include_cs", action="store_true")
    args = parser.parse_args()
    payload = validate(
        args.result_root, seed=args.seed, epochs=args.epochs,
        run_id=args.run_id, include_cs=args.include_cs,
    )
    write_json_artifact(args.manifest, payload)
    print(f"Validated {payload['expected_artifact_count']} exact artifacts: {args.manifest}")


if __name__ == "__main__":
    main()
