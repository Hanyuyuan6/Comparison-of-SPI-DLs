import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import zipfile

import numpy as np
import pytest
import torch
from scipy.io import savemat

import src.utils.checkpoint as checkpoint_module
import scripts.prepare_celeba as prepare_celeba_module
from scripts.prepare_celeba import (
    _validate_prepared as validate_prepared_celeba,
    _verify_checksum as verify_celeba_checksum,
)
import scripts.prepare_datasets as prepare_datasets_module
from scripts.prepare_datasets import DatasetDownloader, parse_checksum_overrides
from scripts.predict_from_bucket import load_bucket_signal
from src.utils.seed import seed_everything
from src.utils.config_parser import load_config
from scripts.validate_reproduction_artifacts import CONDITIONS, MODEL_KEYS, validate
from src.utils.artifacts import sha256_file, sha256_json
from src.trainer import Trainer
from src.utils.path_safety import safe_child_directory


@pytest.mark.parametrize(
    'name', ['', '.', '..', '../escape', '..\\escape', '/tmp/escape', ' trailing'],
)
def test_experiment_output_name_cannot_escape_root(tmp_path, name):
    with pytest.raises((TypeError, ValueError), match='experiment_name'):
        safe_child_directory(tmp_path, name)


def test_experiment_output_name_preserves_released_names(tmp_path):
    assert safe_child_directory(
        tmp_path, 'fn_mnist_512_seed42_r1') == tmp_path / 'fn_mnist_512_seed42_r1'


def test_checkpoint_loader_fails_closed(monkeypatch):
    calls = []

    def fake_load(path, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise ValueError('synthetic safe-loader failure')
        return {'legacy': True}

    monkeypatch.setattr(checkpoint_module.torch, 'load', fake_load)
    with pytest.raises(RuntimeError, match='Refusing to fall back'):
        checkpoint_module.load_checkpoint('legacy.pth', map_location='cpu')
    assert calls == [True], 'default path must never call weights_only=False'

    assert checkpoint_module.load_checkpoint(
        'trusted-legacy.pth', map_location='cpu', allow_unsafe_pickle=True
    ) == {'legacy': True}
    assert calls == [True, True, False]


def test_existing_archives_are_always_checksum_verified(tmp_path):
    archive = tmp_path / 'archive.zip'
    archive.write_bytes(b'verified fixture')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    downloader = DatasetDownloader(tmp_path)

    assert downloader.download_file('https://invalid.example/archive.zip', archive, digest)
    with pytest.raises(ValueError, match='No trusted SHA-256'):
        downloader.download_file('https://invalid.example/archive.zip', archive, None)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        downloader.download_file('https://invalid.example/archive.zip', archive, '0' * 64)
    assert archive.read_bytes() == b'verified fixture', 'a mismatch must not delete user data'

    assert verify_celeba_checksum(archive, digest)
    with pytest.raises(ValueError, match='No trusted SHA-256'):
        verify_celeba_checksum(archive, None)


def test_checksum_override_parser_is_strict():
    digest = 'a' * 64
    assert parse_checksum_overrides([f'data.zip={digest}']) == {'data.zip': digest}
    for bad in ('data.zip', '../data.zip=' + digest, 'data.zip=abc'):
        with pytest.raises(ValueError):
            parse_checksum_overrides([bad])


def test_archive_extraction_rejects_links(tmp_path):
    downloader = DatasetDownloader(tmp_path)
    zip_path = tmp_path / 'linked.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        link = zipfile.ZipInfo('link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, '../outside')
    with pytest.raises(ValueError, match='symbolic link'):
        downloader.extract_archive(zip_path, tmp_path / 'zip_out')

    tar_path = tmp_path / 'linked.tar'
    with tarfile.open(tar_path, 'w') as archive:
        regular = tarfile.TarInfo('inside.txt')
        regular.size = 2
        archive.addfile(regular, io.BytesIO(b'ok'))
        link = tarfile.TarInfo('link')
        link.type = tarfile.SYMTYPE
        link.linkname = '../outside'
        archive.addfile(link)
    with pytest.raises(ValueError, match='non-regular TAR member'):
        downloader.extract_archive(tar_path, tmp_path / 'tar_out')


def test_missing_celeba_preparation_script_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare_datasets_module,
        'PREPARE_CELEBA_SCRIPT',
        tmp_path / 'missing_prepare_celeba.py',
    )
    with pytest.raises(RuntimeError, match='preparation script not found'):
        prepare_datasets_module.prepare_celeba(tmp_path)


def test_partial_celeba_directories_are_not_accepted(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'val').mkdir()
    (tmp_path / 'train' / 'junk.txt').write_text('not an image')
    (tmp_path / 'val' / 'junk.txt').write_text('not an image')
    with pytest.raises(RuntimeError, match='Partial or unverified'):
        validate_prepared_celeba(tmp_path)


def test_celeba_manifest_binds_archive_and_prepared_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_celeba_module, 'EXPECTED_TOTAL', 3)
    monkeypatch.setattr(prepare_celeba_module, 'EXPECTED_TRAIN', 2)
    monkeypatch.setattr(prepare_celeba_module, 'EXPECTED_VAL', 1)
    train = tmp_path / 'train'
    val = tmp_path / 'val'
    train.mkdir()
    val.mkdir()
    (train / 'a.jpg').write_bytes(b'jpeg-a')
    (train / 'b.jpg').write_bytes(b'jpeg-b')
    (val / 'c.jpg').write_bytes(b'jpeg-c')
    archive = tmp_path / 'celeba.zip'
    archive.write_bytes(b'trusted-archive-fixture')
    train_files = sorted(train.glob('*.jpg'))
    val_files = sorted(val.glob('*.jpg'))
    meta = {
        'schema_version': 2,
        'source_archive': 'celeba.zip',
        'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'random_seed': 42,
        'val_split': 0.1,
        'total': 3,
        'train_count': 2,
        'val_count': 1,
        'train_names_sha256': prepare_celeba_module._name_digest(train_files),
        'val_names_sha256': prepare_celeba_module._name_digest(val_files),
        'train_content_sha256': prepare_celeba_module._content_digest(train_files),
        'val_content_sha256': prepare_celeba_module._content_digest(val_files),
    }
    (tmp_path / '_prepared_meta.json').write_text(json.dumps(meta), encoding='utf-8')
    assert validate_prepared_celeba(tmp_path)

    (train / 'a.jpg').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='manifest mismatch'):
        validate_prepared_celeba(tmp_path)


def test_mat_bucket_requires_unambiguous_or_explicit_key(tmp_path):
    path = tmp_path / 'ambiguous.mat'
    savemat(path, {
        'B': np.arange(8, dtype=np.float32),
        'd_B': np.arange(8, dtype=np.float32) + 100,
    })

    with pytest.raises(ValueError, match='Multiple eligible bucket vectors'):
        load_bucket_signal(str(path), bucket_size=8)
    assert np.array_equal(
        load_bucket_signal(str(path), bucket_size=8, mat_key='d_B'),
        np.arange(8, dtype=np.float32) + 100,
    )


@pytest.mark.parametrize('values', [
    np.array([0.0, np.nan, 2.0], dtype=np.float32),
    np.array([0.0, np.inf, 2.0], dtype=np.float32),
    np.ones(3, dtype=np.float32),
])
def test_bucket_input_rejects_nonfinite_or_constant_values(tmp_path, values):
    path = tmp_path / 'bucket.npy'
    np.save(path, values)
    with pytest.raises(ValueError):
        load_bucket_signal(str(path), bucket_size=3)


def _tiny_trainer_config(name, refuse=False):
    return {
        'training': {
            'epochs': 1, 'amp': False, 'experiment_name': name,
            'refuse_existing_output': refuse, 'intra_epoch_val_times': 0,
        },
        'logging': {'use_wandb': False},
    }


def test_trainer_refuses_stale_output_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'checkpoints' / 'stale').mkdir(parents=True)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(FileExistsError, match='refusing to reuse'):
        Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.1), torch.nn.MSELoss(),
            [], [], None, torch.device('cpu'), _tiny_trainer_config('stale', refuse=True),
        )


def test_trainer_rejects_nonfinite_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [{'bucket': torch.ones(1, 2), 'image': torch.ones(1, 2)}]
    criterion = lambda pred, target: pred.sum() * torch.tensor(float('nan'))
    trainer = Trainer(
        model, optimizer, criterion, loader, loader, None, torch.device('cpu'),
        _tiny_trainer_config('nan'),
    )
    try:
        with pytest.raises(FloatingPointError, match='non-finite training loss'):
            trainer._train_epoch(1)
    finally:
        trainer.writer.close()


def test_seed_configures_cublas_determinism(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    seed_everything(123)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    first = torch.rand(4)
    seed_everything(123)
    assert torch.equal(first, torch.rand(4))


def test_full_sweep_is_fail_closed_and_emits_artifacts():
    script = (Path(__file__).parents[1] / 'scripts' / 'run_all.sh').read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert '|| true' not in script
    assert '--out_json' in script
    assert 'for bucket_size in 512 1024 2048' in script
    assert 'manifest.json' in script
    assert 'refusing to reuse existing RESULT_ROOT' in script


def test_reproduction_validator_requires_exact_matrix_and_identity(tmp_path):
    root = tmp_path / 'results' / 'r1'
    (root / 'profile').mkdir(parents=True)
    (root / 'table1').mkdir()
    (root / 'baselines').mkdir()
    model_names = {
        key: load_config(f'configs/experiments_mnist/{key}.yaml')['model']['name']
        for key in MODEL_KEYS
    }
    for bucket_size in (512, 1024, 2048):
        payload = [{
            'model_name': model_names[key], 'num_params': 1,
            'inference_time_ms': 0.1, 'flops_G': None,
            'bucket_size': bucket_size,
        } for key in MODEL_KEYS]
        (root / 'profile' / f'benchmark_mnist_{bucket_size}.json').write_text(
            json.dumps(payload), encoding='utf-8')

    metrics = {'l1': 0.1, 'mse': 0.1, 'psnr': 1.0, 'ssim': 0.5}
    checkpoint_root = tmp_path / 'checkpoints'
    for key in MODEL_KEYS:
        for label, dataset, bucket_size, n in CONDITIONS:
            name = f'{key}_{label}_{bucket_size}_seed42_r1'
            config_group = 'experiments_celeba' if label == 'celeba' else 'experiments_mnist'
            checkpoint_config = load_config(f'configs/{config_group}/{key}.yaml')
            checkpoint_config['model']['params']['bucket_size'] = bucket_size
            checkpoint_config['data']['dataset'] = dataset
            checkpoint_config['data']['val_dir'] = f'data/{dataset}'
            checkpoint_config['data']['bucket_size'] = bucket_size
            checkpoint_config['training']['experiment_name'] = name
            checkpoint_config['training']['seed'] = 42
            checkpoint_config['training']['epochs'] = 1
            checkpoint_path = checkpoint_root / name / 'best.pth'
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({
                'config': checkpoint_config,
                'epoch': 1,
                'model_state_dict': {},
            }, checkpoint_path)
            payload = {
                'schema_version': 1, 'script': 'scripts.evaluate',
                'model': model_names[key], 'dataset': dataset,
                'bucket_size': bucket_size, 'split': 'val',
                'experiment_name': name, 'seed': 42, 'configured_epochs': 1,
                'checkpoint_epoch': 1, 'n': n,
                'checkpoint_sha256': sha256_file(checkpoint_path),
                'effective_config_sha256': sha256_json(checkpoint_config),
                'metrics': metrics,
            }
            (root / 'table1' / f'{name}.json').write_text(
                json.dumps(payload), encoding='utf-8')
    for label, dataset, bucket_size, n in CONDITIONS:
        gi_config = load_config('configs/experiments_mnist/GI.yaml')
        gi_config['data']['dataset'] = dataset
        gi_config['data']['val_dir'] = f'data/{dataset}'
        gi_config['data']['bucket_size'] = bucket_size
        payload = {
            'schema_version': 1, 'script': 'scripts.GI_rec_eval',
            'method': 'Hadamard adjoint', 'dataset': dataset,
            'bucket_size': bucket_size, 'split': 'val', 'n': n,
            'source_config_sha256': sha256_file('configs/experiments_mnist/GI.yaml'),
            'effective_config_sha256': sha256_json(gi_config), 'metrics': metrics,
        }
        (root / 'table1' / f'physics_{label}_{bucket_size}.json').write_text(
            json.dumps(payload), encoding='utf-8')

    payload = validate(
        root, seed=42, epochs=1, run_id='r1', checkpoint_root=checkpoint_root,
    )
    assert payload['expected_artifact_count'] == 48

    nested_manifest = root / 'junk' / 'manifest.json'
    nested_manifest.parent.mkdir()
    nested_manifest.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory mismatch'):
        validate(root, seed=42, epochs=1, run_id='r1', checkpoint_root=checkpoint_root)
    nested_manifest.unlink()

    lineage = root / 'table1' / 'fn_mnist_512_seed42_r1.json'
    lineage_content = json.loads(lineage.read_text(encoding='utf-8'))
    original_sha = lineage_content['checkpoint_sha256']
    lineage_content['checkpoint_sha256'] = '0' * 64
    lineage.write_text(json.dumps(lineage_content), encoding='utf-8')
    with pytest.raises(ValueError, match='checkpoint SHA-256 does not match'):
        validate(root, seed=42, epochs=1, run_id='r1', checkpoint_root=checkpoint_root)
    lineage_content['checkpoint_sha256'] = original_sha
    lineage.write_text(json.dumps(lineage_content), encoding='utf-8')

    wrong = root / 'table1' / 'gru_mnist_512_seed42_r1.json'
    content = json.loads(wrong.read_text(encoding='utf-8'))
    content['model'] = 'WrongModel'
    wrong.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ValueError, match='model='):
        validate(root, seed=42, epochs=1, run_id='r1', checkpoint_root=checkpoint_root)
