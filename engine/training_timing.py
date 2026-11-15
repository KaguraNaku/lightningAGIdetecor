"""Optional shared-recorder hooks around real native compute; no extra batches."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import socket
import time


class FitTiming:
    def __init__(self, spec, *, runtime, loader, validation_loader, output):
        self.started = time.perf_counter()
        self.wall_seconds = {}
        self.output, self.meter = Path(output), None
        self.ids, self.exposures, self.steps, self.epochs = set(), 0, 0, 0
        if not spec:
            return
        import torch
        path = Path(spec['module'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError('Shared timing module checksum changed')
        definition = importlib.util.spec_from_file_location('authorized_gpu_timing', path)
        module = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(module)
        metadata = {**spec.get('metadata', {}), 'Python_version': platform.python_version(),
            'torch_version': str(torch.__version__), 'CUDA_version': torch.version.cuda,
            'CPU_threads': torch.get_num_threads(), 'precision': runtime.precision,
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'TF32_cudnn': torch.backends.cudnn.allow_tf32, 'batch_size': loader.batch_size,
            'workers': loader.num_workers, 'cold_start': 'model construction excluded;first real batch retained',
            'train_scope': 'H2D+forward+backward+native optimizer/scaler;decode and provenance excluded',
            'source_validation_scope': 'H2D+predict;score-copy and prediction serialization excluded',
            'extra_warmup_batches': 0}
        metadata.update(origin_host=socket.gethostname(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            manifest_sha256=self._sha(loader.dataset.manifest_path),
            validation_manifest_sha256=self._sha(validation_loader.dataset.manifest_path) if validation_loader else None)
        if runtime.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(runtime.device)
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
            metadata['GPU_uuid_and_name'] = gpu
            metadata['gpu_uuid'], metadata['gpu_name'] = [x.strip() for x in gpu.split(',', 1)]
            metadata['peak_reset_scope'] = 'This fit entry;includes model-to-device and restored optimizer allocation'
        self.meter = module.GPUTimeRecorder(device=str(runtime.device), warmup_batches=3, metadata=metadata)

    @staticmethod
    def _sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    @contextmanager
    def wall(self, phase):
        before = time.perf_counter()
        try:
            yield
        finally:
            self.wall_seconds[phase] = self.wall_seconds.get(phase, 0.) + time.perf_counter() - before

    @contextmanager
    def batch(self, batch):
        if self.meter is None:
            yield
            return
        with self.meter.region('train_step', samples=len(batch['labels']),
                scope='H2D+forward+backward+native optimizer/scaler'):
            yield
        self.steps += 1
        self.exposures += len(batch['labels'])
        self.ids.update(m['id'] for m in batch['metadata'])

    def finish(self, status):
        if self.meter is None:
            return
        self.meter.metadata.update(status=status, actual_unique_images=len(self.ids),
            image_exposures=self.exposures, optimizer_steps=self.steps, epoch_count=self.epochs,
            batch_count=self.steps, wall_fit_seconds=time.perf_counter() - self.started,
            wall_train_seconds=self.wall_seconds.get('train', 0.),
            wall_source_validation_seconds=self.wall_seconds.get('source_validation', 0.),
            wall_checkpoint_restore_seconds=self.wall_seconds.get('checkpoint_restore', 0.),
            wall_checkpoint_seconds=self.wall_seconds.get('checkpoint', 0.))
        self.meter.write(self.output / 'timing.json')


def timed_fit(function):
    @wraps(function)
    def wrapped(self, loader, *, validation_loader, runtime, settings, output_dir):
        timing = FitTiming(settings.get('gpu_timing'), runtime=runtime, loader=loader,
            validation_loader=validation_loader, output=output_dir)
        self._timing = timing
        status = 'failed'
        try:
            result = function(self, loader, validation_loader=validation_loader,
                runtime=runtime, settings=settings, output_dir=output_dir)
            status = 'complete'
            if timing.meter is not None:
                result['timing'] = 'timing.json'
            return result
        finally:
            timing.finish(status)
            self._timing = None
    return wrapped


def training_region(trainer, batch):
    timer = getattr(trainer, '_timing', None)
    return timer.batch(batch) if timer is not None else nullcontext()


def host_region(trainer, phase):
    timer = getattr(trainer, '_timing', None)
    return timer.wall(phase) if timer is not None else nullcontext()


def validation_meter(trainer):
    timer = getattr(trainer, '_timing', None)
    return timer.meter if timer is not None else None


def update_export_timing(output, *, fit_wall, export_wall, model_path):
    path = Path(output) / 'timing.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data['metadata']['wall_fit_seconds'] = fit_wall
    data['metadata']['wall_checkpoint_seconds'] += export_wall
    data['metadata']['model_export_wall_seconds'] = export_wall
    with Path(model_path).open('rb') as stream:
        data['metadata']['selected_model_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    data['metadata']['checkpoint_sha256'] = data['metadata']['selected_model_sha256']
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
