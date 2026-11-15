"""Optional CUDA-event and host timing; never changes RNG, tensors or batch order."""
from contextlib import contextmanager
import json
from pathlib import Path
import time


class GPUTimeRecorder:
    """Events bound real compute calls; one synchronization only when flushing.

    CUDA elapsed is stream elapsed, not a sum of active SM kernel durations.
    Callers must state whether H2D/score copies are included in the region.
    No synthetic warmup runs are executed. All natural batches remain in totals.
    """
    def __init__(self, *, device='cuda:0', warmup_batches=3, metadata=None,
                 cuda_backend=None, clock=time.perf_counter):
        if warmup_batches < 0:
            raise ValueError('warmup_batches must be nonnegative')
        self.device, self.warmup_batches = device, warmup_batches
        self.metadata, self.clock = dict(metadata or {}), clock
        self.cuda = cuda_backend
        if self.cuda is None and str(device).startswith('cuda'):
            import torch
            self.cuda = torch.cuda
            self.metadata.setdefault('torch_version', str(torch.__version__))
            self.metadata.setdefault('CUDA_version', torch.version.cuda)
        self.cuda_enabled = self.cuda is not None and self.cuda.is_available() and str(device).startswith('cuda')
        self.pending, self.rows, self.indices = [], [], {}
        self.created_at = self.clock()

    @contextmanager
    def region(self, phase, *, samples=0, scope='', is_batch=True):
        if samples < 0:
            raise ValueError('samples must be nonnegative')
        index = self.indices.get(phase, 0)
        self.indices[phase] = index + 1
        row = {'phase': phase, 'scope': scope, 'samples': int(samples),
               'is_batch': is_batch, 'batch_index': index if is_batch else None,
               'natural_warmup': bool(is_batch and index < self.warmup_batches),
               'completed': False, 'cuda_event_ms': None}
        start = end = None
        if self.cuda_enabled:
            stream = self.cuda.current_stream(self.device)
            start = self.cuda.Event(enable_timing=True)
            end = self.cuda.Event(enable_timing=True)
            start.record(stream)
        begin = self.clock()
        try:
            yield
            row['completed'] = True
        finally:
            row['host_region_seconds'] = self.clock() - begin
            if self.cuda_enabled:
                end.record(stream)
                self.pending.append((row, start, end))
            self.rows.append(row)

    def flush(self):
        if self.pending:
            self.cuda.synchronize(self.device)
            for row, start, end in self.pending:
                row['cuda_event_ms'] = float(start.elapsed_time(end))
            self.pending.clear()

    @staticmethod
    def _aggregate(rows, cuda_enabled):
        images = sum(r['samples'] for r in rows)
        ms = sum(r['cuda_event_ms'] for r in rows) if rows and cuda_enabled else None
        wall = sum(r['host_region_seconds'] for r in rows)
        return {'calls': len(rows), 'samples': images,
                'completed_calls': sum(r['completed'] for r in rows),
                'failed_calls': sum(not r['completed'] for r in rows),
                'cuda_event_seconds': None if ms is None else ms / 1000.,
                'host_region_seconds': wall,
                'cuda_event_ms_per_image': ms / images if ms is not None and images else None,
                'cuda_event_images_per_second': images * 1000 / ms if ms is not None and ms > 0 else None}

    def summary(self):
        self.flush()
        phases = {}
        for phase in sorted({r['phase'] for r in self.rows}):
            rr = [r for r in self.rows if r['phase'] == phase]
            phases[phase] = {'all_real_calls': self._aggregate(rr, self.cuda_enabled),
                'excluding_natural_warmup': self._aggregate([r for r in rr if not r['natural_warmup']], self.cuda_enabled),
                'scopes': sorted({r['scope'] for r in rr})}
        memory = {'scope': 'process allocator peak since caller last reset;timer does not reset counters',
                  'max_allocated_bytes': None, 'max_reserved_bytes': None}
        if self.cuda_enabled:
            memory.update(max_allocated_bytes=int(self.cuda.max_memory_allocated(self.device)),
                          max_reserved_bytes=int(self.cuda.max_memory_reserved(self.device)))
        return {'version': 1, 'measurement': 'CUDA event elapsed on current stream;not SM active-kernel sum',
            'cuda_available': self.cuda_enabled, 'device': str(self.device),
            'missing_cuda_reason': None if self.cuda_enabled else 'CPU phase or CUDA unavailable;not a measured zero',
            'warmup': {'mode': 'exclude first natural batches only from steady summary;no added runs',
                       'batches_per_phase': self.warmup_batches, 'total_includes_warmup': True},
            'synchronization': 'Once per flush,never automatically per batch',
            'recorder_lifetime_wall_seconds': self.clock() - self.created_at,
            'metadata': self.metadata, 'phases': phases, 'peak_memory': memory}

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        tmp = path.with_suffix('.partial.json')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)
        return data
