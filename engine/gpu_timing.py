"""Optional accelerator-event and host timing."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch

from .accelerator import available_accelerator, resolve_device


class GPUTimeRecorder:
    """Events bound real compute calls; one synchronization only when flushing.

    Device elapsed time is measured on the current stream.
    No synthetic warmup runs are executed. All natural batches remain in totals.
    """
    def __init__(self, *, device=None, warmup_batches=3, metadata=None,
                 clock=time.perf_counter):
        if warmup_batches < 0:
            raise ValueError('warmup_batches must be nonnegative')
        self.device = resolve_device("auto") if device is None else torch.device(device)
        self.warmup_batches = warmup_batches
        self.metadata, self.clock = dict(metadata or {}), clock
        current = available_accelerator() if self.device.type in {"cuda", "xpu"} else None
        self.device_events = current is not None and current.type == self.device.type
        if self.device_events and self.device.type == "cuda":
            self.metadata.setdefault('torch_version', str(torch.__version__))
            self.metadata.setdefault('CUDA_version', torch.version.cuda)
        self.pending, self.rows, self.indices = [], [], {}
        self.created_at = self.clock()

    @contextmanager
    def region(self, phase, *, samples=0, is_batch=True, measure_device=True):
        if samples < 0:
            raise ValueError('samples must be nonnegative')
        index = self.indices.get(phase, 0)
        self.indices[phase] = index + 1
        row = {'phase': phase, 'samples': int(samples),
               'is_batch': is_batch, 'batch_index': index if is_batch else None,
               'natural_warmup': bool(is_batch and index < self.warmup_batches),
               'completed': False, 'device_event_ms': None}
        start = end = None
        device_timed = self.device_events and measure_device
        if device_timed:
            stream = torch.accelerator.current_stream(self.device)
            start = torch.Event(device=self.device, enable_timing=True)
            end = torch.Event(device=self.device, enable_timing=True)
            start.record(stream)
        begin = self.clock()
        try:
            yield
            row['completed'] = True
        finally:
            row['host_region_seconds'] = self.clock() - begin
            if device_timed:
                end.record(stream)
                self.pending.append((row, start, end))
            self.rows.append(row)

    def flush(self):
        if self.pending:
            torch.accelerator.synchronize(self.device)
            for row, start, end in self.pending:
                row['device_event_ms'] = float(start.elapsed_time(end))
            self.pending.clear()

    def _aggregate(self, rows):
        images = sum(r['samples'] for r in rows)
        elapsed = [r['device_event_ms'] for r in rows if r['device_event_ms'] is not None]
        ms = sum(elapsed) if elapsed else None
        wall = sum(r['host_region_seconds'] for r in rows)
        device_timings = {
            'event_seconds': None if ms is None else ms / 1000.,
            'event_ms_per_image': ms / images if ms is not None and images else None,
            'event_images_per_second': images * 1000 / ms if ms is not None and ms > 0 else None,
        }
        return {'calls': len(rows), 'samples': images,
                'completed_calls': sum(r['completed'] for r in rows),
                'failed_calls': sum(not r['completed'] for r in rows),
                'host_region_seconds': wall,
                **{f"device_{key}": value for key, value in device_timings.items()},
                **{f"cuda_{key}": value if self.device.type == "cuda" else None
                   for key, value in device_timings.items()}}

    def summary(self):
        self.flush()
        phases = {}
        for phase in sorted({r['phase'] for r in self.rows}):
            rr = [r for r in self.rows if r['phase'] == phase]
            phases[phase] = {'all_real_calls': self._aggregate(rr),
                'excluding_natural_warmup': self._aggregate([r for r in rr if not r['natural_warmup']])}
        memory = {'scope': 'process allocator peak since caller last reset;timer does not reset counters',
                  'max_allocated_bytes': None, 'max_reserved_bytes': None}
        if self.device_events:
            memory.update(max_allocated_bytes=int(torch.accelerator.max_memory_allocated(self.device)),
                          max_reserved_bytes=int(torch.accelerator.max_memory_reserved(self.device)))
        cuda_events = self.device_events and self.device.type == "cuda"
        return {'version': 1, 'measurement': 'Device event elapsed on the current stream',
            'cuda_available': cuda_events, 'device_events_available': self.device_events, 'device': str(self.device),
            'missing_cuda_reason': None if cuda_events else 'CUDA events are unavailable for the selected device',
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
