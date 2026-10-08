import random
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(preferred_gpu):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if preferred_gpu < 0 or preferred_gpu >= gpu_count:
            raise ValueError(f"GPU index {preferred_gpu} is unavailable. Detected GPUs: {gpu_count}")
        device = torch.device(f"cuda:{preferred_gpu}")
        print(f"Using GPU {preferred_gpu}: {torch.cuda.get_device_name(preferred_gpu)}")
        return device

    print("CUDA is unavailable. Using CPU.")
    return torch.device("cpu")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class TeeStream:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    @property
    def encoding(self):
        return self.terminal.encoding


@contextmanager
def capture_console(log_path, mode):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open(mode, encoding="utf-8", buffering=1) as log_file:
        stdout_stream = TeeStream(sys.stdout, log_file)
        stderr_stream = TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            yield
