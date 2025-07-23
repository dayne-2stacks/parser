import logging
import subprocess
from datetime import datetime
from typing import Optional

try:
    import torch
except Exception:
    torch = None

# Configure a module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    # handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    file_handler = logging.FileHandler("logs/gpu.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def log_gpu_memory_nvidia_smi(message: str = "") -> None:
    """Log GPU memory usage using nvidia-smi."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        used, total = result.stdout.strip().split(',')
        logger.info(f"[{timestamp}] GPU Memory {message}: Used={used.strip()}MB, Total={total.strip()}MB")
    except Exception as e:
        logger.warning(f"[{timestamp}] Unable to query GPU memory with nvidia-smi: {e}")

def log_cuda_memory_pytorch(message: str = "") -> None:
    """Log GPU memory usage using PyTorch if available."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if torch is None or not torch.cuda.is_available():
        logger.info(f"[{timestamp}] CUDA not available. {message}")
        return
    try:
        total, reserved = torch.cuda.mem_get_info()
        used = reserved / (1024**2)
        total_mb = total / (1024**2)
        free = total_mb - used
        logger.info(
            f"[{timestamp}] CUDA Memory {message}: Used={used:.2f}MB, Free={free:.2f}MB, Total={total_mb:.2f}MB"
        )
    except Exception as e:
        logger.warning(f"[{timestamp}] Error querying CUDA memory: {e}")

def flush_cuda_cache() -> None:
    """Attempt to free unused GPU memory."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Flushed CUDA cache")

