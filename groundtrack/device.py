"""Pick the PyTorch device: cuda (NVIDIA) > mps (Apple Silicon) > cpu."""

from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    """Return a device string usable by Ultralytics ("cuda:0", "mps" or "cpu").

    requested: "auto", "cuda", "cuda:N", "mps" or "cpu". An explicit request for an
    unavailable accelerator raises instead of silently falling back to CPU, so you
    never wait hours for a run you thought was on the GPU.
    """
    import torch

    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but PyTorch cannot see a CUDA GPU. Reinstall the CUDA "
                "build of PyTorch (see README 'Install') or set device: auto/cpu."
            )
        return req if ":" in req else "cuda:0"
    if req == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("device=mps requested but MPS is not available on this machine.")
        return "mps"
    if req == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; use auto, cuda, mps or cpu.")


def describe_device(device: str) -> str:
    import torch

    lines = [f"torch {torch.__version__}", f"selected device: {device}"]
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        lines.append(
            f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB, "
            f"compute capability {props.major}.{props.minor}, CUDA {torch.version.cuda}"
        )
    elif device == "mps":
        lines.append("Apple Silicon GPU via Metal Performance Shaders")
    else:
        lines.append("CPU only: expect roughly 1-5 frames/s with yolo26m at 1280 px")
    return "\n".join(lines)


def half_supported(device: str) -> bool:
    """FP16 inference is only a win (and only reliable) on CUDA."""
    return device.startswith("cuda")
