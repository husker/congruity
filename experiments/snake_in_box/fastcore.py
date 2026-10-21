"""ctypes bridge to snake_core.c — compiles on first import, caches the .so.

Exposes two accelerated primitives with Python-identical semantics:
  rollout_kinny(n, seed, budget)  -> list[int]      (greedy Kinny rollout)
  complete(n, prefix, node_cap, rng_seed=-1, forbid_first=-1)
                                  -> (best_seq, nodes, exhausted)

`CORE_AVAILABLE` is False (with `LOAD_ERROR` set) if no C compiler is present
or the build fails; callers fall back to the pure-Python reference. The build
is keyed by source hash, so editing snake_core.c triggers a rebuild.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sysconfig
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "snake_core.c"

CORE_AVAILABLE = False
LOAD_ERROR: str | None = None
_LIB = None


def _build_and_load():
    global CORE_AVAILABLE, LOAD_ERROR, _LIB
    src = _SRC.read_text()
    tag = hashlib.sha256(src.encode()).hexdigest()[:12]
    ext = ".dylib" if sysconfig.get_platform().startswith("macos") else ".so"
    so = _HERE / f"_snake_core_{tag}{ext}"
    if not so.exists():
        # clean stale builds from previous source versions
        for old in _HERE.glob("_snake_core_*"):
            old.unlink()
        cc = sysconfig.get_config_var("CC") or "cc"
        cc = cc.split()[0]
        base = [cc, "-O3", "-shared", "-fPIC", str(_SRC), "-o", str(so)]
        # -march=native fails on Apple clang (wants -mcpu); try tuned flags,
        # then fall back to a portable -O3 build.
        for tune in (["-march=native"], ["-mcpu=native"], []):
            try:
                subprocess.run(base[:2] + tune + base[2:],
                               check=True, capture_output=True, timeout=120)
                break
            except subprocess.CalledProcessError as e:
                LOAD_ERROR = f"CalledProcessError: {e.stderr!r}"
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                LOAD_ERROR = f"{type(e).__name__}: {getattr(e, 'stderr', b'')!r}"
                return
        if not so.exists():
            return
    LOAD_ERROR = None  # a tune flag may have failed before one succeeded
    lib = ctypes.CDLL(str(so))
    lib.sib_rollout.restype = ctypes.c_int
    lib.sib_rollout.argtypes = [ctypes.c_int, ctypes.c_uint64,
                                ctypes.c_longlong, ctypes.POINTER(ctypes.c_int)]
    lib.sib_complete.restype = ctypes.c_int
    lib.sib_complete.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int)]
    lib.sib_complete_blocked.restype = ctypes.c_int
    lib.sib_complete_blocked.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int)]
    lib.sib_complete_frozen.restype = ctypes.c_int
    lib.sib_complete_frozen.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int)]
    _LIB = lib
    CORE_AVAILABLE = True


try:
    _build_and_load()
except Exception as e:  # never let an import kill the caller
    LOAD_ERROR = f"{type(e).__name__}: {e}"


def rollout_kinny(n: int, seed: int, budget: int) -> list[int]:
    cap = (1 << n) + 1
    out = (ctypes.c_int * cap)()
    length = _LIB.sib_rollout(n, ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
                              ctypes.c_longlong(budget), out)
    return list(out[: length + 1])


def complete_blocked(n: int, prefix: list[int], blocked: list[int],
                     node_cap: int, rng_seed: int = -1):
    """Longest induced path extending `prefix` while avoiding `blocked`
    (complement-constrained search — Wynn-style joint construction)."""
    cap = (1 << n) + 1
    parr = (ctypes.c_int * len(prefix))(*prefix)
    barr = (ctypes.c_int * max(1, len(blocked)))(*(blocked or [0]))
    out = (ctypes.c_int * cap)()
    nodes = ctypes.c_longlong(0)
    exhausted = ctypes.c_int(0)
    length = _LIB.sib_complete_blocked(
        n, parr, len(prefix), barr, len(blocked),
        ctypes.c_longlong(node_cap), ctypes.c_longlong(rng_seed),
        out, ctypes.byref(nodes), ctypes.byref(exhausted))
    return list(out[: length + 1]), int(nodes.value), bool(exhausted.value)


def complete_frozen(n: int, prefix: list[int], frozen: list[int],
                    blocked: list[int], node_cap: int, rng_seed: int = -1):
    """Longest induced path extending `prefix`: `frozen` = committed
    same-layer snake vertices (excluded + neighbor counts — adjacency to
    them is a chord); `blocked` = label-only exclusions (other layer's
    shadow). The fragment router's expansion primitive."""
    cap = (1 << n) + 1
    parr = (ctypes.c_int * len(prefix))(*prefix)
    farr = (ctypes.c_int * max(1, len(frozen)))(*(frozen or [0]))
    barr = (ctypes.c_int * max(1, len(blocked)))(*(blocked or [0]))
    out = (ctypes.c_int * cap)()
    nodes = ctypes.c_longlong(0)
    exhausted = ctypes.c_int(0)
    length = _LIB.sib_complete_frozen(
        n, parr, len(prefix), farr, len(frozen), barr, len(blocked),
        ctypes.c_longlong(node_cap), ctypes.c_longlong(rng_seed),
        out, ctypes.byref(nodes), ctypes.byref(exhausted))
    return list(out[: length + 1]), int(nodes.value), bool(exhausted.value)


def complete(n: int, prefix: list[int], node_cap: int,
             rng_seed: int = -1, forbid_first: int = -1):
    cap = (1 << n) + 1
    parr = (ctypes.c_int * len(prefix))(*prefix)
    out = (ctypes.c_int * cap)()
    nodes = ctypes.c_longlong(0)
    exhausted = ctypes.c_int(0)
    length = _LIB.sib_complete(
        n, parr, len(prefix), ctypes.c_longlong(node_cap),
        ctypes.c_longlong(rng_seed), ctypes.c_int(forbid_first),
        out, ctypes.byref(nodes), ctypes.byref(exhausted))
    return list(out[: length + 1]), int(nodes.value), bool(exhausted.value)
