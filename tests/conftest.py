"""Collected before every test module (pytest loads this conftest first): the
stale-shadow guard.

The dev-loop landmine it exists for: an early non-abi3 ``maturin develop``
leaves ``python/tors/_tors.cpython-*.so`` sitting next to the fresh
``_tors.abi3.so``, and CPython's extension-suffix order binds the
version-specific .so FIRST, so a plain ``pytest`` silently tests an old
extension (a stale shadow exports only ``normalize``/``finalize``) while
the collection looks healthy. This guard fails collection LOUDLY instead:
under the canonical import the module's ``__file__`` is the abi3 artifact; a
version-specific shadow makes it end in something else, which is exactly what
is asserted.

The fix, named in the failure message: ``rm python/tors/_tors.cpython-*.so``
(or ``make dev``, which removes such shadows before reinstalling, leaving
exactly one fresh ``_tors.abi3.so``).

Design note: a harness that PRELOADS the abi3 extension under the
``tors._tors`` name (importlib straight from ``_tors.abi3.so``, before any
``import tors``) passes this guard by construction: preloading IS binding
the abi3 build, and the guard's job is to make every other invocation
resolve the abi3 artifact too.
"""

from __future__ import annotations

import tors._tors as _tors

if not str(getattr(_tors, "__file__", "")).endswith(".abi3.so"):
    raise AssertionError(
        f"tors._tors resolved to {getattr(_tors, '__file__', None)!r}: a "
        "version-specific (non-abi3) extension, i.e. a STALE SHADOW outranking "
        "the fresh python/tors/_tors.abi3.so (CPython's extension-suffix order "
        "binds _tors.cpython-*.so first, so every test in this run would "
        "execute against an old extension). Fix: rm "
        "python/tors/_tors.cpython-*.so (or `make dev`, which removes such "
        "shadows before reinstalling), then re-run."
    )
