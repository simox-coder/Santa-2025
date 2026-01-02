"""
Santa 2025 Solver - Collision Backends Package

Provides multiple collision detection backends:
- python: Pure Python/NumPy implementation (baseline)
- numba: JIT-compiled implementation (primary target)
- cpp: C++ with pybind11 (optional high-performance)
"""

from santa2025_solver.collision_backends.python_backend import (
    PythonCollisionBackend,
    check_polygon_overlap_python,
)

# Try to import numba backend
try:
    from santa2025_solver.collision_backends.numba_backend import (
        NumbaCollisionBackend,
        check_polygon_overlap_numba,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    NumbaCollisionBackend = None
    check_polygon_overlap_numba = None

# Try to import C++ backend
try:
    from santa2025_solver.collision_backends.cpp_backend import (
        CppCollisionBackend,
        check_polygon_overlap_cpp,
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    CppCollisionBackend = None
    check_polygon_overlap_cpp = None


__all__ = [
    'PythonCollisionBackend',
    'NumbaCollisionBackend',
    'CppCollisionBackend',
    'NUMBA_AVAILABLE',
    'CPP_AVAILABLE',
    'get_backend',
]


def get_backend(name: str = 'auto'):
    """
    Get collision backend by name.
    
    Args:
        name: 'auto', 'python', 'numba', or 'cpp'
    
    Returns:
        Backend class
    """
    if name == 'python':
        return PythonCollisionBackend
    elif name == 'numba':
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba backend not available")
        return NumbaCollisionBackend
    elif name == 'cpp':
        if not CPP_AVAILABLE:
            raise ImportError("C++ backend not available")
        return CppCollisionBackend
    elif name == 'auto':
        # Auto-select best available
        if CPP_AVAILABLE:
            return CppCollisionBackend
        elif NUMBA_AVAILABLE:
            return NumbaCollisionBackend
        else:
            return PythonCollisionBackend
    else:
        raise ValueError(f"Unknown backend: {name}")
