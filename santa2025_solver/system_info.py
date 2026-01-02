"""
Santa 2025 Solver - System Information Module

Detects CPU, RAM, and other system information for worker auto-sizing.
"""

import os
import sys
import platform
from typing import Dict, Optional


def get_cpu_count() -> int:
    """Get number of logical CPU cores."""
    return os.cpu_count() or 1


def get_physical_cpu_count() -> int:
    """Get number of physical CPU cores."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or get_cpu_count()
    except ImportError:
        # Fallback: try to parse /proc/cpuinfo on Linux
        if sys.platform == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    physical_ids = set()
                    core_ids = set()
                    for line in f:
                        if line.startswith('physical id'):
                            physical_ids.add(line.split(':')[1].strip())
                        elif line.startswith('core id'):
                            core_ids.add(line.split(':')[1].strip())
                    if physical_ids and core_ids:
                        return len(physical_ids) * len(core_ids)
            except:
                pass
        
        # Fallback: assume half of logical cores
        return max(1, get_cpu_count() // 2)


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        # Fallback: try to read /proc/meminfo on Linux
        if sys.platform == 'linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            kb = int(line.split()[1])
                            return kb / (1024 ** 2)
            except:
                pass
        
        # Fallback: assume 4GB available
        return 4.0


def get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
    except ImportError:
        # Fallback: try to read /proc/meminfo on Linux
        if sys.platform == 'linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb / (1024 ** 2)
            except:
                pass
        
        # Fallback: assume 8GB total
        return 8.0


def get_python_version() -> str:
    """Get Python version string."""
    return platform.python_version()


def get_system_info() -> Dict:
    """Get comprehensive system information."""
    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': get_python_version(),
        'cpu_count_logical': get_cpu_count(),
        'cpu_count_physical': get_physical_cpu_count(),
        'memory_total_gb': round(get_total_memory_gb(), 2),
        'memory_available_gb': round(get_available_memory_gb(), 2),
    }


def compute_safe_workers(
    mem_per_worker_gb: float = 0.5,
    reserve_cores: int = 1,
    max_workers: Optional[int] = None
) -> int:
    """
    Compute safe number of worker processes.
    
    Args:
        mem_per_worker_gb: Estimated memory usage per worker in GB
        reserve_cores: Number of cores to reserve for OS/orchestrator
        max_workers: Optional maximum cap on workers
    
    Returns:
        Safe number of workers to spawn
    """
    # Get system resources
    available_mem = get_available_memory_gb()
    physical_cores = get_physical_cpu_count()
    logical_cores = get_cpu_count()
    
    # Compute safe workers based on memory
    mem_based = max(1, int(available_mem / mem_per_worker_gb))
    
    # Compute safe workers based on CPU
    cpu_based = max(1, physical_cores - reserve_cores)
    
    # Take minimum of memory and CPU constraints
    safe = min(mem_based, cpu_based, logical_cores)
    
    # Apply maximum cap if specified
    if max_workers is not None:
        safe = min(safe, max_workers)
    
    return max(1, safe)


def format_system_info_markdown() -> str:
    """Format system info as Markdown for dashboard."""
    info = get_system_info()
    
    lines = [
        "## System Information",
        "",
        f"- **Platform**: {info['platform']} {info['platform_release']}",
        f"- **Architecture**: {info['architecture']}",
        f"- **Python Version**: {info['python_version']}",
        f"- **CPU Cores (Logical)**: {info['cpu_count_logical']}",
        f"- **CPU Cores (Physical)**: {info['cpu_count_physical']}",
        f"- **Total Memory**: {info['memory_total_gb']} GB",
        f"- **Available Memory**: {info['memory_available_gb']} GB",
        "",
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Print system info when run directly
    info = get_system_info()
    print("System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nSafe workers: {compute_safe_workers()}")
