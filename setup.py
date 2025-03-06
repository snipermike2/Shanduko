import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["torch", "numpy", "pandas", "matplotlib", "ttkbootstrap", "sqlalchemy"],
    "include_files": ["checkpoints/"]
}

base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="Shanduko",
    version="0.1",
    description="Water Quality Monitoring System",
    options={"build_exe": build_exe_options},
    executables=[Executable("run_shanduko.py", base=base, target_name="Shanduko.exe")]
)