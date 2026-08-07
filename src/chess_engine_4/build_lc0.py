"""Build the vendored lc0 fork with the chess-engine-4 inference backend."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def build_lc0() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels-build-dir", type=Path, default=Path("kernels/build/inference"))
    parser.add_argument("--lc0-build-type", choices=("debug", "release"), default="release")
    args = parser.parse_args()

    root = _project_root()
    kernels_dir = root / "kernels"
    kernels_build_dir = (root / args.kernels_build_dir).resolve()
    lc0_dir = root / "third_party" / "lc0"
    env = os.environ.copy()

    subprocess.run(
        [
            "cmake",
            "-S",
            str(kernels_dir),
            "-B",
            str(kernels_build_dir),
            "-G",
            "Ninja",
            "-DCE4_BUILD_PYTHON=OFF",
            "-DCE4_BUILD_INFERENCE=ON",
            "-DCE4_CUDA_ARCH=100a",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        ["cmake", "--build", str(kernels_build_dir), "--parallel"],
        check=True,
        env=env,
    )

    subprocess.run(
        [
            str(lc0_dir / "build.sh"),
            args.lc0_build_type,
            "-Dgtest=false",
            "-Donnx=false",
            "-Dcudnn=false",
            "-Dplain_cuda=true",
            "-Dcutlass=false",
            "-Dnative_arch=false",
            "-Dnative_cuda=false",
            "-Dcc_cuda=100",
            "-Ddefault_backend=ce4",
            f"-Dce4_kernels_dir={kernels_dir}",
        ],
        cwd=lc0_dir,
        check=True,
        env=env,
    )
    print(lc0_dir / "build" / args.lc0_build_type / "lc0")


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "kernels" / "CMakeLists.txt").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    build_lc0()
