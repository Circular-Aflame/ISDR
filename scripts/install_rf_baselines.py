#!/usr/bin/env python3
"""Install the compatible RF baseline snapshots used by the experiments."""

import importlib.metadata
import subprocess
import sys

from packaging.version import Version


SOURCES = {
    "rfgap": ("jakerhodes/RF-GAP-Python", "1bdb4f8688e587b07df1bd35e07463ce1b889b42"),
    "rfphate": ("jakerhodes/RF-PHATE", "7dde22349f0342ddf3dbc4a446e358bfcf8fed17"),
    "rfae": ("JakeSRhodesLab/RF-AE", "1506fa81d667b468bd1bbccadacdc877d531636d"),
}


def main():
    for name in ["phate", "graphtools", "seaborn"]:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit("Install dependencies first: pip install -e '.[baselines]'")
    if Version(importlib.metadata.version("scikit-learn")) >= Version("1.9"):
        raise SystemExit("RF-GAP requires scikit-learn<1.9. Install: pip install -e '.[baselines]'")
    # RF-AE's unpinned dependency otherwise installs the incompatible new RF-PHATE API.
    packages = [f"{name} @ git+https://github.com/{repo}@{commit}" for name, (repo, commit) in SOURCES.items()]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *packages, *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
