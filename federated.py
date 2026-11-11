"""Single-experiment root entry; forward arguments to the original src program."""
from pathlib import Path
import os
import runpy
import sys

if __name__ == '__main__':
    source = Path(__file__).resolve().parent/'src'
    sys.path.insert(0, str(source))
    os.chdir(source)
    runpy.run_path(str(source/'federated.py'), run_name='__main__')
