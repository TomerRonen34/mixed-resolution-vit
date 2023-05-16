import os
import sys
from pathlib import Path

cache_path = Path("cache_path.txt")
if cache_path.exists():
    os.environ["XDG_CACHE_HOME"] = cache_path.read_text()

if '..' not in sys.path:
    sys.path.append('..')
