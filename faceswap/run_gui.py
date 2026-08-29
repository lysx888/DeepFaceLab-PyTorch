import os
import sys

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

_dll_dirs = os.environ.get("DFL_DLL_DIRS", "")
if _dll_dirs:
    for d in _dll_dirs.split(os.pathsep):
        d = d.strip()
        if d and os.path.isdir(d):
            os.add_dll_directory(d)

_root = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_root)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from faceswap.gui import main

if __name__ == "__main__":
    main()
