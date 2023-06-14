import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))