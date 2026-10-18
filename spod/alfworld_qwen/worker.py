"""Atomic JSON writes shared by all pipeline processes."""
import os,json
from pathlib import Path

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x));t.replace(p)
