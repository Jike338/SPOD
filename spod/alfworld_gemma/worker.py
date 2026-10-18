import os,json
from pathlib import Path
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f".{os.getpid()}.tmp");tmp.write_text(json.dumps(value));tmp.replace(p)
