"""Process setup shared by the trainer and the teacher scorer."""
import threading,functools

def setup():
    import torch
    from triton.runtime.autotuner import Autotuner
    from fla.ops.utils.cache import CachedAutotuner
    lock=threading.RLock()
    def wrap(fn):
        @functools.wraps(fn)
        def f(self,*a,**kw):
            with lock:return fn(self,*a,**kw)
        return f
    for cls in (Autotuner,CachedAutotuner):cls.run=wrap(cls.run)
    torch.set_num_threads(8);torch.manual_seed(20260911);torch.cuda.manual_seed_all(20260911)
    return torch
