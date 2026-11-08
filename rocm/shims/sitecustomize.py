import os
_p = os.environ.get("PIXAL_FH")
if _p:
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("PIXAL_FH_S", "60")), repeat=True, file=open(_p, "w"))
