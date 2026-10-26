"""Lazy observer bootstrap. No torch/vendor imports or configuration changes here."""
import os
if os.environ.get("AS_STRIDED_OBS_ENABLED") == "1":
    try:
        import passive_view_observer
        if "MSPROBE_CAPTURE_ENABLED" in os.environ:
            import msprobe_bridge
            msprobe_bridge.install(passive_view_observer)
        passive_view_observer.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        if "MSPROBE_CAPTURE_ENABLED" in os.environ:
            # site.py normally swallows sitecustomize exceptions; terminate only
            # this independent test process instead of running unobserved.
            os._exit(78)
