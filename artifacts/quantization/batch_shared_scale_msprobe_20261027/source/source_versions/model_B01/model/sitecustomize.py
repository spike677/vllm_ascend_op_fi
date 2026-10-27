import os
if os.environ.get('BATCH_OUTLIER_HOOK')=='1':
    try:
        import native_capture_hook
        native_capture_hook.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
