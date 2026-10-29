import os
if os.environ.get('FAULT28_CALIBRATION') == '1':
    try:
        import model_hook
        model_hook.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
