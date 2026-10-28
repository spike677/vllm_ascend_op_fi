import os
if os.environ.get('DCAT_V2_HOOK')=='1':
    try:
        import model_hook
        model_hook.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
