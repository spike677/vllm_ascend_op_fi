"""Coexistence contracts; only actual-image CPU/NPU evidence validates ELFs."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import paged33_loading as loading


class LibraryCoexistenceTests(unittest.TestCase):
    def test_only_exact_private_then_stock_order_is_accepted(self):
        private='/private/vendor'
        with patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':':'.join(loading.vendor_order(private)),
                                    'LD_LIBRARY_PATH':'/sdk/lib:/stock/lib'},clear=True):
            loading.validate_environment(private,platform_ready=True)
            for value in (private,str(loading.STOCK_VENDOR)+':'+private,
                          ':'.join(loading.vendor_order(private))+':/foreign/vendor'):
                os.environ['ASCEND_CUSTOM_OPP_PATH']=value
                with self.assertRaisesRegex(RuntimeError,'EXACT_STOCK_PRIVATE_VENDOR_ORDER_REQUIRED'):
                    loading.validate_environment(private,platform_ready=True)

    def test_global_private_directory_is_rejected_even_with_correct_vendor_order(self):
        private='/private/vendor'
        with patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':private,
                                    'LD_LIBRARY_PATH':private+'/op_api/lib:/sdk/lib'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'MUST_NOT_SHADOW_STOCK_SONAME'):
                loading.validate_environment(private,platform_ready=False)

    def test_actual_symbol_owner_must_match_role_not_only_symbol_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stock=root/'stock';private=root/'private'
            bundle=dict(vendor=str(private),artifacts={loading.PRIVATE_LIB:dict(sha256='private')})
            owner=loading.LibraryCoexistence(bundle)
            owner.stock=stock/'op_api/lib/libcust_opapi.so'
            owner.stock.parent.mkdir(parents=True);owner.stock.write_text('stock')
            owner.private.parent.mkdir(parents=True);owner.private.write_text('private')
            owner.handles={'stock':SimpleNamespace(_handle=1),'bare':SimpleNamespace(_handle=1),
                           'private':SimpleNamespace(_handle=2)}
            def symbols(lib,symbol):
                path=owner.stock if lib._handle==1 else owner.private
                return dict(symbol=symbol,path=str(path.resolve()),sha256='stock' if lib._handle==1 else 'private')
            with patch.object(loading,'STOCK_ROOT',root), patch.object(loading,'STOCK_FILES',
                {owner.stock.relative_to(root).as_posix():'stock'}), \
                patch.object(owner,'check_files'),patch.object(loading,'symbol_owner',side_effect=symbols), \
                patch.object(loading.ctypes,'CDLL',return_value=owner.handles['bare']), \
                patch.object(loading.os,'RTLD_LOCAL',0,create=True),patch.object(loading.os,'RTLD_NOW',2,create=True):
                self.assertTrue(owner.inspect_bindings()['distinct_library_handles'])
                owner.handles['private']=owner.handles['stock']
                with self.assertRaisesRegex(RuntimeError,'PRIVATE_FIA_SYMBOL_BOUND_TO_STOCK'):
                    owner.inspect_bindings()
                owner.handles['bare']=SimpleNamespace(_handle=2)
                with self.assertRaisesRegex(RuntimeError,'BARE_CUSTOM_LIBRARY_BINDING_CHANGED'):
                    owner.inspect_bindings()

    def test_runtime_cannot_substitute_an_unbootstrapped_vendor(self):
        with patch.object(loading,'_ACTIVE',None):
            with self.assertRaisesRegex(RuntimeError,'BOOTSTRAP_MISSING_OR_DIFFERENT'):
                loading.verify_runtime(dict(vendor='/private/vendor',artifacts={}))


if __name__=='__main__':
    unittest.main()
