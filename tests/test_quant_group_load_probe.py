import copy
import unittest

from experiments.c2c3_closure_20261117.providers_quant_group_load_probe import require_zero_state
from fault_injection.quant_group_native import RuntimeState, snapshot


class QuantGroupLoadProbeTests(unittest.TestCase):
    def setUp(self):
        self.state = snapshot(RuntimeState())
        self.state['census']['capacity'] = 32768
        self.state['binding_diagnostics'] = dict(callback_error_count=0, callback_error_overflow=0, errors=[])

    def test_loaded_uninstalled_state_is_allowed(self):
        require_zero_state(self.state)

    def test_any_native_or_callback_activity_or_owned_resource_is_rejected(self):
        for path in (('core', 'installed'), ('core', 'device_bytes'), ('core', 'host_bytes'),
                     ('core', 'borrowed_workspace_bytes'), ('census', 'pending'), ('census', 'native_executes'),
                     ('public_active',), ('native_stack_active',), ('abi_exceptions',), ('orphan_descriptors',),
                     ('binding_diagnostics', 'callback_error_count')):
            with self.subTest(path=path):
                changed = copy.deepcopy(self.state)
                parent = changed
                for name in path[:-1]:
                    parent = parent[name]
                parent[path[-1]] = 1
                with self.assertRaisesRegex(RuntimeError, 'ACTIVITY_OR_RESOURCES'):
                    require_zero_state(changed)

    def test_changed_abi_or_census_bound_is_rejected(self):
        for name, value in (('version', 2), ('size', 408)):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.state)
                changed[name] = value
                with self.assertRaisesRegex(RuntimeError, 'REVIEWED_A3'):
                    require_zero_state(changed)


if __name__ == '__main__':
    unittest.main()
