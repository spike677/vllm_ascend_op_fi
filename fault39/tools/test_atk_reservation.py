"""Pure reservation policy tests; no Docker, /proc, fcntl or NPU calls."""
import ast
from pathlib import Path
import unittest

# Compile the actual pure helper, avoiding Linux-only controller imports on
# the Windows CPU test host. No rewritten policy implementation is tested.
source = Path(__file__).with_name('run_124.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'dedicated_atk_reservations')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
dedicated_atk_reservations = namespace['dedicated_atk_reservations']


def container(name, mapped=(2,), running=True, privileged=False):
    return {'Id': 'exact-container-id', 'Name': name, 'State': {'Running': running},
            'HostConfig': {'Privileged': privileged,
                           'Devices': [{'PathOnHost': f'/dev/davinci{card}'} for card in mapped]}}


class AtkReservationTests(unittest.TestCase):
    def test_running_atk_reserves_only_selected_mapped_cards(self):
        item = container('/ptq_test_dev2_slot0', (2,))
        report = dedicated_atk_reservations(item, [2, 5])
        self.assertEqual([entry['card'] for entry in report], [2])
        self.assertEqual(report[0]['reason'], 'DEDICATED_ATK_CONTAINER_RESERVATION')
        self.assertEqual(dedicated_atk_reservations(item, [5]), [])
        self.assertEqual([entry['card'] for entry in
                          dedicated_atk_reservations(container('/ptq_pair', (2, 5)), [5])], [5])

    def test_regular_persistent_or_privileged_mapping_is_not_reservation(self):
        for name in ('/gyl_test', '/persistent_shell', '/not_ptq_experiment'):
            self.assertEqual(dedicated_atk_reservations(container(name, (2, 5), privileged=True), [2, 5]), [])

    def test_stopped_atk_inspect_race_does_not_reserve(self):
        self.assertEqual(dedicated_atk_reservations(container('/ptq_finished', (2, 5), running=False), [2, 5]), [])


if __name__ == '__main__':
    unittest.main()
