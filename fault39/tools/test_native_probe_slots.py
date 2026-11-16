"""Pure CPU tests of probe proof checks. No torch import and no NPU initialization."""
import copy
import unittest
from native_probe import verify_dynamic_slot_proof


def snapshot(mask=3, observer=False):
    original = [4096, 8192, 12288]
    run = list(original)
    if not observer:
        for index in range(2):
            if mask & (1 << index):
                run[index] += 65536
    return {"owner": "probe", "generation": 7,
            "last_observation": {"owner": "probe", "generation": 7,
                                 "proof": {"api": "aclnnMatmul", "generation": 7,
                                           "slots": [0, 1, 2], "formal_order_proven": True,
                                           "input_mask": mask,
                                           "tensors": [{"original_address": a, "run_address": b}
                                                       for a, b in zip(original, run)]}}}


class DynamicSlots(unittest.TestCase):
    addresses = [4096, 8192, 12288]

    def test_observer_changes_no_address(self):
        result = verify_dynamic_slot_proof(snapshot(observer=True), self.addresses, self.addresses,
                                           3, observer_only=True)
        self.assertTrue(result["observer_only"])
        self.assertFalse(result["buffer_substitution"])

    def test_all_lossless_masks(self):
        for mask in (1, 2, 3):
            self.assertEqual(verify_dynamic_slot_proof(snapshot(mask), self.addresses,
                                                      self.addresses, mask)["status"], "PASS")

    def test_rejects_wrong_private_order(self):
        value = snapshot(observer=True)
        tensors = value["last_observation"]["proof"]["tensors"]
        tensors[0], tensors[1] = tensors[1], tensors[0]
        with self.assertRaisesRegex(RuntimeError, "ADDRESS_MISMATCH"):
            verify_dynamic_slot_proof(value, self.addresses, self.addresses, 3, observer_only=True)

    def test_rejects_observer_substitution(self):
        with self.assertRaisesRegex(RuntimeError, "REPLACEMENT_MISMATCH"):
            verify_dynamic_slot_proof(snapshot(), self.addresses, self.addresses, 3, observer_only=True)

    def test_rejects_changed_output_storage(self):
        value = snapshot()
        value["last_observation"]["proof"]["tensors"][2]["run_address"] += 1
        with self.assertRaisesRegex(RuntimeError, "REPLACEMENT_MISMATCH"):
            verify_dynamic_slot_proof(value, self.addresses, self.addresses, 3)

    def test_rejects_changed_live_storage(self):
        with self.assertRaisesRegex(RuntimeError, "ADDRESSES_INVALID_OR_CHANGED"):
            verify_dynamic_slot_proof(snapshot(), self.addresses, [4096, 8192, 65536], 3)

    def test_rejects_stale_generation(self):
        value = copy.deepcopy(snapshot())
        value["last_observation"]["proof"]["generation"] = 6
        with self.assertRaisesRegex(RuntimeError, "PROOF_INCOMPLETE"):
            verify_dynamic_slot_proof(value, self.addresses, self.addresses, 3)


if __name__ == "__main__":
    unittest.main()
