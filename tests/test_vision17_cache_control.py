import unittest
from experiments.worker_core_v11_20261118.vision17_cache_control import IDLE_PHASES, reset_idle_mm_cache


class PublicLLM:
    def __init__(self):
        self.llm_engine = self
        self.pending = False
        self.calls = 0
        self.result = None
    def has_unfinished_requests(self):
        return self.pending
    def reset_mm_cache(self):
        self.calls += 1
        return self.result


class CacheControlTests(unittest.TestCase):
    def test_public_reset_is_not_encoder_proof(self):
        llm = PublicLLM()
        row = reset_idle_mm_cache(llm, 'healthy_1')
        self.assertEqual(llm.calls, 1)
        self.assertFalse(row['encoder_execution_proven'] or row['qualification_granted'])
    def test_pending_request_cannot_be_reset(self):
        llm = PublicLLM(); llm.pending = True
        with self.assertRaisesRegex(RuntimeError, 'IDLE_DECLARED'):
            reset_idle_mm_cache(llm, 'healthy_1')
        self.assertEqual(llm.calls, 0)
    def test_wrong_phase_cannot_reset(self):
        llm = PublicLLM()
        with self.assertRaisesRegex(RuntimeError, 'IDLE_DECLARED'):
            reset_idle_mm_cache(llm, 'fault')
        self.assertEqual(llm.calls, 0)
    def test_changed_public_return_is_not_acknowledged(self):
        llm = PublicLLM(); llm.result = True
        with self.assertRaisesRegex(RuntimeError, 'RETURN_OR_IDLE'):
            reset_idle_mm_cache(llm, 'healthy_0')

    def test_declared_group_phases_are_preserved_and_never_qualify(self):
        for phase in sorted(IDLE_PHASES):
            with self.subTest(phase=phase):
                llm = PublicLLM()
                result = reset_idle_mm_cache(llm, phase)
                self.assertEqual(result['phase'], phase)
                self.assertEqual(llm.calls, 1)
                self.assertFalse(result['encoder_execution_proven'])
                self.assertFalse(result['qualification_granted'])

    def test_all_group_phases_refuse_while_request_pending(self):
        for phase in sorted(IDLE_PHASES):
            with self.subTest(phase=phase):
                llm = PublicLLM(); llm.pending = True
                with self.assertRaisesRegex(RuntimeError, 'IDLE_DECLARED'):
                    reset_idle_mm_cache(llm, phase)
                self.assertEqual(llm.calls, 0)


if __name__ == '__main__':
    unittest.main()
