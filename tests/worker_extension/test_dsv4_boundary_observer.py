import unittest
from experiments.worker_core_v11_20261118.dsv4_boundary_observer import BoundaryObserver
from experiments.worker_core_v11_20261118.dsv4_probe_support import source


class Selector:
    def __init__(self):
        self.calls = 0

    def select(self, *, scores):
        self.calls += 1
        return scores


class Dispatch:
    def __init__(self, selector):
        self.selector = selector

    def route(self, values):
        return self.selector.select(scores=values)


class BoundaryTests(unittest.TestCase):
    def target(self, obj, method="select", fields=("scores",)):
        return (method, obj, method, source(getattr(obj, method))["sha256"], fields)

    def observer(self, obj, **kwargs):
        return BoundaryObserver([self.target(obj)], lambda x: x, lambda: {"step": 1, "rank": 0}, **kwargs)

    def test_preserves_result_kwargs_and_exact_method_identity(self):
        obj = Selector()
        original = obj.select
        observer = self.observer(obj)
        observer.install()
        value = [1.0, 2.0]
        self.assertIs(obj.select(scores=value), value)
        report = observer.stop()
        self.assertEqual(obj.calls, 1)
        self.assertIs(obj.select.__func__, original.__func__)
        self.assertIs(obj.select.__self__, original.__self__)
        self.assertNotIn("select", vars(obj))
        self.assertFalse(report["qualification_granted"])

    def test_nested_boundaries_keep_parent_call(self):
        obj = Selector()
        outer = Dispatch(obj)
        observer = BoundaryObserver([self.target(obj), self.target(outer, "route", ("values",))],
                                    lambda x: x, lambda: {"step": 3, "layer": "reviewed"})
        observer.install()
        outer.route([1])
        rows = observer.stop()["records"]
        self.assertEqual(rows[0]["parent_call_id"], rows[1]["call_id"])

    def test_bad_source_rejected_before_any_install(self):
        obj = Selector()
        target = list(self.target(obj))
        target[3] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE_SHA_MISMATCH"):
            BoundaryObserver([target], lambda x: x, lambda: {})
        self.assertNotIn("select", vars(obj))

    def test_wrapped_method_cannot_be_approved_by_hash(self):
        obj = Selector()
        obj.select = lambda **kwargs: kwargs
        with self.assertRaises(ValueError):
            self.observer(obj)

    def test_unknown_argument_refused(self):
        obj = Selector()
        with self.assertRaisesRegex(ValueError, "ARGUMENT_UNPROVEN"):
            BoundaryObserver([self.target(obj, fields=("hidden",))], lambda x: x, lambda: {})

    def test_budget_truncates_observation_not_model(self):
        obj = Selector()
        observer = self.observer(obj, max_calls=1)
        observer.install()
        for _ in range(3):
            self.assertEqual(obj.select(scores=[2]), [2])
        report = observer.stop()
        self.assertTrue(report["truncated"])
        self.assertEqual(obj.calls, 3)
        self.assertEqual(len(report["records"]), 1)

    def test_snapshot_error_does_not_change_result(self):
        obj = Selector()
        observer = BoundaryObserver([self.target(obj)], lambda x: 1 / 0, lambda: {})
        observer.install()
        self.assertEqual(obj.select(scores=[3]), [3])
        self.assertTrue(observer.stop()["errors"])

    def test_original_error_preserved(self):
        obj = Selector()
        observer = self.observer(obj)
        observer.install()
        with self.assertRaises(TypeError):
            obj.select(not_scores=[1])
        observer.stop()
        self.assertEqual(observer.active, 0)

    def test_foreign_wrapper_not_removed(self):
        obj = Selector()
        observer = self.observer(obj)
        observer.install()
        owned = obj.select
        foreign = lambda **kw: None
        obj.select = foreign
        with self.assertRaisesRegex(ValueError, "OWNER_CHANGED"):
            observer.stop()
        self.assertIs(obj.select, foreign)
        obj.select = owned
        observer.stop()

    def test_byte_budget(self):
        obj = Selector()
        observer = self.observer(obj, max_bytes=1)
        observer.install()
        obj.select(scores=[1])
        report = observer.stop()
        self.assertTrue(report["truncated"])
        self.assertEqual(report["records"], [])

    def test_parallel_threads_are_not_parent_and_child(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        barrier = threading.Barrier(2)
        obj = Selector()
        def context():
            barrier.wait(timeout=5)
            return {"step": 1}
        observer = BoundaryObserver([self.target(obj)], lambda x: x, context)
        observer.install()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(obj.select, scores=[n]) for n in range(2)]
            self.assertEqual([f.result() for f in futures], [[0], [1]])
        rows = observer.stop()["records"]
        self.assertEqual(len({r["thread_id"] for r in rows}), 2)
        self.assertTrue(all(r["parent_call_id"] is None for r in rows))


if __name__ == "__main__":
    unittest.main()
