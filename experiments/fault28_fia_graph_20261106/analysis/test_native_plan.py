import json
from pathlib import Path
import tempfile
import unittest
from native_plan import parse

NAME='te_fusedinferattentionscore_abcdef1234'
SELECT='[INFO] OP(22,python): [GenTilingKey][22] OpName:[FusedInferAttentionScore] FIA tilingKey_: 109000000000000000.\n'
BINARY=f'FillTaskDefAfterGenTask:Set kernel_name[{NAME}] for the kernel_def of node[FusedInferAttentionScore, FusedInferAttentionScore]\n'
LAUNCH=f'[INFO] RUNTIME(87,python): StreamLaunchKernelV2: kernel info : device_id=0, stream_id=44, task_id=0, kernel_name={NAME}, arg_size=312, coreDim=16, schemMode=1\n'
TASK='Distribute:KernelTaskInfo Distribute Success, op: FusedInferAttentionScore, taskid: 1, blockdim: 16, streamid: 44\n'


def marker(call,event):
    return json.dumps({'schema':'fault28.fia_graph.call_marker.v1','event':event,'call_id':call,
                       'pid':87,'batch':8,'layout':'BNSD','phase':'compile_warmup' if call==0 else 'graph_replay'})+'\n'


def fixture(root,extra='',missing_csv=False,launch=LAUNCH):
    text='BuildNewKernel: register every unrelated static key\n'
    for call in (0,1,2):
        text+=marker(call,'enter')
        if call==0:text+=SELECT+BINARY+launch+TASK
        text+=marker(call,'exit')
        if not (missing_csv and call==2):
            path=root/'msprobe_graph/graph_1/1'/str(call);path.mkdir(parents=True)
            (path/'FusedInferAttentionScore.FusedInferAttentionScore.1.csv').write_text('actual placeholder for native census fixture')
    path=root/'log';path.write_text(text+extra)
    return path


class NativePlanTests(unittest.TestCase):
    def test_one_instantiation_and_three_graph_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=parse(fixture(root),root)
            self.assertEqual(len(r['graph_calls']),3)
            self.assertEqual(r['selected']['jit_template_key'],'109000000000000000')

    def test_unscoped_extra_launch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):parse(fixture(root,extra=LAUNCH),root)

    def test_missing_actual_iteration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):parse(fixture(root,missing_csv=True),root)

    def test_foreign_process_launch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):parse(fixture(root,launch=LAUNCH.replace('RUNTIME(87,','RUNTIME(88,')),root)

    def test_registration_without_actual_launch_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):parse(fixture(root,launch='BuildNewKernel: '+NAME+' registered\n'),root)


if __name__=='__main__':unittest.main()
