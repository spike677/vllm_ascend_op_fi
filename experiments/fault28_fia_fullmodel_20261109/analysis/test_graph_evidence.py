"""Negative localization checks: upstream changes, missing data, null and empty."""
import csv
from pathlib import Path
import tempfile
import unittest
from graph_evidence import ops, compare_row, ge_compare, infer_first
from statistics_values import compare_values

GRAPH='''name: "graph_1"
op {
  name: "q"
  type: "Data"
  attr { key: "label" value { s: "escaped \\" { }" } }
}
op {
  name: "operation_without_expected_fault_name"
  type: "Operator"
  input: "q:0"
}
op {
  name: "NetOutput"
  type: "NetOutput"
  input: "operation_without_expected_fault_name:0"
}
'''


def row(mean='1',size='32'):
    return {'DataSize':size,'DataType':'DT_FLOAT16','Format':'ND','Shape':'16','Count':'16',
            'MaxValue':'2','MinValue':'0','AvgValue':mean,
            'NanCount':'0','NegativeInfCount':'0','PositiveInfCount':'0'}


def fixture(root,mean='1',data_mean='1',missing=False):
    ge=root/'msprobe_graph';ge.mkdir(parents=True)
    (ge/'dynamo_optimized_graph_1.txt').write_text(GRAPH)
    directory=ge/'graph_1/1/0';directory.mkdir(parents=True)
    for node,kind,avg in [('q','Data',data_mean),('operation_without_expected_fault_name','Operator',mean)]:
        if missing and node=='q':continue
        values={'Input/Output':'Output','Index':'0',**row(avg)}
        with (directory/(kind+'.'+node+'.1.1.1.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values));writer.writeheader();writer.writerow(values)


class LocalizationTests(unittest.TestCase):
    def test_parser_handles_quoted_braces(self):
        self.assertEqual([n['name'] for n in ops(GRAPH)],['q','operation_without_expected_fault_name','NetOutput'])

    def test_broken_dependency_rejected(self):
        with self.assertRaises(ValueError):ops(GRAPH.replace('q:0','missing:0'))

    def test_changed_input_cannot_be_assigned_to_target(self):
        with tempfile.TemporaryDirectory() as directory:
            b,f=Path(directory)/'B',Path(directory)/'F'
            fixture(b);fixture(f,mean='4',data_mean='3')
            report=ge_compare(b,f,0)
            self.assertEqual(report['first_observed_difference']['node'],'q')

    def test_operator_agnostic_first(self):
        with tempfile.TemporaryDirectory() as directory:
            b,f=Path(directory)/'B',Path(directory)/'F'
            fixture(b);fixture(f,mean='3')
            report=ge_compare(b,f,0)
            self.assertEqual(report['first_observed_difference']['type'],'Operator')

    def test_missing_upstream_is_gap_not_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            b,f=Path(directory)/'B',Path(directory)/'F'
            fixture(b,missing=True);fixture(f,mean='3',missing=True)
            report=ge_compare(b,f,0)
            self.assertEqual(report['status'],'FIRST_OBSERVED_WITH_GAPS')
            self.assertEqual(report['missing_nodes_before_first'],['q'])

    def test_no_difference_retained(self):
        self.assertFalse(compare_row(row(),row())['differences'])

    def test_null_is_not_equal(self):
        a={'type':'torch.Tensor','dtype':'torch.float16','shape':[16],
           'Max':None,'Min':None,'Mean':None,'Norm':None}
        report=compare_values(a,a)
        self.assertEqual(len(report['unobservable']),4)
        self.assertEqual(report['matching_finite_statistics'],0)

    def test_empty_tensor_stale_statistics_ignored(self):
        result=compare_row(row('123','0'),row('456','0'))
        self.assertFalse(result['differences']);self.assertTrue(result['unobservable'])

    def test_alias_inference_uses_topology_not_expected_name(self):
        nodes=ops(GRAPH)
        aliases={'output_edges':[{'module_output_slot':0,'producer':'operation_without_expected_fault_name',
                                 'producer_type':'Operator','producer_output_index':0,
                                 'edge':'operation_without_expected_fault_name:0'}]}
        mix={'first_observed_difference':{'node':'Module.arbitrary','boundary':'output',
                                          'differences':[{'path':'$[0].Mean','B':1,'F':2}]}}
        r=infer_first(nodes,{'first_observed_difference':None},mix,aliases)
        self.assertEqual(r['first']['type'],'Operator')

    def test_earlier_data_difference_overrides_output_alias(self):
        nodes=ops(GRAPH)
        aliases={'output_edges':[{'module_output_slot':0,'producer':'operation_without_expected_fault_name',
                                 'producer_type':'Operator','producer_output_index':0,
                                 'edge':'operation_without_expected_fault_name:0'}]}
        mix={'first_observed_difference':{'node':'Module.arbitrary','boundary':'output',
                                          'differences':[{'path':'$[0].Mean','B':1,'F':2}]}}
        ge={'first_observed_difference':{'node':'q','type':'Data','boundary':'Output','index':0,
                                        'differences':[{'field':'AvgValue','B':1,'F':2}]}}
        self.assertEqual(infer_first(nodes,ge,mix,aliases)['first']['node'],'q')

    def test_graph_boundary_input_difference_blocks_producer_claim(self):
        r=infer_first(ops(GRAPH),{'first_observed_difference':None},
                      {'first_observed_difference':{'boundary':'input','differences':[{'path':'$[0].Mean'}]}},
                      {'output_edges':[]})
        self.assertEqual(r['status'],'OBSERVED_GRAPH_BOUNDARY_INPUT_DIFFERENCE')


if __name__=='__main__':unittest.main()
