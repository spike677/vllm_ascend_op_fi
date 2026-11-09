"""Shared-input observations: retain duplicates, reject conflicting time states."""
from pathlib import Path
import tempfile
import unittest
from graph_evidence import observed_rows

HEADER='Input/Output,Index,Data Size,Data Type,Format,Shape,Max Value,Min Value,Avg Value,Count,Nan Count,Negative Inf Count,Positive Inf Count\n'
ROW='Output,0,2,DT_FLOAT16,ND,1,1,1,1,1,0,0,0\n'


class RepeatedCSVTests(unittest.TestCase):
    def make(self,root,model,tag,row=ROW):
        p=root/'msprobe_graph/raw/graph_1'/str(model)/'0'/('Data.shared.'+tag+'.csv')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(HEADER+row);return p

    def test_same_observed_rows_preserve_both_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make(root,1,'a');self.make(root,1,'b')
            rows,source=observed_rows(root,{'name':'shared','type':'Data'},0)
            self.assertEqual(rows[('Output',0)]['AvgValue'],'1')
            self.assertEqual(len(source['observations']),2)

    def test_conflicting_observations_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make(root,1,'a');self.make(root,1,'b',ROW.replace(',1,1,1,1,',',2,1,1.5,1,'))
            with self.assertRaises(ValueError):observed_rows(root,{'name':'shared','type':'Data'},0)

    def test_different_execution_groups_remain_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make(root,1,'a');self.make(root,2,'b')
            with self.assertRaises(ValueError):observed_rows(root,{'name':'shared','type':'Data'},0)


if __name__=='__main__':unittest.main()
