"""Map real serialized list wrappers, never trim indices to fit a target."""
import unittest
from graph_evidence import output_tensor_paths,infer_first


class OutputTreeTests(unittest.TestCase):
    def test_flat_and_nested_raw_output_paths(self):
        tensor={'type':'torch.Tensor','dtype':'torch.float16','shape':[1]}
        self.assertEqual(output_tensor_paths([tensor,tensor]),['$[0]','$[1]'])
        self.assertEqual(output_tensor_paths([[tensor,tensor]]),['$[0][0]','$[0][1]'])

    def test_unknown_scalar_container_is_rejected(self):
        with self.assertRaises(ValueError):output_tensor_paths([42])

    def test_nested_output_uses_real_edge_not_tuple_order(self):
        nodes=[{'name':'Data','type':'Data','inputs':[]},
               {'name':'attention','type':'FusedInferAttentionScore','inputs':['Data:0']},
               {'name':'logits','type':'MatMul','inputs':['attention:0']}]
        aliases={'output_edges':[{'module_output_slot':0,'producer':'logits','producer_type':'MatMul',
            'producer_output_index':0,'edge':'logits:0','serialized_output_path':'$[0][0]'},
            {'module_output_slot':1,'producer':'attention','producer_type':'FusedInferAttentionScore',
            'producer_output_index':0,'edge':'attention:0','serialized_output_path':'$[0][1]'}]}
        mix={'first_observed_difference':{'node':'Module.graph_region.GraphBoundary.forward.0',
            'boundary':'output','differences':[{'path':'$[0][0].Mean','B':1,'F':2},
                                             {'path':'$[0][1].Norm','B':3,'F':4}]}}
        r=infer_first(nodes,{'first_observed_difference':None},mix,aliases)
        self.assertEqual(r['first']['node'],'attention')
        mix['first_observed_difference']['differences'][0]['path']='$[0][9].Mean'
        with self.assertRaises(ValueError):infer_first(nodes,{'first_observed_difference':None},mix,aliases)


if __name__=='__main__':unittest.main()
