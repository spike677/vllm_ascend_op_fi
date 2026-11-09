"""Reject incomplete per-layer evidence even when a shared kernel exists."""
import copy
import unittest
from coverage import verify_census


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.names={'FusedInferAttentionScore'+('_'+str(i) if i else '') for i in range(28)}
        self.native={'windows':[{'pid':46,'call_id':i} for i in range(3)],
            'binaries':[{'node':n,'kernel':'shared'} for n in self.names],
            'tasks':[{'node':n,'block_dim':16} for n in self.names],
            'launches':[{'kernel':'shared','pid':46,'call_id':0,'block_dim':16} for _ in self.names]}

    def test_one_shared_binary_covers_all_nodes_only_with_complete_bindings(self):
        verify_census(self.native,self.names,46)

    def test_one_missing_layer_is_rejected(self):
        self.native['tasks'].pop()
        with self.assertRaises(ValueError):verify_census(self.native,self.names,46)

    def test_duplicate_layer_cannot_replace_missing_layer(self):
        self.native['binaries'][0]=dict(self.native['binaries'][1])
        with self.assertRaises(ValueError):verify_census(self.native,self.names,46)

    def test_missing_shared_kernel_instantiation_is_rejected(self):
        self.native['launches'].pop()
        with self.assertRaises(ValueError):verify_census(self.native,self.names,46)

    def test_foreign_runtime_process_is_rejected(self):
        self.native['launches'][0]['pid']=47
        with self.assertRaises(ValueError):verify_census(self.native,self.names,46)

    def test_late_instantiation_is_rejected(self):
        self.native['launches'][0]['call_id']=1
        with self.assertRaises(ValueError):verify_census(self.native,self.names,46)


if __name__=='__main__':unittest.main()
