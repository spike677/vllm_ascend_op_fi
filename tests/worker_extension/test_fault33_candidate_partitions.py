import copy,unittest
from tools.fault33_candidate_partitions import partitions


def fixture(length):
    tiles=(length+511)//512;rows=[]
    for h in range(2):
        for t in range(tiles):rows.append((h,t,t+1))
    f=dict(numHeads=4,kvHeads=2,maxKvSeqlen=length,maxQSeqlen=1,batch=1,sparseMode=3,
           embeddingSizeV=8,totalSplitNodeNum=2 if tiles>1 else 0,
           splitLseTotalSize=2*tiles*2*4 if tiles>1 else 0,
           splitOTotalSize=2*tiles*2*8*4 if tiles>1 else 0)
    for name in ('startBIdx','endBIdx','startS1Idx','endS1Idx'):f['coreInfo.'+name]=[0]*len(rows)
    for name,idx in [('startN1Idx',0),('endN1Idx',0),('startS2Idx',1),('endS2Idx',2)]:
        f['coreInfo.'+name]=[r[idx] for r in rows]
    f['coreInfo.firstSplitKVTaskLseOffset']=[i*2 if tiles>1 else 0 for i in range(len(rows))]
    f['coreInfo.firstSplitKVTaskOOffset']=[i*16 if tiles>1 else 0 for i in range(len(rows))]
    for k,v in dict(batchIdx=[0,0],headStartIdx=[0,2],headEndIdx=[2,4],qStartIdx=[0,0],qEndIdx=[1,1],
                    splitNum=[tiles,tiles],lseTaskOffset=[0,tiles*2],oTaskOffset=[0,tiles*16]).items():f['splitInfo.'+k]=v
    return dict(schema='dcat.fault33.candidate_decode.v1',shape_consistent=True,issues=[],
                key=5100000000010200203,fields=f,block_dim=len(rows),capture_sha256='a'*64)


class PartitionTests(unittest.TestCase):
    def test_three_synthetic_lengths_and_tail_do_not_grant(self):
        for n in (129,513,2305):
            r=partitions(fixture(n));self.assertEqual(r['per_kv_head_spans'][0][-1][1],n)
            self.assertTrue(r['complete_key_set']);self.assertFalse(r['qualification_granted'])
    def test_core_gap_and_split_metadata_refuse(self):
        d=fixture(513);d['fields']['coreInfo.startS2Idx'][1]=0
        with self.assertRaises(ValueError):partitions(d)
        d=fixture(513);d['fields']['splitInfo.splitNum'][0]=1
        with self.assertRaisesRegex(ValueError,'SPLIT_NODE_FIELDS'):partitions(d)
    def test_wrong_branch_and_wider_range_refuse(self):
        d=fixture(513);d['key']+=10
        with self.assertRaisesRegex(ValueError,'KEY'):partitions(d)
        d=fixture(513);d['fields']['batch']=2
        with self.assertRaisesRegex(ValueError,'B1_Q1'):partitions(d)
    def test_workspace_and_core_offsets_are_checked(self):
        for field in ('splitOTotalSize','splitLseTotalSize'):
            d=fixture(513);d['fields'][field]+=4
            with self.assertRaisesRegex(ValueError,'WORKSPACE'):partitions(d)
        d=fixture(513);d['fields']['coreInfo.firstSplitKVTaskOOffset'][1]+=1
        with self.assertRaisesRegex(ValueError,'OFFSET'):partitions(d)


if __name__=='__main__':unittest.main()
