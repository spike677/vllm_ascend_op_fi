"""Read a completed graph run and bounded native log excerpts, with no NPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT=Path('/home/fault28-fia-grain-20261109')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    args=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]{1,40}',args.attempt)
    root=ROOT/'outputs'/args.attempt
    completion=json.loads((root/'completion.json').read_bytes())
    files=['completion.json','source_manifest.json','selected_cases.json','container_identity.json']
    excerpts=[]
    for c in completion['cases']:
        name=c['case_id']
        folder=root/name
        result=json.loads((folder/'result.json').read_bytes())
        files += [name+'/'+x for x in ['result.json','case.json','inputs.json','graph_config.json','fx_graph_0.json'] if (folder/x).is_file()]
        files += [name+'/'+x for x in result.get('ge_files',[]) if Path(x).suffix in ('.txt','.csv')]
        digest=hashlib.sha256(); hits=[];total=0;counts={}
        with (root/(name+'.log')).open('rb') as stream:
            for number,line in enumerate(stream,1):
                digest.update(line); total+=len(line)
                text=line.decode('utf-8','replace').rstrip()
                terms=['FIA sInnerSize_', 'mBaseSize_', 's1BaseSize', 'sOuterFactor', 'tilingKey:',
                       'tilingKey is', 'tiling key:', 'blockDim:', 'splitKVFlag_', 'deterministic',
                       'tiling_key =', 'tilingkey:', 'tilingKey =', 'FIA tiling', 'kernelName =']
                matched=[x for x in terms if x in text]
                for term in matched: counts[term]=counts.get(term,0)+1
                if matched and len(hits)<600:
                    hits.append({'line':number,'text':text[:4000],'terms':matched})
        excerpts.append({'case_id':name,'source_log_sha256':digest.hexdigest(),'bytes':total,
                         'counts':counts,'unclassified_log_matches':hits,
                         'note':'Discovery excerpts; static registration and actual launch not yet separated. No acceptance decision.'})
    target=ROOT/'outputs'/(args.attempt+'_inspection.json')
    with target.open('x') as stream: json.dump(excerpts,stream,indent=2);stream.write('\n')
    archive=ROOT/'outputs'/(args.attempt+'_metadata.tar.gz')
    with tarfile.open(archive,'x:gz') as stream:
        for name in files:
            path=root/name
            assert path.is_file() and path.stat().st_size<8*1024**2
            stream.add(path,arcname=name,recursive=False)
        stream.add(target,arcname='inspection.json',recursive=False)
    print(json.dumps({'archive':str(archive),'files':len(files)+1,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'native_match_counts':[x['counts'] for x in excerpts]}))


if __name__=='__main__': main()
