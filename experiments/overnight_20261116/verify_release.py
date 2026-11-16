"""Check the reviewed publication surface; never stage private run artifacts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
BASE = 'bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf'
SELF = (E/'checksums.json').relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(raw):
    return raw.replace(b'\r\n', b'\n')


def git(*args):
    return subprocess.check_output(['git', *args],cwd=ROOT).decode('utf-8').strip()


def main():
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    assert git('branch','--show-current') == 'codex/overnight-17-fault-closure-20261116'
    # Refresh the two subtask manifests whose documents/cards received final
    # execution overlays. Original baseline source hashes remain pinned in cards.
    for name in ('audit_delivery.json','model_environment_delivery.json'):
        path=E/name; data=read(path)
        data['manifest_scope']='Final files after documented execution overlays; subtask test/NPU counts remain scoped'
        for rel, value in data['files'].items():
            raw=(ROOT/rel).read_bytes()
            if isinstance(value,dict):
                value['canonical_lf_sha256']=sha(canonical(raw))
            else:
                data['files'][rel]=sha(raw)
        path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    paths=set(git('diff','--name-only',BASE).splitlines())
    paths.update(git('ls-files','--others','--exclude-standard').splitlines())
    paths.discard(SELF)
    exact={'README.md','fault_injection/environment.py','fault_injection/model_discovery.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/runtime.py',
        'tests/worker_extension/test_extension.py','tests/worker_extension/test_probe_contract.py',
        'tests/test_overnight_model_environment.py','tests/test_overnight_evidence_contract.py'}
    records={}; issues=[]
    for rel in sorted(paths):
        path=ROOT/rel
        if not (rel in exact or rel.startswith(('docs/','profiles/','experiments/overnight_20261116/'))):
            issues.append('Outside reviewed scope: '+rel)
        if ('private' in path.parts or path.name in ('dump.json','stack.json','construct.json')
            or path.suffix.lower() in ('.so','.dll','.pt','.whl','.bin','.safetensors','.tar','.gz','.zip')):
            issues.append('Forbidden public payload: '+rel)
        raw=path.read_bytes()
        if len(raw)>1_000_000:
            issues.append('Large unaudited file: '+rel)
        if path.suffix=='.json':
            read(path)
        if path.suffix=='.md':
            text=raw.decode('utf-8-sig')
            for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'):
                    continue
                target=link.split('#',1)[0].strip('<>')
                if target and (path.parent/target).resolve() == (E/'checksums.json').resolve():
                    continue  # Generated after the link check below.
                if target and not (path.parent/target).exists():
                    issues.append('Broken local link: '+rel+' -> '+target)
        records[rel]=dict(bytes=len(raw),raw_sha256=sha(raw),canonical_lf_sha256=sha(canonical(raw)))
    matrix=read(E/'fault_matrix.json')
    assert len(matrix['mechanisms'])==17 and matrix['cpu_tests']==445
    assert matrix['new_npu_mechanism_pairs']==3 and matrix['new_common_worker_fault_closed']==0
    assert matrix['new_strict_independent_pairs_closed']==0
    assert len({r['mechanism'] for r in matrix['mechanisms']})==17
    final=read(E/'host_inventory-final.json')
    docker=next(x for x in final['inventory'] if x['argv'][:2]==['docker','ps'])
    assert docker['returncode']==0
    containers=[json.loads(s) for s in docker['stdout'].splitlines() if s.strip()]
    owned=[c['Names'] for c in containers if 'dcat.task=overnight-17-20261116' in c.get('Labels','')
           or 'demoncat.task=overnight17-cpu' in c.get('Labels','')]
    if owned:
        issues.append('Owned containers still running: '+repr(owned))
    if issues:
        raise RuntimeError('\n'.join(issues))
    out=dict(schema='demoncat.overnight.publication.v1',baseline=BASE,
        branch=git('branch','--show-current'),status='REVIEWED_SOURCE_AND_SUMMARIES_ONLY',
        file_count=len(records),files=records,self_excluded=SELF,
        git_normalization='canonical_lf_sha256 verifies Git text blobs; raw_sha256 verifies this working copy',
        new_raw_statistics_published=False,new_vendor_binaries_published=False,
        temporary_owned_containers_running=owned)
    (E/'checksums.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],files=len(records),total_bytes=sum(x['bytes'] for x in records.values()))))


if __name__=='__main__':
    main()
