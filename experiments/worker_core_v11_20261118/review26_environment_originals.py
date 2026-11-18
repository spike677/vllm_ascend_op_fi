"""Reconcile the saved held-function environment; never resolve a new symbol."""
GROUP_SYMBOLS=('aclrtGetDevice','aclGetDataType','aclGetFormat','aclGetViewOffset','aclGetViewShape',
    'aclGetStorageShape','aclGetViewStrides','aclGetRawTensorAddr','aclCreateTensor','aclDestroyTensor',
    'aclrtMalloc','aclrtFree','aclrtMemcpy','aclrtSynchronizeStream')
NAMES=('libdcat26_group_v2.so','libfault26_readonly_trace_abi0.so','libnnopbase.so','libascendcl.so')


def review(C,a,env,expected_sha,lifetime,*,native_images=None):
    rows=env['loaded_libraries'];bound=env.get('bound_symbol_evidence')
    a.check(env['maps_status']=='OBSERVED' and env['pid']==lifetime['pid'] and
        set(NAMES).issubset(x['name'] for x in rows),'SAVED_ACTUAL_MAPPED_ENVIRONMENT')
    common=dict(machine=env['machine'],python=env['python'],packages=env['packages'],torch_cxx11_abi=env['torch_cxx11_abi'],
        driver=dict(running_module_versions=env['driver']['running_module_versions'],
                    installation_sha256=(env['driver']['installation'] or {}).get('sha256')))
    if bound is None:
        a.check(all(x['status']=='MAPPED_FILE_VERIFIED' for x in rows),'EXACT_VERIFIED_MAPPED_ENVIRONMENT')
        for name in NAMES:
            values=[(x['sha256'],x['elf_build_id']) for x in rows if x['name']==name]
            a.check(bool(values) and len(set(values))==1,'NO_AMBIGUOUS_REQUIRED_MAPPED_LIBRARY',name)
        mapped={x['name']:[x['sha256'],x['elf_build_id']] for x in rows}
        material=dict(common,libraries={name:mapped[name] for name in sorted(NAMES)})
    else:
        a.check(bound['schema']=='demoncat.bound_environment.v1' and bound['status']=='EXACT_MAPPED_FUNCTION_BINDINGS_OBSERVED' and
            bound['qualification_granted'] is bound['serializable_authority'] is False and not bound.get('refusal'),
            'SAVED_BOUND_PROFILE_NOT_LIVE_AUTHORITY')
        paths={x['path']:x for x in rows}
        a.check(len(paths)==len(rows) and all(x.get('file_identity_status',x['status'])=='MAPPED_FILE_VERIFIED' and
            C.sha(x['sha256']) and x['mapping']['deleted'] is False for x in rows),
            'ALL_MAPPED_IMAGES_RETAINED_AND_INDIVIDUALLY_VERIFIED')
        expected={('dcat26_group_cached_address_v1',i):name for i,name in enumerate(GROUP_SYMBOLS)}
        expected.update({('dcat26_custom_cached_address_v1',0):'aclnnAddRmsNormBias',
                         ('dcat26_custom_cached_address_v1',1):'aclnnAddRmsNormBiasGetWorkspaceSize'})
        seen={};edges=[];bound_paths=set()
        for value in bound['bindings']:
            getter,target=value['getter'],value['target'];key=(getter['symbol'],value['slot'])
            a.check(key not in seen and type(value['slot']) is int and expected.get(key)==value['symbol'] and
                value['kind']=='CACHED_FUNCTION_POINTER','COMPLETE_SOURCE_DEFINED_HELD_BINDING',str(key));seen[key]=value['symbol']
            for which,origin,symbol in (('getter',getter,getter['symbol']),('target',target,value['symbol'])):
                a.check(C.positive(origin['address']) and C.positive(origin['base']) and origin['base']<=origin['address'] and
                    origin['address']==origin['symbol_address'] and origin['symbol']==symbol,'EXACT_HELD_SYMBOL_ADDRESS',str(key)+':'+which)
                matches=[]
                for row in rows:
                    if any(int(segment['address'].split('-')[0],16)<=origin['address']<int(segment['address'].split('-')[1],16)
                           for segment in row['mapping']['executable_segments']):matches.append(row)
                a.check(len(matches)==1 and matches[0]['path']==origin['mapped_path'] and
                    matches[0]['sha256']==value[which+'_sha256'],'UNIQUE_SAVED_BOUND_EXECUTABLE_IMAGE',str(key)+':'+which)
                if which=='target':a.check(matches[0]['elf_build_id']==value['target_elf_build_id'],'TARGET_ELF_BUILD_ID',str(key))
                bound_paths.add(origin['mapped_path'])
            expected_name='libdcat26_group_v2.so' if key[0]=='dcat26_group_cached_address_v1' else 'libcust_opapi.so'
            a.check(paths[getter['mapped_path']]['name']==expected_name,'ACTUAL_GETTER_IMAGE_NOT_SAME_NAME_SUBSTITUTE',str(key))
            a.check(native_images is not None and native_images.get(expected_name)==value['getter_sha256'],
                    'HELD_GETTER_MATCHES_SOURCE_BOUND_METADATA_IMAGE',str(key))
            if expected_name=='libdcat26_group_v2.so':
                a.check(value['getter_sha256']==lifetime['native_writer_sha256'],'HELD_WRITER_MATCHES_ORIGINAL_BUILD',str(key))
            edges.append(dict(kind=value['kind'],getter_path=getter['mapped_path'],getter_symbol=getter['symbol'],
                getter_sha256=value['getter_sha256'],slot=value['slot'],symbol=value['symbol'],
                target_path=target['mapped_path'],target_sha256=value['target_sha256']))
        a.check(seen==expected and len(bound['bindings'])==16,'ALL_SIXTEEN_ACTUAL_HELD_FUNCTIONS')
        collisions=[]
        for name in sorted({x['name'] for x in rows}):
            group=[x['path'] for x in rows if x['name']==name]
            if len(group)>1:
                selected=sorted(bound_paths.intersection(group))
                a.check(bool(selected),'COLLISION_REQUIRES_ACTUAL_HELD_FUNCTION',name)
                collisions.append(dict(name=name,complete_paths=group,actual_bound_paths=selected,all_images_retained=True))
        a.check(bound['collision_groups']==collisions,'COMPLETE_COLLISION_SET_NOT_HIDDEN')
        material=dict(common,profile=bound['schema'],images=[{k:x[k] for k in ('path','name','sha256','elf_build_id')}
            for x in sorted(rows,key=lambda row:row['path'])],bindings=edges)
        a.check(bound['scope_material']==material and bound['fingerprint']==C.digest(material),'REBUILT_HELD_SCOPE_MATERIAL')
    a.check(expected_sha==C.digest(material),'EXACT_ENVIRONMENT_SCOPE_SHA')
    return dict(mapped_images=len(rows),held_bindings=len(bound['bindings']) if bound else 0,
                saved_receipts_only=True,live_symbols_resolved=False)
