"""Raw-only B/repeat/F readiness review, without execution or expected locations.

This adds controls and observability distinctions to the frozen comparator.
It never filters records, rewrites raw statistics or grants mechanism E5.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from fault_injection import statistics_audit as comparator
from experiments.c2c3_closure_20261117.audit_statistics_series import captures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stack_context(directory, first):
    """Report possible diagnostic provenance after first difference is fixed."""
    stack,_=comparator.read_json(Path(directory)/'stack.json')
    construct,_=comparator.read_json(Path(directory)/'construct.json')
    frames=Counter();suspect=[];first_stacks=[]
    op=first['identity'].split('/',1)[0] if first else None
    for key,value in stack.items():
        if not isinstance(value,list) or len(value)!=2 or not all(isinstance(v,list) for v in value):
            return dict(schema_supported=False,malformed_stack_group=key,source_membership_proves_no_contamination=False)
        names,items=value
        if op in names:first_stacks.append(key)
        for item in items:
            if not isinstance(item,str):raise ValueError('STACK_FRAME_NOT_TEXT')
            match=re.match(r'File (.+), line ([0-9]+), in ([^,]+),',item)
            if match is None:raise ValueError('STACK_FRAME_UNPARSED')
            file,_,function=match.groups()
            if '/fault_injection/' in file or '/demoncat/' in file:
                identity=file.split('/source/',1)[-1]+':'+function
                frames[identity]+=1
                # These are owned diagnostic bodies, not every dispatch wrapper.
                # A finding flags the complete capture; no record is removed.
                if function in ('cpu','_capture','_proxy_return','metadata','tensor_geometry'):
                    suspect.append(dict(stack_group=key,source_function=identity,operators=names))
    return dict(schema_supported=True,first_operator=op,first_operator_stack_groups=first_stacks,
        first_operator_construct_parent=construct.get(op) if op else None,
        owned_source_frame_occurrences=dict(sorted(frames.items())),suspected_diagnostic_records=suspect,
        any_record_excluded=False,source_membership_proves_no_contamination=False)


def observability(directory):
    document,_=comparator.read_json(Path(directory)/'dump.json');counts=Counter()
    def walk(value):
        if isinstance(value,dict):
            if value.get('type')=='torch.Tensor':
                for field in comparator.METRICS:
                    if value[field] is None:counts[(value.get('dtype','UNDECLARED'),field)]+=1
            else:
                for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(document['data'])
    return [dict(dtype=dtype,field=field,null_fields=count) for (dtype,field),count in sorted(counts.items())]


def unfiltered(root):
    config,_=comparator.read_json(Path(root)/'config.json');stats=config.get('statistics',{})
    return (all(stats.get(key)==[] for key in ('scope','list','tensor_list')) and stats.get('data_mode')==['all'])


def review(baseline,baseline_repeat,fault):
    roots=dict(baseline=Path(baseline),baseline_repeat=Path(baseline_repeat),fault=Path(fault))
    steps={name:captures(root) for name,root in roots.items()}
    same=all(set(v)==set(steps['baseline']) for v in steps.values());rows=[];full={}
    # These three are numerical originals only. No capture/result/hit receipt is
    # opened by this reader, and no operator/expected-first argument exists.
    for step,rank in sorted(set.intersection(*(set(v) for v in steps.values()))):
        pairs={}
        for label,left,right in (('self','baseline','baseline'),('repeat','baseline','baseline_repeat'),
                ('bf','baseline','fault'),('repeat_f','baseline_repeat','fault')):
            b,f=roots[left],roots[right]
            value=comparator.compare_raw_statistics(baseline=steps[left][step,rank],fault=steps[right][step,rank],
                baseline_config=b/'config.json',fault_config=f/'config.json',
                baseline_collector=b/'collector_identity.json',fault_collector=f/'collector_identity.json')
            full[f'step{step}-rank{rank}-{label}']=value;pairs[label]=comparator.compact_summary(value)
        bf=pairs['bf'];first=bf['first_exceedance']
        control_ok=all(pairs[key]['e5_statistics_comparable'] and pairs[key].get('exceedance_count')==0 for key in ('self','repeat'))
        observed=control_ok and bf['e5_statistics_comparable'] and first is not None
        repeat_first=pairs['repeat_f']['first_exceedance']
        same_first=first is not None and repeat_first is not None and all(first[k]==repeat_first[k] for k in ('identity','field'))
        context=stack_context(steps['baseline'][step,rank],first)
        no_known_diagnostics=context['schema_supported'] and not context.get('suspected_diagnostic_records')
        rows.append(dict(step=step,rank=rank,pairs=pairs,baseline_controls_comparable_at_fixed_tolerance=control_ok,
            first_observable_difference_supported=bool(observed and same_first),
            no_known_diagnostic_body_in_stack=no_known_diagnostics,
            first_location_stable_against_independent_baseline=same_first,
            all_preceding_statistic_fields_observed=bool(first is not None and first['preceding_unobservable_fields']==0 and
                first['preceding_incomparable_fields']==0),
            null_field_inventory={name:observability(paths[step,rank]) for name,paths in steps.items()},
            stack_context=context))
    config_ok={name:unfiltered(root) for name,root in roots.items()}
    return dict(schema='demoncat.raw_statistics_readiness.v1',read_only=True,runtime_executed=False,
        comparator_sha256=sha(comparator.__file__),reviewer_sha256=sha(__file__),rtol='1e-5',atol='1e-8',
        comparison_rule='abs(F-B) > 1e-8 + 1e-5 * abs(B)',native_logs_read=False,private_tensors_read=False,
        expected_location_read=False,raw_statistics_rewritten=False,
        step_rank_sets={name:[list(v) for v in sorted(values)] for name,values in steps.items()},
        actual_step_rank_sets_equal=same,unfiltered_configuration=config_ok,pairs=rows,
        all_supplied_steps_have_controlled_observable_difference=bool(same and rows and all(config_ok.values()) and
            all(row['first_observable_difference_supported'] and row['no_known_diagnostic_body_in_stack'] for row in rows)),
        full_numeric_prefix_observed=bool(same and rows and all(row['all_preceding_statistic_fields_observed'] for row in rows)),
        fault_repeat_stability_measured=False,source_or_scope_identity_verified=False,
        root_cause_identified=False,current_public_scope_qualified=False,public_qualification_granted=False,e5=False,
        limits=['A preceding null is unobservable, not equal. Raw first observable change is not a unique cause.',
            'Only B, independent B repeat and one F capture are compared; no repeated-F stability is claimed.',
            'Stack inspection occurs after numerical selection and does not delete or reorder any record.',
            'Absence of known diagnostic frames is not an exhaustive source/collector purity proof.',
            'Source, model, capture-window and public lifecycle identity require a separate review.']),full


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','baseline-repeat','fault','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--comparator-sha256',required=True)
    args=parser.parse_args(argv)
    if sha(comparator.__file__)!=args.comparator_sha256:raise ValueError('FROZEN_COMPARATOR_BYTES_CHANGED')
    if str(comparator.RTOL)!='0.00001' or str(comparator.ATOL)!='1E-8':raise ValueError('FIXED_TOLERANCE_CHANGED')
    report,full=review(args.baseline,args.baseline_repeat,args.fault)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,value in full.items():
        path=args.out/(name+'-full.json');path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report['full_report_sha256']={name:sha(args.out/(name+'-full.json')) for name in full}
    path=args.out/'readiness.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary_sha256=sha(path),controlled_observable_difference=report['all_supplied_steps_have_controlled_observable_difference'],
        full_numeric_prefix_observed=report['full_numeric_prefix_observed'],e5=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
