"""Config, attested adapter resolution and recorded-discovery CLI; no driver imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .adapters.flash_attention import FlashAttentionAdapter, selector_projection
from .certificates import load_catalog, resolve_catalog, strict_json
from .config import load_config
from .discovery import DiscoveryCase, discover
from .models import ResourceEstimate, require
from .policies import Policy
from .registry import AdapterRegistration, AdapterRegistry

PACKAGE_ROOT = Path(__file__).resolve().parent


def registry_adapter(config, catalog: Path, evidence_root: Path, allow_experimental: bool):
    loaded = resolve_catalog(catalog, evidence_root=evidence_root, version=config.adapter.version,
                             operator=config.adapter.name, route=config.adapter.route or 'S1S2',
                             allow_experimental=allow_experimental)
    adapter = FlashAttentionAdapter(loaded.path, evidence_root=evidence_root,
        selection_target=config.selection_target, allow_experimental=allow_experimental,
        on_unsupported=config.adapter.on_unsupported)
    registry = AdapterRegistry()
    registry.register(AdapterRegistration(adapter.name, adapter.version, adapter.fingerprints,
        loaded.data['validation_state'], loaded.evidence, route=adapter.route), adapter)
    return registry.resolve(config.adapter)


def emit(value: dict, output: Path | None = None):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if output is None:
        print(text, end='')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)


def bound(base: Path, row: dict) -> Path:
    path = (base / row['path']).resolve(strict=True)
    require(path.is_relative_to(base.resolve()) and path.is_file(), 'Escaping recorded-observation path')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'Recorded observation SHA changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate-config')
    validate.add_argument('config', type=Path)
    for name in ('adapters', 'plan', 'discover'):
        p = sub.add_parser(name)
        p.add_argument('--catalog', type=Path, default=PACKAGE_ROOT / 'data/catalog.json')
        p.add_argument('--evidence-root', type=Path, default=PACKAGE_ROOT)
        p.add_argument('--output', type=Path)
        if name != 'adapters':
            p.add_argument('--config', type=Path, required=True)
            p.add_argument('--allow-experimental-validation', action='store_true')
        if name == 'plan':
            p.add_argument('--call', type=Path, required=True)
            p.add_argument('--native-record', type=Path)
            p.add_argument('--observation-root', type=Path)
        if name == 'discover':
            p.add_argument('--records', type=Path, required=True,
                           help='SHA-bound actual observation manifest; does not execute new native calls')
    args = parser.parse_args()
    if args.command == 'validate-config':
        emit({'status': 'CONFIG_VALID', 'normalized': asdict(load_config(args.config)), 'native_calls': 0})
        return
    if args.command == 'adapters':
        values = load_catalog(args.catalog, evidence_root=args.evidence_root, allow_experimental=True)
        emit({'adapters': [{'adapter': value.data['adapter'], 'validation_state': value.data['validation_state'],
                           'fingerprints': value.fingerprints.as_dict()} for value in values],
              'native_calls': 0}, args.output)
        return
    config = load_config(args.config)
    resolved = registry_adapter(config, args.catalog, args.evidence_root, args.allow_experimental_validation)
    adapter = resolved.adapter
    if args.command == 'plan':
        context = adapter.inspect(strict_json(args.call.read_bytes()))
        reason = adapter.unsupported_reason(context, controlled=True)
        if reason and config.adapter.on_unsupported == 'passthrough':
            emit({'status': 'UNSUPPORTED_CONTEXT_PASSTHROUGH', 'reason': reason, 'resolved': resolved.as_dict(),
                  'assessment': {'strategy_switched': None, 'numerics_observed': None, 'model_impact': None}, 'native_calls': 0}, args.output)
            return
        require(args.native_record is not None and args.observation_root is not None,
                'A real native observation and its explicit evidence directory are required')
        adapter.bind_evidence_directory(args.observation_root)
        observation = adapter.parse_native_evidence(context, strict_json(args.native_record.read_bytes()))
        decision = Policy(config.policy).evaluate(context, observation.strategy,
            adapter.legal_alternatives(context, observation.strategy), adapter.validate_candidate)
        emit({'status': 'INTENT_ONLY_NO_EXECUTION', 'resolved': resolved.as_dict(), 'context': context.as_dict(),
              'decision': decision.as_dict(), 'native_calls': 0}, args.output)
        return
    manifest = strict_json(args.records.read_bytes())
    require(manifest['schema'] == 'nes.recorded_discovery.v1', 'Wrong recorded discovery schema')
    base = args.records.resolve().parent
    cases, observations, estimates, rows = [], {}, {}, {}
    for row in manifest['observations']:
        context = adapter.inspect(row['actual_call'])
        case = DiscoveryCase(row['case_id'], context)
        require(case.case_id not in rows, 'Duplicate discovery case')
        rows[case.case_id] = row
        estimates[case.case_id] = ResourceEstimate(**row['resource_estimate'])
        cases.append(case)
    def observe(case):
        row = rows[case.case_id]
        path = bound(base, row['record'])
        directory = (base / row['observation_root']).resolve(strict=True)
        require(directory.is_relative_to(base), 'Escaping observation root')
        adapter.bind_evidence_directory(directory)
        value = adapter.parse_native_evidence(case.context, strict_json(path.read_bytes()))
        observations[case.case_id] = value
        return value
    report = discover(cases, observe, config.discovery,
                      lambda case: estimates[case.case_id]).as_dict()
    selectors = {json.dumps(selector_projection(value.strategy), sort_keys=True) for value in observations.values()}
    report.update(resolved_adapter=resolved.as_dict(), selector_changes_observed=len(selectors) > 1 if selectors else None,
                  selector_scope='M/R/keytile; complete execution plans remain in strategy records',
                  analysis_scope='Read-only source-bound recorded observations; no new device execution',
                  elapsed_scope='Analysis admission/processing time; original-call limits are enforced by its capture harness',
                  native_calls=0)
    emit(report, args.output)


if __name__ == '__main__':
    main()
