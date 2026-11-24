import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

from demoncat.runtime.vllm_ascend.worker_extension.external_provider import (
    ExternalSite, ExternalFaultConfig, external_site, external_capabilities,
    register_external_site, factory_sha256, create_external_provider)
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


def example_factory(*args, **kwargs):
    raise AssertionError('Registration must not instantiate a site')


class ExternalProviderTests(unittest.TestCase):
    def setUp(self):
        self.worker=NS(model_runner=NS(model=object()))
        self.authority=NS(authorize_operator=lambda principal,action: principal=='trusted')
        self.actual=dict(model_config_sha256='a'*64,tp=2)
        self.spec=ExternalSite('external.example',example_factory,factory_sha256(example_factory),self.actual.copy())
        self.patches=[patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=self.authority),
                      patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.observe_deployment',return_value=self.actual)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def install(self,spec=None,operator='trusted'):
        return register_external_site(self.worker,spec or self.spec,authority=self.authority,operator=operator)

    def test_inert_registration_does_not_add_public_builtin_or_grant(self):
        report=self.install()
        self.assertFalse(report['qualification_granted'])
        self.assertEqual(len(external_capabilities(self.worker)),1)
        self.assertIs(external_site(self.worker,'external.example',validate=True).factory,example_factory)
        with self.assertRaises(ValueError):FaultConfig.from_dict(dict(fault_id='external',variant='example'))
        self.assertEqual(ExternalFaultConfig('external','example').mechanism,'external.example')

    def test_cannot_shadow_or_load_arbitrary_module(self):
        for name,factory in [('18.api_half',example_factory),('external.bad','module:callable')]:
            with self.assertRaises(RuntimeError):self.install(ExternalSite(name,factory,self.spec.source_sha256,self.actual))
        with self.assertRaises(RuntimeError):self.install(operator='user')

    def test_source_identity_and_full_deployment_identity_are_required(self):
        with self.assertRaisesRegex(RuntimeError,'SOURCE_SHA'):self.install(ExternalSite(self.spec.mechanism,example_factory,'0'*64,self.actual))
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):self.install(ExternalSite(self.spec.mechanism,example_factory,self.spec.source_sha256,{'tp':2}))
        self.install();self.actual['tp']=1
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):external_site(self.worker,self.spec.mechanism,validate=True)

    def test_model_replacement_cannot_reuse_registration(self):
        self.install();self.worker.model_runner.model=object()
        with self.assertRaisesRegex(RuntimeError,'MODEL_IDENTITY_CHANGED'):external_site(self.worker,self.spec.mechanism,validate=True)

    def test_missing_permit_cannot_resolve_even_registered_site(self):
        self.install()
        with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.deployment_acceptance_report',return_value=dict(can_prepare=False,acceptance_permit=None)):
            with self.assertRaisesRegex(RuntimeError,'PERMIT_REQUIRED'):
                create_external_provider(self.worker,self.worker.model_runner.model,{},self.spec.mechanism,'1')

    def test_existing_binding_exclusion_is_preserved(self):
        self.worker._demoncat_continuous_bindings={'18.api_half':object()}
        with self.assertRaisesRegex(RuntimeError,'IDLE_WORKER'):self.install()



if __name__=='__main__':unittest.main()
