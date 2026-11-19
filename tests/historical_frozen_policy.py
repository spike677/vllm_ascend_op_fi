"""Opt-in historical SHA audits; excluded from current-release CPU acceptance.

Run on their frozen source/evidence checkout, not the changing development tree.
No frozen pin or original assertion is rewritten to match current code.
"""
import os
import unittest
frozen_source_check=unittest.skipUnless(
    os.environ.get('DEMONCAT_RUN_HISTORICAL_SHA_CHECKS')=='1',
    'Historical frozen-source SHA audit; not an RC2 current-source acceptance test. '
    'Use its frozen checkout and explicit DEMONCAT_RUN_HISTORICAL_SHA_CHECKS=1.')
