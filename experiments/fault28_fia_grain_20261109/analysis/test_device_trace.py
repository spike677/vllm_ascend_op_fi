"""Negative checks using original real SDK scalar output, not expected profiles."""
from collections import Counter
from pathlib import Path
import unittest
from device_trace import parse,expand,key


class DeviceScalarTests(unittest.TestCase):
    def setUp(self):
        raw=Path(__file__).with_name('fixtures').joinpath('device_sdk_stdout.txt').read_text()
        self.rows=[row for line in raw.splitlines() if (row:=parse(line)) is not None]
        self.unique=list({key(r):r for r in self.rows}.values())

    def test_actual_sdk_records_and_complete_registered_query_tiles(self):
        self.assertEqual(len(self.rows),42)
        self.assertEqual(set(Counter(map(key,self.rows)).values()),{3})
        points=expand(self.unique,128,1)
        self.assertEqual(len(points),14)
        self.assertEqual(next(r['measured_columns'] for r in points if r['query']==63),128)

    def test_missing_actual_query_is_rejected(self):
        rows=[r for r in self.unique if not r['row']<=63<r['row']+r['rows']]
        with self.assertRaisesRegex(ValueError,'Missing'):expand(rows,128,1)

    def test_duplicate_measurement_is_not_silently_deduplicated(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):expand(self.unique+[self.unique[0]],128,1)

    def test_legal_but_forged_column_count_is_rejected(self):
        rows=[dict(r) for r in self.unique]
        index=next(i for i,r in enumerate(rows) if r['row']<=63<r['row']+r['rows'])
        rows[index].update(columns=256,actual=256)
        with self.assertRaisesRegex(ValueError,'Measured'):expand(rows,128,1)

    def test_different_actual_raw_m_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'actual key/M'):expand(self.unique,512,1)

    def test_literal_compiler_format_string_is_not_device_data(self):
        self.assertIsNone(parse('F28_GRAIN_V1 block=%u m=%u tile=%u row=%u rows=%u'))

    def test_foreign_block_and_invalid_row_interval_rejected(self):
        text='F28_GRAIN_V1 block=32 m=128 tile=0 row=0 rows=64 s2=0 base=512 columns=128 actual=128 prune=1 brc=1'
        with self.assertRaisesRegex(ValueError,'geometry'):parse(text)
        with self.assertRaisesRegex(ValueError,'row interval'):parse(text.replace('block=32','block=0').replace('rows=64','rows=129'))


if __name__=='__main__':unittest.main()
