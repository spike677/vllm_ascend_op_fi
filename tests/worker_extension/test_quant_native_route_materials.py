"""Parser/refusal contracts only; these fixtures are not Native execution proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review_quant_native_route_materials as audit


HEADER = '''
typedef struct aclTensorList aclTensorList;
typedef struct aclIntArray aclIntArray;
aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size);
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
aclnnStatus aclDestroyTensorList(const aclTensorList *array);
aclnnStatus aclDestroyIntArray(const aclIntArray *array);
aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size);
aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size);
'''


def origin():
    return dict(actual_npu_calls=0, framework_imported=False, image=audit.IMAGE,
                loaded_owner_proven=False, executed_api_proven=False,
                sources=[dict(path='/sdk/header.h', text=HEADER, sha256=audit.sha(HEADER.encode()))])


def elf():
    return dict(path='/sdk/libopapi.so', resolved_path='/sdk/actual/libopapi.so',
                sha256='a' * 64, bytes=123, symbol_scan_returncode=0,
                symbols=['7: 0000000000001234 128 FUNC GLOBAL DEFAULT 12 aclnnExample'],
                metadata=dict(dynamic_returncode=0, notes_returncode=0,
                    dynamic='0x1 (NEEDED) Shared library: [libnnopbase.so]',
                    notes='Build ID: abc123'))


class NativeRouteMaterialTests(unittest.TestCase):
    def test_public_counts_do_not_establish_members(self):
        result = audit.container_contract(HEADER)
        self.assertEqual(result['getter_names'], sorted(audit.SIZES))
        self.assertFalse(result['readable_member_api_in_reviewed_header'])
        self.assertFalse(result['private_layout_read_authorized'])
        self.assertFalse(result['python_argument_backfill_authorized'])
        self.assertEqual(set(result['creator_destroy_declarations']), set(audit.CREATORS + audit.DESTROYERS))

    def test_added_tensor_member_getter_requires_fresh_review(self):
        text = HEADER + '\naclnnStatus aclGetTensorFromList(const aclTensorList *list, uint64_t i, aclTensor **out);'
        with self.assertRaisesRegex(ValueError, 'GETTER_SET_CHANGED'):
            audit.container_contract(text)

    def test_added_integer_values_getter_requires_fresh_review(self):
        text = HEADER + '\naclnnStatus aclGetIntArrayData(const aclIntArray *a, int64_t **out);'
        with self.assertRaisesRegex(ValueError, 'GETTER_SET_CHANGED'):
            audit.container_contract(text)

    def test_private_struct_layout_is_not_a_public_member_getter(self):
        with self.assertRaisesRegex(ValueError, 'LAYOUT_IS_NOT'):
            audit.container_contract(HEADER + '\nstruct aclTensorList { void *members; };')

    def test_creator_const_pointer_contract_is_exact(self):
        with self.assertRaisesRegex(ValueError, 'CREATOR_SIGNATURE_CHANGED'):
            audit.container_contract(HEADER.replace('const aclTensor *const *value', 'aclTensor **value'))

    def test_size_does_not_substitute_missing_destroy(self):
        with self.assertRaisesRegex(ValueError, 'PUBLIC_CONTAINER_FUNCTION_MISSING:aclDestroyIntArray'):
            audit.container_contract(HEADER.replace('aclnnStatus aclDestroyIntArray(const aclIntArray *array);', ''))

    def test_comments_are_not_abi_declarations(self):
        text = HEADER.replace('aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);', '')
        text += '/* aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size); */'
        with self.assertRaisesRegex(ValueError, 'PUBLIC_CONTAINER_FUNCTION_MISSING:aclCreateIntArray'):
            audit.container_contract(text)

    def test_zero_must_be_actual_integer_not_false(self):
        value = origin(); value['actual_npu_calls'] = False
        with self.assertRaisesRegex(ValueError, 'DEVICE_FREE_ORIGIN_REQUIRED'):
            audit.validate_origin(value, 'native_disk')

    def test_no_execution_does_not_allow_foreign_image(self):
        value = origin(); value['image'] = 'sha256:' + 'b' * 64
        with self.assertRaisesRegex(ValueError, 'FIXED_IMAGE_REQUIRED'):
            audit.validate_origin(value, 'native_disk')

    def test_loaded_or_executed_claim_cannot_be_promoted_from_disk(self):
        for key, error in (('loaded_owner_proven', 'DISK_IS_NOT_LOADED_OWNER'),
                           ('executed_api_proven', 'DISK_IS_NOT_EXECUTED_API')):
            with self.subTest(key=key):
                value = origin(); value[key] = True
                with self.assertRaisesRegex(ValueError, error):
                    audit.validate_origin(value, 'native_disk')

    def test_embedded_text_is_independently_hashed(self):
        value = origin(); value['sources'][0]['text'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'SOURCE_BYTES_CHANGED'):
            audit.validate_origin(value, 'native_disk')

    def test_duplicate_source_path_is_not_a_second_observation(self):
        value = origin(); value['sources'].append(copy.deepcopy(value['sources'][0]))
        with self.assertRaisesRegex(ValueError, 'UNIQUE_SOURCE_PATH_REQUIRED'):
            audit.validate_origin(value, 'native_disk')

    def test_source_aliases_must_have_exact_reviewed_bytes(self):
        value = origin(); expected = value['sources'][0]['sha256']
        self.assertEqual(audit.source(value, '/header.h', expected), value['sources'][0])
        value['sources'].append(dict(path='/alternate/header.h', text='different', sha256='c' * 64))
        with self.assertRaisesRegex(ValueError, 'REVIEWED_SOURCE_REQUIRED'):
            audit.source(value, '/header.h', expected)

    def test_actual_defined_elf_symbol_is_only_disk_evidence(self):
        result = audit.elf_fact(elf(), 'a' * 64, ('aclnnExample',))
        self.assertEqual(result['needed'], ['libnnopbase.so'])
        self.assertEqual(result['build_ids'], ['abc123'])
        self.assertFalse(result['loaded_owner_proven'])
        self.assertFalse(result['executed_api_proven'])

    def test_undefined_dependency_symbol_does_not_count(self):
        row = elf(); row['symbols'][0] = row['symbols'][0].replace('DEFAULT 12', 'DEFAULT UND')
        with self.assertRaisesRegex(ValueError, 'EXACT_UNVERSIONED_EXPORT_MISSING'):
            audit.elf_fact(row, 'a' * 64, ('aclnnExample',))

    def test_symbol_version_is_not_silently_stripped(self):
        row = elf(); row['symbols'][0] += '@@OTHER_2'
        with self.assertRaisesRegex(ValueError, 'EXACT_UNVERSIONED_EXPORT_MISSING'):
            audit.elf_fact(row, 'a' * 64, ('aclnnExample',))

    def test_wrong_image_hash_and_failed_symbol_scan_refused(self):
        for key, value, error in (('sha256', 'b' * 64, 'EXACT_ELF_DISK_IDENTITY'),
                                  ('symbol_scan_returncode', 1, 'ELF_SYMBOL_SCAN_FAILED')):
            with self.subTest(key=key):
                row = elf(); row[key] = value
                with self.assertRaisesRegex(ValueError, error):
                    audit.elf_fact(row, 'a' * 64, ('aclnnExample',))

    def test_cli_loader_requires_immutable_material_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'material.json'
            path.write_text(json.dumps(origin()))
            with self.assertRaisesRegex(ValueError, 'FIXED_MATERIAL_BYTES_CHANGED:native_disk'):
                audit.load_material(path, 'native_disk')

    def test_missing_original_cannot_be_qualified(self):
        with self.assertRaisesRegex(ValueError, 'COMPLETE_FOUR_MATERIAL_SET_REQUIRED'):
            audit.review({})


if __name__ == '__main__':
    unittest.main()
