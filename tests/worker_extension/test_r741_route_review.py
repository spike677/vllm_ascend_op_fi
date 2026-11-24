from copy import deepcopy
import unittest
from tools.review_r741_routes import route_checks


class ReviewTests(unittest.TestCase):
    def receipt(self,vision=False):
        mechanism='external.conv3d_output_gain' if vision else '18.variance_over4'
        c=dict(named_variant=mechanism,mean_square_multiplier=4,gain=.5,epsilon_exact=False,
            shape=[2,3],covered_elements=6,changed_elements=6,changed=1,request_ids=['r'],
            consumer_operator='aten::mm',consumed=True,restored=True,before_sha256='old',restored_sha256='old')
        r=dict(request_id='r',coverage={'applied':1},norm_variance_effect={'calls':[c]},
            external_provider=dict(stage='vision',encoder_calls=1,embedding_calls=1,readonly_inputs_weights_unchanged=True,
                native_producers=[dict(native_operator='aten::convolution',native_return_identity=True,
                    input_shape=[1,1,2,2,2],weight_shape=[2,1,1,1,1],request_ids=['r'])]),
            restore_kind='LIVE_REGION_BYTES')
        return dict(mechanism=mechanism,image_pixel_sha256=['a','b','c'],
                    scopes=[{'requests':[{'result':{'ranks':[r]}}]}])
    def test_valid_new_routes_do_not_grant(self):
        for vision in (True,False):self.assertFalse(route_checks(self.receipt(vision))['qualification_granted'])
    def test_one_element_cannot_pass_as_whole_norm(self):
        d=self.receipt();d['scopes'][0]['requests'][0]['result']['ranks'][0]['norm_variance_effect']['calls'][0]['covered_elements']=1
        with self.assertRaisesRegex(ValueError,'WHOLE_NORM'):route_checks(d)
    def test_missing_visual_readonly_or_consumption_refuses(self):
        for key in ('readonly_inputs_weights_unchanged','encoder_calls','embedding_calls'):
            d=self.receipt(True);del d['scopes'][0]['requests'][0]['result']['ranks'][0]['external_provider'][key]
            with self.assertRaisesRegex(ValueError,'IMAGE_REAL_CONSUMER'):route_checks(d)
    def test_exact_kernel_claim_refuses(self):
        d=self.receipt();d['scopes'][0]['requests'][0]['result']['ranks'][0]['norm_variance_effect']['calls'][0]['epsilon_exact']=True
        with self.assertRaisesRegex(ValueError,'PROXY_RULE'):route_checks(d)
    def test_class_name_without_actual_conv_is_insufficient(self):
        d=self.receipt(True)
        d['scopes'][0]['requests'][0]['result']['ranks'][0]['external_provider']['native_producers']=[]
        with self.assertRaisesRegex(ValueError,'ACTUAL_CONV_PRODUCER'):route_checks(d)
