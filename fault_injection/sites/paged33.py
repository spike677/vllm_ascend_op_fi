"""Paged decode legal partition proxy, with a native-equal control per cell.

Only cells whose full-key control rounds exactly to the native result may be
replaced. This separates partition rounding from native/reference algorithm
differences. Prefill is observed but never injected. No denominator replay.
"""
import torch
from .gqa28 import GQA28Site
from .mla28 import require,fingerprint
from fault_injection.engines.attention import recompute_attention_row


def partition_pair(q,k,v,*,scale,native,control_block=128,fault_block=32):
    control=recompute_attention_row(q,k,v,scale=scale,block_size=control_block).reshape_as(native)
    alternate=recompute_attention_row(q,k,v,scale=scale,block_size=fault_block).reshape_as(native)
    matched=control==native
    result=torch.where(matched,alternate,native)
    return result,dict(control_block=control_block,fault_block=fault_block,
        same_legal_key_set=True,key_visits=len(k),denominator_replay=False,
        native_equal_control_elements=int(matched.sum().item()),
        partition_changed_elements=int((control!=alternate).sum().item()),
        selected_changed_elements=int(((result!=native)&matched).sum().item()),
        nonselected_exact=bool(torch.equal(result[~matched],native[~matched])),
        control_sha256=fingerprint(control),alternate_sha256=fingerprint(alternate))


class Paged33Site(GQA28Site):
    control_block=128
    fault_block=32
    def eligible_call(self,count,length):
        if count!=1 or length<=1:return False,'PREFILL_NOT_PAGED_DECODE'
        if length<=128:return False,'MULTIPLE_REDUCTION_BLOCKS_REQUIRED'
        return True,None

    def calculate(self,q,k,v,impl,backup,event):
        require(event.get('page_mapping',{}).get('logical_keys')==len(k),'PAGED_ACTUAL_CACHE_MAPPING_MISSING')
        value,report=partition_pair(q,k,v,scale=impl.scale,native=backup,control_block=self.control_block,fault_block=self.fault_block)
        require(report['nonselected_exact'],'PAGED_NONSELECTED_CHANGED')
        event['partition_reference']=report
        return value
