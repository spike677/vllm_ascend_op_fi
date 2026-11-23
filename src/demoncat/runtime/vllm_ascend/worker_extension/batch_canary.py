"""Fixed serial B1 recovery observations; no changes to fault batching."""
PROTOCOL='SERIAL_B1_FIXED_PROMPT_ORDER_V1'

def stable(rows,count,reference=None):
    if len(rows)<count:return False
    last=rows[-count:]
    for row in last:
        if row.get('canary_protocol')!=PROTOCOL or row.get('all_finite') is not True:return False
        outputs=row.get('output');schedule=row.get('schedule')
        if not isinstance(outputs,list) or not 2<=len(outputs)<=8 or not isinstance(schedule,list) or len(schedule)!=len(outputs):return False
        for index,(values,steps) in enumerate(zip(outputs,schedule)):
            if len(values)!=4 or len(steps)!=4:return False
            if any(s.get('batch_size')!=1 or s.get('prompt_index')!=index for s in steps):return False
            if steps[0].get('computed')!=0 or steps[0].get('scheduled')!=steps[0].get('prompt_length'):return False
            for i,s in enumerate(steps[1:],1):
                if s.get('scheduled')!=1 or s.get('computed')!=s.get('prompt_length',0)+i-1:return False
        target=reference if reference is not None else last[0]
        if row['output']!=target['output'] or row['schedule']!=target['schedule']:return False
    return True
