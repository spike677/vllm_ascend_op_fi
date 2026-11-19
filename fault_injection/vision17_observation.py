"""Pure identities shared by ordinary tasks and historical validation."""
def call_signature(row):
    return dict(path=row['path'],sequence=row['sequence'],operator=row['operator'],schema=row['schema'],
        geometry=row['actual_geometry'],inputs=row['inputs_before'],
        input_metadata=[{k:v for k,v in t.items() if k in ('shape','stride','offset','dtype','device','actual_format')}
                        for t in row['actual_input_metadata']])


def output_identity(row):
    return {key:row[key] for key in ('encoder_output','hidden','logits')}


