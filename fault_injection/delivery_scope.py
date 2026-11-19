"""Current delivery decisions, distinct from historical proof and live admission.

Reading this inventory does not inspect a model, import Torch, load a provider,
or authorize injection. Only the existing Worker Runtime owns those decisions.
"""
from copy import deepcopy

REVISION = 'product-integration-20261119'
INTEGRATION_STATUS = 'INNER_FRAMEWORK_INTEGRATION_PENDING'

# State, declared deployment, next concrete step. Historical evidence stays in
# the existing catalog/reports; these decisions never rewrite C2/C3 results.
_SCOPES = {
    '8.restore_offset': ('ACCEPTED_SCOPE_RETAINED',
        ['actual MoE restore indices, routing weights and consumer'],
        'Wire the existing accepted serving mapping into the actual host.'),
    '15.alias_detach': ('NO_APPLICABLE_TARGET_IN_CURRENT_DEPLOYMENT',
        ['live shared storage', 'real subsequent view write and original-base consumer'],
        'Use only a discovered real write/read path; do not construct a substitute model.'),
    '17.spatial_padding': ('ACCEPTED_SCOPE_RETAINED',
        ['actual visual model/image request', 'accepted Torch Conv3d padding boundary'],
        'Wire the accepted single-projection Torch path; Native/E5 are not prerequisites.'),
    '18.api_half': ('ACCEPTED_SCOPE_RETAINED',
        ['actual source-bound RMSNorm return and consumer'],
        'Wire the existing serving return action and bounded effect summary.'),
    '20.buffer_substitute': ('ACCEPTED_SCOPE_RETAINED',
        ['matched WeightNz/QMM ABI', 'live semantic input slot and consumer'],
        'Wire the accepted Native single target without extending its deployment bounds.'),
    '22.wrong_row': ('ACCEPTED_SCOPE_RETAINED',
        ['actual fused RoPE position/index/consumer mapping'],
        'Reuse the accepted actual-call mapping with the common row/storage actions.'),
    '25.reduce_scatter': ('UNSUPPORTED', [],
        'Stopped; no communication-output corruption substitute.'),
    '26.residual_stomp': ('ACCEPTED_SCOPE_RETAINED',
        ['matched Native QMM', 'live allocator adjacency', 'exact residual consumer/recovery'],
        'Prefer the accepted Native single target; report inapplicable if adjacency is absent.'),
    '26.fia_destination': ('ACCEPTED_SCOPE_RETAINED',
        ['actual FIA writer', 'current-request live residual destination/consumer'],
        'Rebind live objects in the actual deployment; reuse accepted lifecycle and output.'),
    'quant.batch_shared_scale': ('ORDINARY_W8A8_RUNTIME_IMPLEMENTED_ACCEPTANCE_PENDING',
        ['matched Qwen W8A8 ordinary projection', 'TP1 B8 pure decode',
         'real dynamic quantization with jointly updated Q and scale'],
        'Validate the ordinary single-site six-batch runtime on the prepared small W8A8 model; main-project wiring remains pending.'),
    '28.fa.legal_tiling': ('HISTORICAL_ONLY', ['previous pinned FA scope'],
        'Retain; no expansion in this delivery.'),
    '28.fia.legal_tiling': ('HISTORICAL_ONLY', ['previous pinned FIA scope'],
        'Retain; no expansion in this delivery.'),
    '28.fia.denominator_replay': ('TND_FAULT_IMPLEMENTATION_INCOMPLETE',
        ['actual main-project TND FIA', 'real online-softmax denominator replay'],
        'Validate the source-bound denominator kernel candidate and implement actual prepared-input/launch binding before allowing F.'),
    '33.paged_decode': ('SOURCE105_SINGLE_SITE_ACCEPTED',
        ['Source105 matched paged-decode single site', 'controlled S2 strategy perturbation'],
        'Wire that single site; do not resume the 28-target group experiment.'),
    '39.matmul_range': ('ACCEPTED_SCOPE_RETAINED',
        ['matched Native MatMul arithmetic and finite input bounds'],
        'Wire the accepted single target; no blanket CANN-version claim.'),
    '39.output_proxy': ('SINGLE_REQUEST_SCOPE_ACCEPTED',
        ['independent task scope', 'at most one target per request',
         'finite-before guard, actual consumer and recovery'],
        'Source116 public four-request runtime passed on Qwen3-0.6B; wire the actual host and never reset an attempted object.'),
    '41.tail_column': ('ACCEPTED_SCOPE_RETAINED',
        ['actual live KV/cache tail, writer and consumer'],
        'Reuse the accepted scope and shared Native postwrite backend.'),
}


def describe_delivery(mechanism):
    """Machine-readable scope, not a claim about a caller's current model."""
    state, requirements, next_step = _SCOPES[mechanism]
    selected = state not in ('UNSUPPORTED', 'HISTORICAL_ONLY')
    return deepcopy(dict(
        revision=REVISION, state=state, selected_for_current_delivery=selected,
        declaration_only=True, deployment_requirements=requirements,
        live_deployment_status='NOT_EVALUATED',
        main_project_status=INTEGRATION_STATUS,
        actual_project_effect='NOT_VERIFIED', actual_project_off='NOT_VERIFIED',
        historical_acceptance_inherited=False, product_closed=False,
        next_step=next_step,
    ))
