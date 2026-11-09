


def forward(self, input_ids, position_ids, causal_mask):
    arg0, arg1, arg2, = fx_pytree.tree_flatten_spec(([input_ids, position_ids, causal_mask], {}), self._in_spec)
    l_input_ids_ = arg0
    l_position_ids_ = arg1
    l_causal_mask_ = arg2
    l__self___model_model_embed_tokens_weight = self.L__self___model_model_embed_tokens_weight
    inputs_embeds = torch.nn.functional.embedding(l_input_ids_, l__self___model_model_embed_tokens_weight, None, None, 2.0, False, False);  l_input_ids_ = None
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    l__self___model_model_rotary_emb_inv_freq = self.L__self___model_model_rotary_emb_inv_freq
    getitem = l__self___model_model_rotary_emb_inv_freq[(None, slice(None, None, None), None)];  l__self___model_model_rotary_emb_inv_freq = None
    float_1 = getitem.float();  getitem = None
    expand = float_1.expand(1, -1, 1);  float_1 = None
    inv_freq_expanded = expand.to(device(type='cpu'));  expand = None
    getitem_3 = l_position_ids_[(slice(None, None, None), None, slice(None, None, None))];  l_position_ids_ = None
    position_ids_expanded = getitem_3.float();  getitem_3 = None
    float_3 = inv_freq_expanded.float();  inv_freq_expanded = None
    float_4 = position_ids_expanded.float();  position_ids_expanded = None
    matmul = float_3 @ float_4;  float_3 = float_4 = None
    freqs = matmul.transpose(1, 2);  matmul = None
    emb = torch.cat((freqs, freqs), dim = -1);  freqs = None
    cos = emb.cos()
    cos_1 = cos * 1.0;  cos = None
    sin = emb.sin();  emb = None
    sin_1 = sin * 1.0;  sin = None
    cos_2 = cos_1.to(dtype = torch.float16);  cos_1 = None
    sin_2 = sin_1.to(dtype = torch.float16);  sin_1 = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
    _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module');  _log_api_usage_once = None
    hidden_states = inputs_embeds.to(torch.float32)
    pow_1 = hidden_states.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_1 = hidden_states * rsqrt;  hidden_states = rsqrt = None
    l__self___model_model_layers_0_input_layernorm_weight = self.L__self___model_model_layers_0_input_layernorm_weight
    to_4 = hidden_states_1.to(torch.float16);  hidden_states_1 = None
    hidden_states_2 = l__self___model_model_layers_0_input_layernorm_weight * to_4;  l__self___model_model_layers_0_input_layernorm_weight = to_4 = None
    size_1 = hidden_states_2.size()
    getitem_5 = size_1[1];  size_1 = None
    l__self___model_model_layers_0_self_attn_q_proj_weight = self.L__self___model_model_layers_0_self_attn_q_proj_weight
    linear = torch._C._nn.linear(hidden_states_2, l__self___model_model_layers_0_self_attn_q_proj_weight, None);  l__self___model_model_layers_0_self_attn_q_proj_weight = None
    view = linear.view((1, getitem_5, -1, 128));  linear = None
    hidden_states_3 = view.to(torch.float32);  view = None
    pow_2 = hidden_states_3.pow(2)
    variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    add_1 = variance_1 + 1e-06;  variance_1 = None
    rsqrt_1 = torch.rsqrt(add_1);  add_1 = None
    hidden_states_4 = hidden_states_3 * rsqrt_1;  hidden_states_3 = rsqrt_1 = None
    l__self___model_model_layers_0_self_attn_q_norm_weight = self.L__self___model_model_layers_0_self_attn_q_norm_weight
    to_6 = hidden_states_4.to(torch.float16);  hidden_states_4 = None
    mul_5 = l__self___model_model_layers_0_self_attn_q_norm_weight * to_6;  l__self___model_model_layers_0_self_attn_q_norm_weight = to_6 = None
    query_states = mul_5.transpose(1, 2);  mul_5 = None
    l__self___model_model_layers_0_self_attn_k_proj_weight = self.L__self___model_model_layers_0_self_attn_k_proj_weight
    linear_1 = torch._C._nn.linear(hidden_states_2, l__self___model_model_layers_0_self_attn_k_proj_weight, None);  l__self___model_model_layers_0_self_attn_k_proj_weight = None
    view_1 = linear_1.view((1, getitem_5, -1, 128));  linear_1 = None
    hidden_states_5 = view_1.to(torch.float32);  view_1 = None
    pow_3 = hidden_states_5.pow(2)
    variance_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
    add_2 = variance_2 + 1e-06;  variance_2 = None
    rsqrt_2 = torch.rsqrt(add_2);  add_2 = None
    hidden_states_6 = hidden_states_5 * rsqrt_2;  hidden_states_5 = rsqrt_2 = None
    l__self___model_model_layers_0_self_attn_k_norm_weight = self.L__self___model_model_layers_0_self_attn_k_norm_weight
    to_8 = hidden_states_6.to(torch.float16);  hidden_states_6 = None
    mul_7 = l__self___model_model_layers_0_self_attn_k_norm_weight * to_8;  l__self___model_model_layers_0_self_attn_k_norm_weight = to_8 = None
    key_states = mul_7.transpose(1, 2);  mul_7 = None
    l__self___model_model_layers_0_self_attn_v_proj_weight = self.L__self___model_model_layers_0_self_attn_v_proj_weight
    linear_2 = torch._C._nn.linear(hidden_states_2, l__self___model_model_layers_0_self_attn_v_proj_weight, None);  hidden_states_2 = l__self___model_model_layers_0_self_attn_v_proj_weight = None
    view_2 = linear_2.view((1, getitem_5, -1, 128));  linear_2 = None
    value_states = view_2.transpose(1, 2);  view_2 = None
    cos_3 = cos_2.unsqueeze(1)
    sin_3 = sin_2.unsqueeze(1)
    mul_8 = query_states * cos_3
    x1 = query_states[(Ellipsis, slice(None, 64, None))]
    x2 = query_states[(Ellipsis, slice(64, None, None))];  query_states = None
    neg = -x2;  x2 = None
    cat_1 = torch.cat((neg, x1), dim = -1);  neg = x1 = None
    mul_9 = cat_1 * sin_3;  cat_1 = None
    q_embed = mul_8 + mul_9;  mul_8 = mul_9 = None
    mul_10 = key_states * cos_3;  cos_3 = None
    x1_1 = key_states[(Ellipsis, slice(None, 64, None))]
    x2_1 = key_states[(Ellipsis, slice(64, None, None))];  key_states = None
    neg_1 = -x2_1;  x2_1 = None
    cat_2 = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
    mul_11 = cat_2 * sin_3;  cat_2 = sin_3 = None
    k_embed = mul_10 + mul_11;  mul_10 = mul_11 = None
    size_6 = q_embed.size()
    getitem_29 = size_6[2];  size_6 = None
    q = q_embed.contiguous();  q_embed = None
    k = k_embed.contiguous();  k_embed = None
    v = value_states.contiguous();  value_states = None
    npu_fused_infer_attention_score_default = torch.ops.npu.npu_fused_infer_attention_score.default(q, k, v, atten_mask = l_causal_mask_, actual_seq_lengths = [getitem_29], actual_seq_lengths_kv = [getitem_29], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0);  getitem_29 = None
    output = npu_fused_infer_attention_score_default[0];  npu_fused_infer_attention_score_default = None
    attn_output = output.transpose(1, 2)
    reshape = attn_output.reshape(1, getitem_5, -1);  attn_output = getitem_5 = None
    attn_output_1 = reshape.contiguous();  reshape = None
    l__self___model_model_layers_0_self_attn_o_proj_weight = self.L__self___model_model_layers_0_self_attn_o_proj_weight
    attn_output_2 = torch._C._nn.linear(attn_output_1, l__self___model_model_layers_0_self_attn_o_proj_weight, None);  attn_output_1 = l__self___model_model_layers_0_self_attn_o_proj_weight = None
    hidden_states_7 = inputs_embeds + attn_output_2;  inputs_embeds = attn_output_2 = None
    hidden_states_8 = hidden_states_7.to(torch.float32)
    pow_4 = hidden_states_8.pow(2)
    variance_3 = pow_4.mean(-1, keepdim = True);  pow_4 = None
    add_6 = variance_3 + 1e-06;  variance_3 = None
    rsqrt_3 = torch.rsqrt(add_6);  add_6 = None
    hidden_states_9 = hidden_states_8 * rsqrt_3;  hidden_states_8 = rsqrt_3 = None
    l__self___model_model_layers_0_post_attention_layernorm_weight = self.L__self___model_model_layers_0_post_attention_layernorm_weight
    to_10 = hidden_states_9.to(torch.float16);  hidden_states_9 = None
    hidden_states_10 = l__self___model_model_layers_0_post_attention_layernorm_weight * to_10;  l__self___model_model_layers_0_post_attention_layernorm_weight = to_10 = None
    l__self___model_model_layers_0_mlp_gate_proj_weight = self.L__self___model_model_layers_0_mlp_gate_proj_weight
    linear_4 = torch._C._nn.linear(hidden_states_10, l__self___model_model_layers_0_mlp_gate_proj_weight, None);  l__self___model_model_layers_0_mlp_gate_proj_weight = None
    silu = torch.nn.functional.silu(linear_4);  linear_4 = None
    l__self___model_model_layers_0_mlp_up_proj_weight = self.L__self___model_model_layers_0_mlp_up_proj_weight
    linear_5 = torch._C._nn.linear(hidden_states_10, l__self___model_model_layers_0_mlp_up_proj_weight, None);  hidden_states_10 = l__self___model_model_layers_0_mlp_up_proj_weight = None
    mul_14 = silu * linear_5;  silu = linear_5 = None
    l__self___model_model_layers_0_mlp_down_proj_weight = self.L__self___model_model_layers_0_mlp_down_proj_weight
    down_proj = torch._C._nn.linear(mul_14, l__self___model_model_layers_0_mlp_down_proj_weight, None);  mul_14 = l__self___model_model_layers_0_mlp_down_proj_weight = None
    hidden_states_11 = hidden_states_7 + down_proj;  hidden_states_7 = down_proj = None
    hidden_states_12 = hidden_states_11.to(torch.float32)
    pow_5 = hidden_states_12.pow(2)
    variance_4 = pow_5.mean(-1, keepdim = True);  pow_5 = None
    add_8 = variance_4 + 1e-06;  variance_4 = None
    rsqrt_4 = torch.rsqrt(add_8);  add_8 = None
    hidden_states_13 = hidden_states_12 * rsqrt_4;  hidden_states_12 = rsqrt_4 = None
    l__self___model_model_layers_1_input_layernorm_weight = self.L__self___model_model_layers_1_input_layernorm_weight
    to_12 = hidden_states_13.to(torch.float16);  hidden_states_13 = None
    hidden_states_14 = l__self___model_model_layers_1_input_layernorm_weight * to_12;  l__self___model_model_layers_1_input_layernorm_weight = to_12 = None
    l__self___model_model_layers_1_self_attn_q_proj_weight = self.L__self___model_model_layers_1_self_attn_q_proj_weight
    linear_7 = torch._C._nn.linear(hidden_states_14, l__self___model_model_layers_1_self_attn_q_proj_weight, None);  l__self___model_model_layers_1_self_attn_q_proj_weight = None
    view_3 = linear_7.view((1, 2048, -1, 128));  linear_7 = None
    hidden_states_15 = view_3.to(torch.float32);  view_3 = None
    pow_6 = hidden_states_15.pow(2)
    variance_5 = pow_6.mean(-1, keepdim = True);  pow_6 = None
    add_9 = variance_5 + 1e-06;  variance_5 = None
    rsqrt_5 = torch.rsqrt(add_9);  add_9 = None
    hidden_states_16 = hidden_states_15 * rsqrt_5;  hidden_states_15 = rsqrt_5 = None
    l__self___model_model_layers_1_self_attn_q_norm_weight = self.L__self___model_model_layers_1_self_attn_q_norm_weight
    to_14 = hidden_states_16.to(torch.float16);  hidden_states_16 = None
    mul_18 = l__self___model_model_layers_1_self_attn_q_norm_weight * to_14;  l__self___model_model_layers_1_self_attn_q_norm_weight = to_14 = None
    query_states_1 = mul_18.transpose(1, 2);  mul_18 = None
    l__self___model_model_layers_1_self_attn_k_proj_weight = self.L__self___model_model_layers_1_self_attn_k_proj_weight
    linear_8 = torch._C._nn.linear(hidden_states_14, l__self___model_model_layers_1_self_attn_k_proj_weight, None);  l__self___model_model_layers_1_self_attn_k_proj_weight = None
    view_4 = linear_8.view((1, 2048, -1, 128));  linear_8 = None
    hidden_states_17 = view_4.to(torch.float32);  view_4 = None
    pow_7 = hidden_states_17.pow(2)
    variance_6 = pow_7.mean(-1, keepdim = True);  pow_7 = None
    add_10 = variance_6 + 1e-06;  variance_6 = None
    rsqrt_6 = torch.rsqrt(add_10);  add_10 = None
    hidden_states_18 = hidden_states_17 * rsqrt_6;  hidden_states_17 = rsqrt_6 = None
    l__self___model_model_layers_1_self_attn_k_norm_weight = self.L__self___model_model_layers_1_self_attn_k_norm_weight
    to_16 = hidden_states_18.to(torch.float16);  hidden_states_18 = None
    mul_20 = l__self___model_model_layers_1_self_attn_k_norm_weight * to_16;  l__self___model_model_layers_1_self_attn_k_norm_weight = to_16 = None
    key_states_1 = mul_20.transpose(1, 2);  mul_20 = None
    l__self___model_model_layers_1_self_attn_v_proj_weight = self.L__self___model_model_layers_1_self_attn_v_proj_weight
    linear_9 = torch._C._nn.linear(hidden_states_14, l__self___model_model_layers_1_self_attn_v_proj_weight, None);  hidden_states_14 = l__self___model_model_layers_1_self_attn_v_proj_weight = None
    view_5 = linear_9.view((1, 2048, -1, 128));  linear_9 = None
    value_states_1 = view_5.transpose(1, 2);  view_5 = None
    cos_4 = cos_2.unsqueeze(1)
    sin_4 = sin_2.unsqueeze(1)
    mul_21 = query_states_1 * cos_4
    x1_2 = query_states_1[(Ellipsis, slice(None, 64, None))]
    x2_2 = query_states_1[(Ellipsis, slice(64, None, None))];  query_states_1 = None
    neg_2 = -x2_2;  x2_2 = None
    cat_3 = torch.cat((neg_2, x1_2), dim = -1);  neg_2 = x1_2 = None
    mul_22 = cat_3 * sin_4;  cat_3 = None
    q_embed_1 = mul_21 + mul_22;  mul_21 = mul_22 = None
    mul_23 = key_states_1 * cos_4;  cos_4 = None
    x1_3 = key_states_1[(Ellipsis, slice(None, 64, None))]
    x2_3 = key_states_1[(Ellipsis, slice(64, None, None))];  key_states_1 = None
    neg_3 = -x2_3;  x2_3 = None
    cat_4 = torch.cat((neg_3, x1_3), dim = -1);  neg_3 = x1_3 = None
    mul_24 = cat_4 * sin_4;  cat_4 = sin_4 = None
    k_embed_1 = mul_23 + mul_24;  mul_23 = mul_24 = None
    q_1 = q_embed_1.contiguous();  q_embed_1 = None
    k_1 = k_embed_1.contiguous();  k_embed_1 = None
    v_1 = value_states_1.contiguous();  value_states_1 = None
    npu_fused_infer_attention_score_default_1 = torch.ops.npu.npu_fused_infer_attention_score.default(q_1, k_1, v_1, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_1 = npu_fused_infer_attention_score_default_1[0];  npu_fused_infer_attention_score_default_1 = None
    attn_output_3 = output_1.transpose(1, 2)
    reshape_1 = attn_output_3.reshape(1, 2048, -1);  attn_output_3 = None
    attn_output_4 = reshape_1.contiguous();  reshape_1 = None
    l__self___model_model_layers_1_self_attn_o_proj_weight = self.L__self___model_model_layers_1_self_attn_o_proj_weight
    attn_output_5 = torch._C._nn.linear(attn_output_4, l__self___model_model_layers_1_self_attn_o_proj_weight, None);  attn_output_4 = l__self___model_model_layers_1_self_attn_o_proj_weight = None
    hidden_states_19 = hidden_states_11 + attn_output_5;  hidden_states_11 = attn_output_5 = None
    hidden_states_20 = hidden_states_19.to(torch.float32)
    pow_8 = hidden_states_20.pow(2)
    variance_7 = pow_8.mean(-1, keepdim = True);  pow_8 = None
    add_14 = variance_7 + 1e-06;  variance_7 = None
    rsqrt_7 = torch.rsqrt(add_14);  add_14 = None
    hidden_states_21 = hidden_states_20 * rsqrt_7;  hidden_states_20 = rsqrt_7 = None
    l__self___model_model_layers_1_post_attention_layernorm_weight = self.L__self___model_model_layers_1_post_attention_layernorm_weight
    to_18 = hidden_states_21.to(torch.float16);  hidden_states_21 = None
    hidden_states_22 = l__self___model_model_layers_1_post_attention_layernorm_weight * to_18;  l__self___model_model_layers_1_post_attention_layernorm_weight = to_18 = None
    l__self___model_model_layers_1_mlp_gate_proj_weight = self.L__self___model_model_layers_1_mlp_gate_proj_weight
    linear_11 = torch._C._nn.linear(hidden_states_22, l__self___model_model_layers_1_mlp_gate_proj_weight, None);  l__self___model_model_layers_1_mlp_gate_proj_weight = None
    silu_1 = torch.nn.functional.silu(linear_11);  linear_11 = None
    l__self___model_model_layers_1_mlp_up_proj_weight = self.L__self___model_model_layers_1_mlp_up_proj_weight
    linear_12 = torch._C._nn.linear(hidden_states_22, l__self___model_model_layers_1_mlp_up_proj_weight, None);  hidden_states_22 = l__self___model_model_layers_1_mlp_up_proj_weight = None
    mul_27 = silu_1 * linear_12;  silu_1 = linear_12 = None
    l__self___model_model_layers_1_mlp_down_proj_weight = self.L__self___model_model_layers_1_mlp_down_proj_weight
    down_proj_1 = torch._C._nn.linear(mul_27, l__self___model_model_layers_1_mlp_down_proj_weight, None);  mul_27 = l__self___model_model_layers_1_mlp_down_proj_weight = None
    hidden_states_23 = hidden_states_19 + down_proj_1;  hidden_states_19 = down_proj_1 = None
    hidden_states_24 = hidden_states_23.to(torch.float32)
    pow_9 = hidden_states_24.pow(2)
    variance_8 = pow_9.mean(-1, keepdim = True);  pow_9 = None
    add_16 = variance_8 + 1e-06;  variance_8 = None
    rsqrt_8 = torch.rsqrt(add_16);  add_16 = None
    hidden_states_25 = hidden_states_24 * rsqrt_8;  hidden_states_24 = rsqrt_8 = None
    l__self___model_model_layers_2_input_layernorm_weight = self.L__self___model_model_layers_2_input_layernorm_weight
    to_20 = hidden_states_25.to(torch.float16);  hidden_states_25 = None
    hidden_states_26 = l__self___model_model_layers_2_input_layernorm_weight * to_20;  l__self___model_model_layers_2_input_layernorm_weight = to_20 = None
    l__self___model_model_layers_2_self_attn_q_proj_weight = self.L__self___model_model_layers_2_self_attn_q_proj_weight
    linear_14 = torch._C._nn.linear(hidden_states_26, l__self___model_model_layers_2_self_attn_q_proj_weight, None);  l__self___model_model_layers_2_self_attn_q_proj_weight = None
    view_6 = linear_14.view((1, 2048, -1, 128));  linear_14 = None
    hidden_states_27 = view_6.to(torch.float32);  view_6 = None
    pow_10 = hidden_states_27.pow(2)
    variance_9 = pow_10.mean(-1, keepdim = True);  pow_10 = None
    add_17 = variance_9 + 1e-06;  variance_9 = None
    rsqrt_9 = torch.rsqrt(add_17);  add_17 = None
    hidden_states_28 = hidden_states_27 * rsqrt_9;  hidden_states_27 = rsqrt_9 = None
    l__self___model_model_layers_2_self_attn_q_norm_weight = self.L__self___model_model_layers_2_self_attn_q_norm_weight
    to_22 = hidden_states_28.to(torch.float16);  hidden_states_28 = None
    mul_31 = l__self___model_model_layers_2_self_attn_q_norm_weight * to_22;  l__self___model_model_layers_2_self_attn_q_norm_weight = to_22 = None
    query_states_2 = mul_31.transpose(1, 2);  mul_31 = None
    l__self___model_model_layers_2_self_attn_k_proj_weight = self.L__self___model_model_layers_2_self_attn_k_proj_weight
    linear_15 = torch._C._nn.linear(hidden_states_26, l__self___model_model_layers_2_self_attn_k_proj_weight, None);  l__self___model_model_layers_2_self_attn_k_proj_weight = None
    view_7 = linear_15.view((1, 2048, -1, 128));  linear_15 = None
    hidden_states_29 = view_7.to(torch.float32);  view_7 = None
    pow_11 = hidden_states_29.pow(2)
    variance_10 = pow_11.mean(-1, keepdim = True);  pow_11 = None
    add_18 = variance_10 + 1e-06;  variance_10 = None
    rsqrt_10 = torch.rsqrt(add_18);  add_18 = None
    hidden_states_30 = hidden_states_29 * rsqrt_10;  hidden_states_29 = rsqrt_10 = None
    l__self___model_model_layers_2_self_attn_k_norm_weight = self.L__self___model_model_layers_2_self_attn_k_norm_weight
    to_24 = hidden_states_30.to(torch.float16);  hidden_states_30 = None
    mul_33 = l__self___model_model_layers_2_self_attn_k_norm_weight * to_24;  l__self___model_model_layers_2_self_attn_k_norm_weight = to_24 = None
    key_states_2 = mul_33.transpose(1, 2);  mul_33 = None
    l__self___model_model_layers_2_self_attn_v_proj_weight = self.L__self___model_model_layers_2_self_attn_v_proj_weight
    linear_16 = torch._C._nn.linear(hidden_states_26, l__self___model_model_layers_2_self_attn_v_proj_weight, None);  hidden_states_26 = l__self___model_model_layers_2_self_attn_v_proj_weight = None
    view_8 = linear_16.view((1, 2048, -1, 128));  linear_16 = None
    value_states_2 = view_8.transpose(1, 2);  view_8 = None
    cos_5 = cos_2.unsqueeze(1)
    sin_5 = sin_2.unsqueeze(1)
    mul_34 = query_states_2 * cos_5
    x1_4 = query_states_2[(Ellipsis, slice(None, 64, None))]
    x2_4 = query_states_2[(Ellipsis, slice(64, None, None))];  query_states_2 = None
    neg_4 = -x2_4;  x2_4 = None
    cat_5 = torch.cat((neg_4, x1_4), dim = -1);  neg_4 = x1_4 = None
    mul_35 = cat_5 * sin_5;  cat_5 = None
    q_embed_2 = mul_34 + mul_35;  mul_34 = mul_35 = None
    mul_36 = key_states_2 * cos_5;  cos_5 = None
    x1_5 = key_states_2[(Ellipsis, slice(None, 64, None))]
    x2_5 = key_states_2[(Ellipsis, slice(64, None, None))];  key_states_2 = None
    neg_5 = -x2_5;  x2_5 = None
    cat_6 = torch.cat((neg_5, x1_5), dim = -1);  neg_5 = x1_5 = None
    mul_37 = cat_6 * sin_5;  cat_6 = sin_5 = None
    k_embed_2 = mul_36 + mul_37;  mul_36 = mul_37 = None
    q_2 = q_embed_2.contiguous();  q_embed_2 = None
    k_2 = k_embed_2.contiguous();  k_embed_2 = None
    v_2 = value_states_2.contiguous();  value_states_2 = None
    npu_fused_infer_attention_score_default_2 = torch.ops.npu.npu_fused_infer_attention_score.default(q_2, k_2, v_2, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_2 = npu_fused_infer_attention_score_default_2[0];  npu_fused_infer_attention_score_default_2 = None
    attn_output_6 = output_2.transpose(1, 2)
    reshape_2 = attn_output_6.reshape(1, 2048, -1);  attn_output_6 = None
    attn_output_7 = reshape_2.contiguous();  reshape_2 = None
    l__self___model_model_layers_2_self_attn_o_proj_weight = self.L__self___model_model_layers_2_self_attn_o_proj_weight
    attn_output_8 = torch._C._nn.linear(attn_output_7, l__self___model_model_layers_2_self_attn_o_proj_weight, None);  attn_output_7 = l__self___model_model_layers_2_self_attn_o_proj_weight = None
    hidden_states_31 = hidden_states_23 + attn_output_8;  hidden_states_23 = attn_output_8 = None
    hidden_states_32 = hidden_states_31.to(torch.float32)
    pow_12 = hidden_states_32.pow(2)
    variance_11 = pow_12.mean(-1, keepdim = True);  pow_12 = None
    add_22 = variance_11 + 1e-06;  variance_11 = None
    rsqrt_11 = torch.rsqrt(add_22);  add_22 = None
    hidden_states_33 = hidden_states_32 * rsqrt_11;  hidden_states_32 = rsqrt_11 = None
    l__self___model_model_layers_2_post_attention_layernorm_weight = self.L__self___model_model_layers_2_post_attention_layernorm_weight
    to_26 = hidden_states_33.to(torch.float16);  hidden_states_33 = None
    hidden_states_34 = l__self___model_model_layers_2_post_attention_layernorm_weight * to_26;  l__self___model_model_layers_2_post_attention_layernorm_weight = to_26 = None
    l__self___model_model_layers_2_mlp_gate_proj_weight = self.L__self___model_model_layers_2_mlp_gate_proj_weight
    linear_18 = torch._C._nn.linear(hidden_states_34, l__self___model_model_layers_2_mlp_gate_proj_weight, None);  l__self___model_model_layers_2_mlp_gate_proj_weight = None
    silu_2 = torch.nn.functional.silu(linear_18);  linear_18 = None
    l__self___model_model_layers_2_mlp_up_proj_weight = self.L__self___model_model_layers_2_mlp_up_proj_weight
    linear_19 = torch._C._nn.linear(hidden_states_34, l__self___model_model_layers_2_mlp_up_proj_weight, None);  hidden_states_34 = l__self___model_model_layers_2_mlp_up_proj_weight = None
    mul_40 = silu_2 * linear_19;  silu_2 = linear_19 = None
    l__self___model_model_layers_2_mlp_down_proj_weight = self.L__self___model_model_layers_2_mlp_down_proj_weight
    down_proj_2 = torch._C._nn.linear(mul_40, l__self___model_model_layers_2_mlp_down_proj_weight, None);  mul_40 = l__self___model_model_layers_2_mlp_down_proj_weight = None
    hidden_states_35 = hidden_states_31 + down_proj_2;  hidden_states_31 = down_proj_2 = None
    hidden_states_36 = hidden_states_35.to(torch.float32)
    pow_13 = hidden_states_36.pow(2)
    variance_12 = pow_13.mean(-1, keepdim = True);  pow_13 = None
    add_24 = variance_12 + 1e-06;  variance_12 = None
    rsqrt_12 = torch.rsqrt(add_24);  add_24 = None
    hidden_states_37 = hidden_states_36 * rsqrt_12;  hidden_states_36 = rsqrt_12 = None
    l__self___model_model_layers_3_input_layernorm_weight = self.L__self___model_model_layers_3_input_layernorm_weight
    to_28 = hidden_states_37.to(torch.float16);  hidden_states_37 = None
    hidden_states_38 = l__self___model_model_layers_3_input_layernorm_weight * to_28;  l__self___model_model_layers_3_input_layernorm_weight = to_28 = None
    l__self___model_model_layers_3_self_attn_q_proj_weight = self.L__self___model_model_layers_3_self_attn_q_proj_weight
    linear_21 = torch._C._nn.linear(hidden_states_38, l__self___model_model_layers_3_self_attn_q_proj_weight, None);  l__self___model_model_layers_3_self_attn_q_proj_weight = None
    view_9 = linear_21.view((1, 2048, -1, 128));  linear_21 = None
    hidden_states_39 = view_9.to(torch.float32);  view_9 = None
    pow_14 = hidden_states_39.pow(2)
    variance_13 = pow_14.mean(-1, keepdim = True);  pow_14 = None
    add_25 = variance_13 + 1e-06;  variance_13 = None
    rsqrt_13 = torch.rsqrt(add_25);  add_25 = None
    hidden_states_40 = hidden_states_39 * rsqrt_13;  hidden_states_39 = rsqrt_13 = None
    l__self___model_model_layers_3_self_attn_q_norm_weight = self.L__self___model_model_layers_3_self_attn_q_norm_weight
    to_30 = hidden_states_40.to(torch.float16);  hidden_states_40 = None
    mul_44 = l__self___model_model_layers_3_self_attn_q_norm_weight * to_30;  l__self___model_model_layers_3_self_attn_q_norm_weight = to_30 = None
    query_states_3 = mul_44.transpose(1, 2);  mul_44 = None
    l__self___model_model_layers_3_self_attn_k_proj_weight = self.L__self___model_model_layers_3_self_attn_k_proj_weight
    linear_22 = torch._C._nn.linear(hidden_states_38, l__self___model_model_layers_3_self_attn_k_proj_weight, None);  l__self___model_model_layers_3_self_attn_k_proj_weight = None
    view_10 = linear_22.view((1, 2048, -1, 128));  linear_22 = None
    hidden_states_41 = view_10.to(torch.float32);  view_10 = None
    pow_15 = hidden_states_41.pow(2)
    variance_14 = pow_15.mean(-1, keepdim = True);  pow_15 = None
    add_26 = variance_14 + 1e-06;  variance_14 = None
    rsqrt_14 = torch.rsqrt(add_26);  add_26 = None
    hidden_states_42 = hidden_states_41 * rsqrt_14;  hidden_states_41 = rsqrt_14 = None
    l__self___model_model_layers_3_self_attn_k_norm_weight = self.L__self___model_model_layers_3_self_attn_k_norm_weight
    to_32 = hidden_states_42.to(torch.float16);  hidden_states_42 = None
    mul_46 = l__self___model_model_layers_3_self_attn_k_norm_weight * to_32;  l__self___model_model_layers_3_self_attn_k_norm_weight = to_32 = None
    key_states_3 = mul_46.transpose(1, 2);  mul_46 = None
    l__self___model_model_layers_3_self_attn_v_proj_weight = self.L__self___model_model_layers_3_self_attn_v_proj_weight
    linear_23 = torch._C._nn.linear(hidden_states_38, l__self___model_model_layers_3_self_attn_v_proj_weight, None);  hidden_states_38 = l__self___model_model_layers_3_self_attn_v_proj_weight = None
    view_11 = linear_23.view((1, 2048, -1, 128));  linear_23 = None
    value_states_3 = view_11.transpose(1, 2);  view_11 = None
    cos_6 = cos_2.unsqueeze(1)
    sin_6 = sin_2.unsqueeze(1)
    mul_47 = query_states_3 * cos_6
    x1_6 = query_states_3[(Ellipsis, slice(None, 64, None))]
    x2_6 = query_states_3[(Ellipsis, slice(64, None, None))];  query_states_3 = None
    neg_6 = -x2_6;  x2_6 = None
    cat_7 = torch.cat((neg_6, x1_6), dim = -1);  neg_6 = x1_6 = None
    mul_48 = cat_7 * sin_6;  cat_7 = None
    q_embed_3 = mul_47 + mul_48;  mul_47 = mul_48 = None
    mul_49 = key_states_3 * cos_6;  cos_6 = None
    x1_7 = key_states_3[(Ellipsis, slice(None, 64, None))]
    x2_7 = key_states_3[(Ellipsis, slice(64, None, None))];  key_states_3 = None
    neg_7 = -x2_7;  x2_7 = None
    cat_8 = torch.cat((neg_7, x1_7), dim = -1);  neg_7 = x1_7 = None
    mul_50 = cat_8 * sin_6;  cat_8 = sin_6 = None
    k_embed_3 = mul_49 + mul_50;  mul_49 = mul_50 = None
    q_3 = q_embed_3.contiguous();  q_embed_3 = None
    k_3 = k_embed_3.contiguous();  k_embed_3 = None
    v_3 = value_states_3.contiguous();  value_states_3 = None
    npu_fused_infer_attention_score_default_3 = torch.ops.npu.npu_fused_infer_attention_score.default(q_3, k_3, v_3, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_3 = npu_fused_infer_attention_score_default_3[0];  npu_fused_infer_attention_score_default_3 = None
    attn_output_9 = output_3.transpose(1, 2)
    reshape_3 = attn_output_9.reshape(1, 2048, -1);  attn_output_9 = None
    attn_output_10 = reshape_3.contiguous();  reshape_3 = None
    l__self___model_model_layers_3_self_attn_o_proj_weight = self.L__self___model_model_layers_3_self_attn_o_proj_weight
    attn_output_11 = torch._C._nn.linear(attn_output_10, l__self___model_model_layers_3_self_attn_o_proj_weight, None);  attn_output_10 = l__self___model_model_layers_3_self_attn_o_proj_weight = None
    hidden_states_43 = hidden_states_35 + attn_output_11;  hidden_states_35 = attn_output_11 = None
    hidden_states_44 = hidden_states_43.to(torch.float32)
    pow_16 = hidden_states_44.pow(2)
    variance_15 = pow_16.mean(-1, keepdim = True);  pow_16 = None
    add_30 = variance_15 + 1e-06;  variance_15 = None
    rsqrt_15 = torch.rsqrt(add_30);  add_30 = None
    hidden_states_45 = hidden_states_44 * rsqrt_15;  hidden_states_44 = rsqrt_15 = None
    l__self___model_model_layers_3_post_attention_layernorm_weight = self.L__self___model_model_layers_3_post_attention_layernorm_weight
    to_34 = hidden_states_45.to(torch.float16);  hidden_states_45 = None
    hidden_states_46 = l__self___model_model_layers_3_post_attention_layernorm_weight * to_34;  l__self___model_model_layers_3_post_attention_layernorm_weight = to_34 = None
    l__self___model_model_layers_3_mlp_gate_proj_weight = self.L__self___model_model_layers_3_mlp_gate_proj_weight
    linear_25 = torch._C._nn.linear(hidden_states_46, l__self___model_model_layers_3_mlp_gate_proj_weight, None);  l__self___model_model_layers_3_mlp_gate_proj_weight = None
    silu_3 = torch.nn.functional.silu(linear_25);  linear_25 = None
    l__self___model_model_layers_3_mlp_up_proj_weight = self.L__self___model_model_layers_3_mlp_up_proj_weight
    linear_26 = torch._C._nn.linear(hidden_states_46, l__self___model_model_layers_3_mlp_up_proj_weight, None);  hidden_states_46 = l__self___model_model_layers_3_mlp_up_proj_weight = None
    mul_53 = silu_3 * linear_26;  silu_3 = linear_26 = None
    l__self___model_model_layers_3_mlp_down_proj_weight = self.L__self___model_model_layers_3_mlp_down_proj_weight
    down_proj_3 = torch._C._nn.linear(mul_53, l__self___model_model_layers_3_mlp_down_proj_weight, None);  mul_53 = l__self___model_model_layers_3_mlp_down_proj_weight = None
    hidden_states_47 = hidden_states_43 + down_proj_3;  hidden_states_43 = down_proj_3 = None
    hidden_states_48 = hidden_states_47.to(torch.float32)
    pow_17 = hidden_states_48.pow(2)
    variance_16 = pow_17.mean(-1, keepdim = True);  pow_17 = None
    add_32 = variance_16 + 1e-06;  variance_16 = None
    rsqrt_16 = torch.rsqrt(add_32);  add_32 = None
    hidden_states_49 = hidden_states_48 * rsqrt_16;  hidden_states_48 = rsqrt_16 = None
    l__self___model_model_layers_4_input_layernorm_weight = self.L__self___model_model_layers_4_input_layernorm_weight
    to_36 = hidden_states_49.to(torch.float16);  hidden_states_49 = None
    hidden_states_50 = l__self___model_model_layers_4_input_layernorm_weight * to_36;  l__self___model_model_layers_4_input_layernorm_weight = to_36 = None
    l__self___model_model_layers_4_self_attn_q_proj_weight = self.L__self___model_model_layers_4_self_attn_q_proj_weight
    linear_28 = torch._C._nn.linear(hidden_states_50, l__self___model_model_layers_4_self_attn_q_proj_weight, None);  l__self___model_model_layers_4_self_attn_q_proj_weight = None
    view_12 = linear_28.view((1, 2048, -1, 128));  linear_28 = None
    hidden_states_51 = view_12.to(torch.float32);  view_12 = None
    pow_18 = hidden_states_51.pow(2)
    variance_17 = pow_18.mean(-1, keepdim = True);  pow_18 = None
    add_33 = variance_17 + 1e-06;  variance_17 = None
    rsqrt_17 = torch.rsqrt(add_33);  add_33 = None
    hidden_states_52 = hidden_states_51 * rsqrt_17;  hidden_states_51 = rsqrt_17 = None
    l__self___model_model_layers_4_self_attn_q_norm_weight = self.L__self___model_model_layers_4_self_attn_q_norm_weight
    to_38 = hidden_states_52.to(torch.float16);  hidden_states_52 = None
    mul_57 = l__self___model_model_layers_4_self_attn_q_norm_weight * to_38;  l__self___model_model_layers_4_self_attn_q_norm_weight = to_38 = None
    query_states_4 = mul_57.transpose(1, 2);  mul_57 = None
    l__self___model_model_layers_4_self_attn_k_proj_weight = self.L__self___model_model_layers_4_self_attn_k_proj_weight
    linear_29 = torch._C._nn.linear(hidden_states_50, l__self___model_model_layers_4_self_attn_k_proj_weight, None);  l__self___model_model_layers_4_self_attn_k_proj_weight = None
    view_13 = linear_29.view((1, 2048, -1, 128));  linear_29 = None
    hidden_states_53 = view_13.to(torch.float32);  view_13 = None
    pow_19 = hidden_states_53.pow(2)
    variance_18 = pow_19.mean(-1, keepdim = True);  pow_19 = None
    add_34 = variance_18 + 1e-06;  variance_18 = None
    rsqrt_18 = torch.rsqrt(add_34);  add_34 = None
    hidden_states_54 = hidden_states_53 * rsqrt_18;  hidden_states_53 = rsqrt_18 = None
    l__self___model_model_layers_4_self_attn_k_norm_weight = self.L__self___model_model_layers_4_self_attn_k_norm_weight
    to_40 = hidden_states_54.to(torch.float16);  hidden_states_54 = None
    mul_59 = l__self___model_model_layers_4_self_attn_k_norm_weight * to_40;  l__self___model_model_layers_4_self_attn_k_norm_weight = to_40 = None
    key_states_4 = mul_59.transpose(1, 2);  mul_59 = None
    l__self___model_model_layers_4_self_attn_v_proj_weight = self.L__self___model_model_layers_4_self_attn_v_proj_weight
    linear_30 = torch._C._nn.linear(hidden_states_50, l__self___model_model_layers_4_self_attn_v_proj_weight, None);  hidden_states_50 = l__self___model_model_layers_4_self_attn_v_proj_weight = None
    view_14 = linear_30.view((1, 2048, -1, 128));  linear_30 = None
    value_states_4 = view_14.transpose(1, 2);  view_14 = None
    cos_7 = cos_2.unsqueeze(1)
    sin_7 = sin_2.unsqueeze(1)
    mul_60 = query_states_4 * cos_7
    x1_8 = query_states_4[(Ellipsis, slice(None, 64, None))]
    x2_8 = query_states_4[(Ellipsis, slice(64, None, None))];  query_states_4 = None
    neg_8 = -x2_8;  x2_8 = None
    cat_9 = torch.cat((neg_8, x1_8), dim = -1);  neg_8 = x1_8 = None
    mul_61 = cat_9 * sin_7;  cat_9 = None
    q_embed_4 = mul_60 + mul_61;  mul_60 = mul_61 = None
    mul_62 = key_states_4 * cos_7;  cos_7 = None
    x1_9 = key_states_4[(Ellipsis, slice(None, 64, None))]
    x2_9 = key_states_4[(Ellipsis, slice(64, None, None))];  key_states_4 = None
    neg_9 = -x2_9;  x2_9 = None
    cat_10 = torch.cat((neg_9, x1_9), dim = -1);  neg_9 = x1_9 = None
    mul_63 = cat_10 * sin_7;  cat_10 = sin_7 = None
    k_embed_4 = mul_62 + mul_63;  mul_62 = mul_63 = None
    q_4 = q_embed_4.contiguous();  q_embed_4 = None
    k_4 = k_embed_4.contiguous();  k_embed_4 = None
    v_4 = value_states_4.contiguous();  value_states_4 = None
    npu_fused_infer_attention_score_default_4 = torch.ops.npu.npu_fused_infer_attention_score.default(q_4, k_4, v_4, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_4 = npu_fused_infer_attention_score_default_4[0];  npu_fused_infer_attention_score_default_4 = None
    attn_output_12 = output_4.transpose(1, 2)
    reshape_4 = attn_output_12.reshape(1, 2048, -1);  attn_output_12 = None
    attn_output_13 = reshape_4.contiguous();  reshape_4 = None
    l__self___model_model_layers_4_self_attn_o_proj_weight = self.L__self___model_model_layers_4_self_attn_o_proj_weight
    attn_output_14 = torch._C._nn.linear(attn_output_13, l__self___model_model_layers_4_self_attn_o_proj_weight, None);  attn_output_13 = l__self___model_model_layers_4_self_attn_o_proj_weight = None
    hidden_states_55 = hidden_states_47 + attn_output_14;  hidden_states_47 = attn_output_14 = None
    hidden_states_56 = hidden_states_55.to(torch.float32)
    pow_20 = hidden_states_56.pow(2)
    variance_19 = pow_20.mean(-1, keepdim = True);  pow_20 = None
    add_38 = variance_19 + 1e-06;  variance_19 = None
    rsqrt_19 = torch.rsqrt(add_38);  add_38 = None
    hidden_states_57 = hidden_states_56 * rsqrt_19;  hidden_states_56 = rsqrt_19 = None
    l__self___model_model_layers_4_post_attention_layernorm_weight = self.L__self___model_model_layers_4_post_attention_layernorm_weight
    to_42 = hidden_states_57.to(torch.float16);  hidden_states_57 = None
    hidden_states_58 = l__self___model_model_layers_4_post_attention_layernorm_weight * to_42;  l__self___model_model_layers_4_post_attention_layernorm_weight = to_42 = None
    l__self___model_model_layers_4_mlp_gate_proj_weight = self.L__self___model_model_layers_4_mlp_gate_proj_weight
    linear_32 = torch._C._nn.linear(hidden_states_58, l__self___model_model_layers_4_mlp_gate_proj_weight, None);  l__self___model_model_layers_4_mlp_gate_proj_weight = None
    silu_4 = torch.nn.functional.silu(linear_32);  linear_32 = None
    l__self___model_model_layers_4_mlp_up_proj_weight = self.L__self___model_model_layers_4_mlp_up_proj_weight
    linear_33 = torch._C._nn.linear(hidden_states_58, l__self___model_model_layers_4_mlp_up_proj_weight, None);  hidden_states_58 = l__self___model_model_layers_4_mlp_up_proj_weight = None
    mul_66 = silu_4 * linear_33;  silu_4 = linear_33 = None
    l__self___model_model_layers_4_mlp_down_proj_weight = self.L__self___model_model_layers_4_mlp_down_proj_weight
    down_proj_4 = torch._C._nn.linear(mul_66, l__self___model_model_layers_4_mlp_down_proj_weight, None);  mul_66 = l__self___model_model_layers_4_mlp_down_proj_weight = None
    hidden_states_59 = hidden_states_55 + down_proj_4;  hidden_states_55 = down_proj_4 = None
    hidden_states_60 = hidden_states_59.to(torch.float32)
    pow_21 = hidden_states_60.pow(2)
    variance_20 = pow_21.mean(-1, keepdim = True);  pow_21 = None
    add_40 = variance_20 + 1e-06;  variance_20 = None
    rsqrt_20 = torch.rsqrt(add_40);  add_40 = None
    hidden_states_61 = hidden_states_60 * rsqrt_20;  hidden_states_60 = rsqrt_20 = None
    l__self___model_model_layers_5_input_layernorm_weight = self.L__self___model_model_layers_5_input_layernorm_weight
    to_44 = hidden_states_61.to(torch.float16);  hidden_states_61 = None
    hidden_states_62 = l__self___model_model_layers_5_input_layernorm_weight * to_44;  l__self___model_model_layers_5_input_layernorm_weight = to_44 = None
    l__self___model_model_layers_5_self_attn_q_proj_weight = self.L__self___model_model_layers_5_self_attn_q_proj_weight
    linear_35 = torch._C._nn.linear(hidden_states_62, l__self___model_model_layers_5_self_attn_q_proj_weight, None);  l__self___model_model_layers_5_self_attn_q_proj_weight = None
    view_15 = linear_35.view((1, 2048, -1, 128));  linear_35 = None
    hidden_states_63 = view_15.to(torch.float32);  view_15 = None
    pow_22 = hidden_states_63.pow(2)
    variance_21 = pow_22.mean(-1, keepdim = True);  pow_22 = None
    add_41 = variance_21 + 1e-06;  variance_21 = None
    rsqrt_21 = torch.rsqrt(add_41);  add_41 = None
    hidden_states_64 = hidden_states_63 * rsqrt_21;  hidden_states_63 = rsqrt_21 = None
    l__self___model_model_layers_5_self_attn_q_norm_weight = self.L__self___model_model_layers_5_self_attn_q_norm_weight
    to_46 = hidden_states_64.to(torch.float16);  hidden_states_64 = None
    mul_70 = l__self___model_model_layers_5_self_attn_q_norm_weight * to_46;  l__self___model_model_layers_5_self_attn_q_norm_weight = to_46 = None
    query_states_5 = mul_70.transpose(1, 2);  mul_70 = None
    l__self___model_model_layers_5_self_attn_k_proj_weight = self.L__self___model_model_layers_5_self_attn_k_proj_weight
    linear_36 = torch._C._nn.linear(hidden_states_62, l__self___model_model_layers_5_self_attn_k_proj_weight, None);  l__self___model_model_layers_5_self_attn_k_proj_weight = None
    view_16 = linear_36.view((1, 2048, -1, 128));  linear_36 = None
    hidden_states_65 = view_16.to(torch.float32);  view_16 = None
    pow_23 = hidden_states_65.pow(2)
    variance_22 = pow_23.mean(-1, keepdim = True);  pow_23 = None
    add_42 = variance_22 + 1e-06;  variance_22 = None
    rsqrt_22 = torch.rsqrt(add_42);  add_42 = None
    hidden_states_66 = hidden_states_65 * rsqrt_22;  hidden_states_65 = rsqrt_22 = None
    l__self___model_model_layers_5_self_attn_k_norm_weight = self.L__self___model_model_layers_5_self_attn_k_norm_weight
    to_48 = hidden_states_66.to(torch.float16);  hidden_states_66 = None
    mul_72 = l__self___model_model_layers_5_self_attn_k_norm_weight * to_48;  l__self___model_model_layers_5_self_attn_k_norm_weight = to_48 = None
    key_states_5 = mul_72.transpose(1, 2);  mul_72 = None
    l__self___model_model_layers_5_self_attn_v_proj_weight = self.L__self___model_model_layers_5_self_attn_v_proj_weight
    linear_37 = torch._C._nn.linear(hidden_states_62, l__self___model_model_layers_5_self_attn_v_proj_weight, None);  hidden_states_62 = l__self___model_model_layers_5_self_attn_v_proj_weight = None
    view_17 = linear_37.view((1, 2048, -1, 128));  linear_37 = None
    value_states_5 = view_17.transpose(1, 2);  view_17 = None
    cos_8 = cos_2.unsqueeze(1)
    sin_8 = sin_2.unsqueeze(1)
    mul_73 = query_states_5 * cos_8
    x1_10 = query_states_5[(Ellipsis, slice(None, 64, None))]
    x2_10 = query_states_5[(Ellipsis, slice(64, None, None))];  query_states_5 = None
    neg_10 = -x2_10;  x2_10 = None
    cat_11 = torch.cat((neg_10, x1_10), dim = -1);  neg_10 = x1_10 = None
    mul_74 = cat_11 * sin_8;  cat_11 = None
    q_embed_5 = mul_73 + mul_74;  mul_73 = mul_74 = None
    mul_75 = key_states_5 * cos_8;  cos_8 = None
    x1_11 = key_states_5[(Ellipsis, slice(None, 64, None))]
    x2_11 = key_states_5[(Ellipsis, slice(64, None, None))];  key_states_5 = None
    neg_11 = -x2_11;  x2_11 = None
    cat_12 = torch.cat((neg_11, x1_11), dim = -1);  neg_11 = x1_11 = None
    mul_76 = cat_12 * sin_8;  cat_12 = sin_8 = None
    k_embed_5 = mul_75 + mul_76;  mul_75 = mul_76 = None
    q_5 = q_embed_5.contiguous();  q_embed_5 = None
    k_5 = k_embed_5.contiguous();  k_embed_5 = None
    v_5 = value_states_5.contiguous();  value_states_5 = None
    npu_fused_infer_attention_score_default_5 = torch.ops.npu.npu_fused_infer_attention_score.default(q_5, k_5, v_5, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_5 = npu_fused_infer_attention_score_default_5[0];  npu_fused_infer_attention_score_default_5 = None
    attn_output_15 = output_5.transpose(1, 2)
    reshape_5 = attn_output_15.reshape(1, 2048, -1);  attn_output_15 = None
    attn_output_16 = reshape_5.contiguous();  reshape_5 = None
    l__self___model_model_layers_5_self_attn_o_proj_weight = self.L__self___model_model_layers_5_self_attn_o_proj_weight
    attn_output_17 = torch._C._nn.linear(attn_output_16, l__self___model_model_layers_5_self_attn_o_proj_weight, None);  attn_output_16 = l__self___model_model_layers_5_self_attn_o_proj_weight = None
    hidden_states_67 = hidden_states_59 + attn_output_17;  hidden_states_59 = attn_output_17 = None
    hidden_states_68 = hidden_states_67.to(torch.float32)
    pow_24 = hidden_states_68.pow(2)
    variance_23 = pow_24.mean(-1, keepdim = True);  pow_24 = None
    add_46 = variance_23 + 1e-06;  variance_23 = None
    rsqrt_23 = torch.rsqrt(add_46);  add_46 = None
    hidden_states_69 = hidden_states_68 * rsqrt_23;  hidden_states_68 = rsqrt_23 = None
    l__self___model_model_layers_5_post_attention_layernorm_weight = self.L__self___model_model_layers_5_post_attention_layernorm_weight
    to_50 = hidden_states_69.to(torch.float16);  hidden_states_69 = None
    hidden_states_70 = l__self___model_model_layers_5_post_attention_layernorm_weight * to_50;  l__self___model_model_layers_5_post_attention_layernorm_weight = to_50 = None
    l__self___model_model_layers_5_mlp_gate_proj_weight = self.L__self___model_model_layers_5_mlp_gate_proj_weight
    linear_39 = torch._C._nn.linear(hidden_states_70, l__self___model_model_layers_5_mlp_gate_proj_weight, None);  l__self___model_model_layers_5_mlp_gate_proj_weight = None
    silu_5 = torch.nn.functional.silu(linear_39);  linear_39 = None
    l__self___model_model_layers_5_mlp_up_proj_weight = self.L__self___model_model_layers_5_mlp_up_proj_weight
    linear_40 = torch._C._nn.linear(hidden_states_70, l__self___model_model_layers_5_mlp_up_proj_weight, None);  hidden_states_70 = l__self___model_model_layers_5_mlp_up_proj_weight = None
    mul_79 = silu_5 * linear_40;  silu_5 = linear_40 = None
    l__self___model_model_layers_5_mlp_down_proj_weight = self.L__self___model_model_layers_5_mlp_down_proj_weight
    down_proj_5 = torch._C._nn.linear(mul_79, l__self___model_model_layers_5_mlp_down_proj_weight, None);  mul_79 = l__self___model_model_layers_5_mlp_down_proj_weight = None
    hidden_states_71 = hidden_states_67 + down_proj_5;  hidden_states_67 = down_proj_5 = None
    hidden_states_72 = hidden_states_71.to(torch.float32)
    pow_25 = hidden_states_72.pow(2)
    variance_24 = pow_25.mean(-1, keepdim = True);  pow_25 = None
    add_48 = variance_24 + 1e-06;  variance_24 = None
    rsqrt_24 = torch.rsqrt(add_48);  add_48 = None
    hidden_states_73 = hidden_states_72 * rsqrt_24;  hidden_states_72 = rsqrt_24 = None
    l__self___model_model_layers_6_input_layernorm_weight = self.L__self___model_model_layers_6_input_layernorm_weight
    to_52 = hidden_states_73.to(torch.float16);  hidden_states_73 = None
    hidden_states_74 = l__self___model_model_layers_6_input_layernorm_weight * to_52;  l__self___model_model_layers_6_input_layernorm_weight = to_52 = None
    l__self___model_model_layers_6_self_attn_q_proj_weight = self.L__self___model_model_layers_6_self_attn_q_proj_weight
    linear_42 = torch._C._nn.linear(hidden_states_74, l__self___model_model_layers_6_self_attn_q_proj_weight, None);  l__self___model_model_layers_6_self_attn_q_proj_weight = None
    view_18 = linear_42.view((1, 2048, -1, 128));  linear_42 = None
    hidden_states_75 = view_18.to(torch.float32);  view_18 = None
    pow_26 = hidden_states_75.pow(2)
    variance_25 = pow_26.mean(-1, keepdim = True);  pow_26 = None
    add_49 = variance_25 + 1e-06;  variance_25 = None
    rsqrt_25 = torch.rsqrt(add_49);  add_49 = None
    hidden_states_76 = hidden_states_75 * rsqrt_25;  hidden_states_75 = rsqrt_25 = None
    l__self___model_model_layers_6_self_attn_q_norm_weight = self.L__self___model_model_layers_6_self_attn_q_norm_weight
    to_54 = hidden_states_76.to(torch.float16);  hidden_states_76 = None
    mul_83 = l__self___model_model_layers_6_self_attn_q_norm_weight * to_54;  l__self___model_model_layers_6_self_attn_q_norm_weight = to_54 = None
    query_states_6 = mul_83.transpose(1, 2);  mul_83 = None
    l__self___model_model_layers_6_self_attn_k_proj_weight = self.L__self___model_model_layers_6_self_attn_k_proj_weight
    linear_43 = torch._C._nn.linear(hidden_states_74, l__self___model_model_layers_6_self_attn_k_proj_weight, None);  l__self___model_model_layers_6_self_attn_k_proj_weight = None
    view_19 = linear_43.view((1, 2048, -1, 128));  linear_43 = None
    hidden_states_77 = view_19.to(torch.float32);  view_19 = None
    pow_27 = hidden_states_77.pow(2)
    variance_26 = pow_27.mean(-1, keepdim = True);  pow_27 = None
    add_50 = variance_26 + 1e-06;  variance_26 = None
    rsqrt_26 = torch.rsqrt(add_50);  add_50 = None
    hidden_states_78 = hidden_states_77 * rsqrt_26;  hidden_states_77 = rsqrt_26 = None
    l__self___model_model_layers_6_self_attn_k_norm_weight = self.L__self___model_model_layers_6_self_attn_k_norm_weight
    to_56 = hidden_states_78.to(torch.float16);  hidden_states_78 = None
    mul_85 = l__self___model_model_layers_6_self_attn_k_norm_weight * to_56;  l__self___model_model_layers_6_self_attn_k_norm_weight = to_56 = None
    key_states_6 = mul_85.transpose(1, 2);  mul_85 = None
    l__self___model_model_layers_6_self_attn_v_proj_weight = self.L__self___model_model_layers_6_self_attn_v_proj_weight
    linear_44 = torch._C._nn.linear(hidden_states_74, l__self___model_model_layers_6_self_attn_v_proj_weight, None);  hidden_states_74 = l__self___model_model_layers_6_self_attn_v_proj_weight = None
    view_20 = linear_44.view((1, 2048, -1, 128));  linear_44 = None
    value_states_6 = view_20.transpose(1, 2);  view_20 = None
    cos_9 = cos_2.unsqueeze(1)
    sin_9 = sin_2.unsqueeze(1)
    mul_86 = query_states_6 * cos_9
    x1_12 = query_states_6[(Ellipsis, slice(None, 64, None))]
    x2_12 = query_states_6[(Ellipsis, slice(64, None, None))];  query_states_6 = None
    neg_12 = -x2_12;  x2_12 = None
    cat_13 = torch.cat((neg_12, x1_12), dim = -1);  neg_12 = x1_12 = None
    mul_87 = cat_13 * sin_9;  cat_13 = None
    q_embed_6 = mul_86 + mul_87;  mul_86 = mul_87 = None
    mul_88 = key_states_6 * cos_9;  cos_9 = None
    x1_13 = key_states_6[(Ellipsis, slice(None, 64, None))]
    x2_13 = key_states_6[(Ellipsis, slice(64, None, None))];  key_states_6 = None
    neg_13 = -x2_13;  x2_13 = None
    cat_14 = torch.cat((neg_13, x1_13), dim = -1);  neg_13 = x1_13 = None
    mul_89 = cat_14 * sin_9;  cat_14 = sin_9 = None
    k_embed_6 = mul_88 + mul_89;  mul_88 = mul_89 = None
    q_6 = q_embed_6.contiguous();  q_embed_6 = None
    k_6 = k_embed_6.contiguous();  k_embed_6 = None
    v_6 = value_states_6.contiguous();  value_states_6 = None
    npu_fused_infer_attention_score_default_6 = torch.ops.npu.npu_fused_infer_attention_score.default(q_6, k_6, v_6, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_6 = npu_fused_infer_attention_score_default_6[0];  npu_fused_infer_attention_score_default_6 = None
    attn_output_18 = output_6.transpose(1, 2)
    reshape_6 = attn_output_18.reshape(1, 2048, -1);  attn_output_18 = None
    attn_output_19 = reshape_6.contiguous();  reshape_6 = None
    l__self___model_model_layers_6_self_attn_o_proj_weight = self.L__self___model_model_layers_6_self_attn_o_proj_weight
    attn_output_20 = torch._C._nn.linear(attn_output_19, l__self___model_model_layers_6_self_attn_o_proj_weight, None);  attn_output_19 = l__self___model_model_layers_6_self_attn_o_proj_weight = None
    hidden_states_79 = hidden_states_71 + attn_output_20;  hidden_states_71 = attn_output_20 = None
    hidden_states_80 = hidden_states_79.to(torch.float32)
    pow_28 = hidden_states_80.pow(2)
    variance_27 = pow_28.mean(-1, keepdim = True);  pow_28 = None
    add_54 = variance_27 + 1e-06;  variance_27 = None
    rsqrt_27 = torch.rsqrt(add_54);  add_54 = None
    hidden_states_81 = hidden_states_80 * rsqrt_27;  hidden_states_80 = rsqrt_27 = None
    l__self___model_model_layers_6_post_attention_layernorm_weight = self.L__self___model_model_layers_6_post_attention_layernorm_weight
    to_58 = hidden_states_81.to(torch.float16);  hidden_states_81 = None
    hidden_states_82 = l__self___model_model_layers_6_post_attention_layernorm_weight * to_58;  l__self___model_model_layers_6_post_attention_layernorm_weight = to_58 = None
    l__self___model_model_layers_6_mlp_gate_proj_weight = self.L__self___model_model_layers_6_mlp_gate_proj_weight
    linear_46 = torch._C._nn.linear(hidden_states_82, l__self___model_model_layers_6_mlp_gate_proj_weight, None);  l__self___model_model_layers_6_mlp_gate_proj_weight = None
    silu_6 = torch.nn.functional.silu(linear_46);  linear_46 = None
    l__self___model_model_layers_6_mlp_up_proj_weight = self.L__self___model_model_layers_6_mlp_up_proj_weight
    linear_47 = torch._C._nn.linear(hidden_states_82, l__self___model_model_layers_6_mlp_up_proj_weight, None);  hidden_states_82 = l__self___model_model_layers_6_mlp_up_proj_weight = None
    mul_92 = silu_6 * linear_47;  silu_6 = linear_47 = None
    l__self___model_model_layers_6_mlp_down_proj_weight = self.L__self___model_model_layers_6_mlp_down_proj_weight
    down_proj_6 = torch._C._nn.linear(mul_92, l__self___model_model_layers_6_mlp_down_proj_weight, None);  mul_92 = l__self___model_model_layers_6_mlp_down_proj_weight = None
    hidden_states_83 = hidden_states_79 + down_proj_6;  hidden_states_79 = down_proj_6 = None
    hidden_states_84 = hidden_states_83.to(torch.float32)
    pow_29 = hidden_states_84.pow(2)
    variance_28 = pow_29.mean(-1, keepdim = True);  pow_29 = None
    add_56 = variance_28 + 1e-06;  variance_28 = None
    rsqrt_28 = torch.rsqrt(add_56);  add_56 = None
    hidden_states_85 = hidden_states_84 * rsqrt_28;  hidden_states_84 = rsqrt_28 = None
    l__self___model_model_layers_7_input_layernorm_weight = self.L__self___model_model_layers_7_input_layernorm_weight
    to_60 = hidden_states_85.to(torch.float16);  hidden_states_85 = None
    hidden_states_86 = l__self___model_model_layers_7_input_layernorm_weight * to_60;  l__self___model_model_layers_7_input_layernorm_weight = to_60 = None
    l__self___model_model_layers_7_self_attn_q_proj_weight = self.L__self___model_model_layers_7_self_attn_q_proj_weight
    linear_49 = torch._C._nn.linear(hidden_states_86, l__self___model_model_layers_7_self_attn_q_proj_weight, None);  l__self___model_model_layers_7_self_attn_q_proj_weight = None
    view_21 = linear_49.view((1, 2048, -1, 128));  linear_49 = None
    hidden_states_87 = view_21.to(torch.float32);  view_21 = None
    pow_30 = hidden_states_87.pow(2)
    variance_29 = pow_30.mean(-1, keepdim = True);  pow_30 = None
    add_57 = variance_29 + 1e-06;  variance_29 = None
    rsqrt_29 = torch.rsqrt(add_57);  add_57 = None
    hidden_states_88 = hidden_states_87 * rsqrt_29;  hidden_states_87 = rsqrt_29 = None
    l__self___model_model_layers_7_self_attn_q_norm_weight = self.L__self___model_model_layers_7_self_attn_q_norm_weight
    to_62 = hidden_states_88.to(torch.float16);  hidden_states_88 = None
    mul_96 = l__self___model_model_layers_7_self_attn_q_norm_weight * to_62;  l__self___model_model_layers_7_self_attn_q_norm_weight = to_62 = None
    query_states_7 = mul_96.transpose(1, 2);  mul_96 = None
    l__self___model_model_layers_7_self_attn_k_proj_weight = self.L__self___model_model_layers_7_self_attn_k_proj_weight
    linear_50 = torch._C._nn.linear(hidden_states_86, l__self___model_model_layers_7_self_attn_k_proj_weight, None);  l__self___model_model_layers_7_self_attn_k_proj_weight = None
    view_22 = linear_50.view((1, 2048, -1, 128));  linear_50 = None
    hidden_states_89 = view_22.to(torch.float32);  view_22 = None
    pow_31 = hidden_states_89.pow(2)
    variance_30 = pow_31.mean(-1, keepdim = True);  pow_31 = None
    add_58 = variance_30 + 1e-06;  variance_30 = None
    rsqrt_30 = torch.rsqrt(add_58);  add_58 = None
    hidden_states_90 = hidden_states_89 * rsqrt_30;  hidden_states_89 = rsqrt_30 = None
    l__self___model_model_layers_7_self_attn_k_norm_weight = self.L__self___model_model_layers_7_self_attn_k_norm_weight
    to_64 = hidden_states_90.to(torch.float16);  hidden_states_90 = None
    mul_98 = l__self___model_model_layers_7_self_attn_k_norm_weight * to_64;  l__self___model_model_layers_7_self_attn_k_norm_weight = to_64 = None
    key_states_7 = mul_98.transpose(1, 2);  mul_98 = None
    l__self___model_model_layers_7_self_attn_v_proj_weight = self.L__self___model_model_layers_7_self_attn_v_proj_weight
    linear_51 = torch._C._nn.linear(hidden_states_86, l__self___model_model_layers_7_self_attn_v_proj_weight, None);  hidden_states_86 = l__self___model_model_layers_7_self_attn_v_proj_weight = None
    view_23 = linear_51.view((1, 2048, -1, 128));  linear_51 = None
    value_states_7 = view_23.transpose(1, 2);  view_23 = None
    cos_10 = cos_2.unsqueeze(1)
    sin_10 = sin_2.unsqueeze(1)
    mul_99 = query_states_7 * cos_10
    x1_14 = query_states_7[(Ellipsis, slice(None, 64, None))]
    x2_14 = query_states_7[(Ellipsis, slice(64, None, None))];  query_states_7 = None
    neg_14 = -x2_14;  x2_14 = None
    cat_15 = torch.cat((neg_14, x1_14), dim = -1);  neg_14 = x1_14 = None
    mul_100 = cat_15 * sin_10;  cat_15 = None
    q_embed_7 = mul_99 + mul_100;  mul_99 = mul_100 = None
    mul_101 = key_states_7 * cos_10;  cos_10 = None
    x1_15 = key_states_7[(Ellipsis, slice(None, 64, None))]
    x2_15 = key_states_7[(Ellipsis, slice(64, None, None))];  key_states_7 = None
    neg_15 = -x2_15;  x2_15 = None
    cat_16 = torch.cat((neg_15, x1_15), dim = -1);  neg_15 = x1_15 = None
    mul_102 = cat_16 * sin_10;  cat_16 = sin_10 = None
    k_embed_7 = mul_101 + mul_102;  mul_101 = mul_102 = None
    q_7 = q_embed_7.contiguous();  q_embed_7 = None
    k_7 = k_embed_7.contiguous();  k_embed_7 = None
    v_7 = value_states_7.contiguous();  value_states_7 = None
    npu_fused_infer_attention_score_default_7 = torch.ops.npu.npu_fused_infer_attention_score.default(q_7, k_7, v_7, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_7 = npu_fused_infer_attention_score_default_7[0];  npu_fused_infer_attention_score_default_7 = None
    attn_output_21 = output_7.transpose(1, 2)
    reshape_7 = attn_output_21.reshape(1, 2048, -1);  attn_output_21 = None
    attn_output_22 = reshape_7.contiguous();  reshape_7 = None
    l__self___model_model_layers_7_self_attn_o_proj_weight = self.L__self___model_model_layers_7_self_attn_o_proj_weight
    attn_output_23 = torch._C._nn.linear(attn_output_22, l__self___model_model_layers_7_self_attn_o_proj_weight, None);  attn_output_22 = l__self___model_model_layers_7_self_attn_o_proj_weight = None
    hidden_states_91 = hidden_states_83 + attn_output_23;  hidden_states_83 = attn_output_23 = None
    hidden_states_92 = hidden_states_91.to(torch.float32)
    pow_32 = hidden_states_92.pow(2)
    variance_31 = pow_32.mean(-1, keepdim = True);  pow_32 = None
    add_62 = variance_31 + 1e-06;  variance_31 = None
    rsqrt_31 = torch.rsqrt(add_62);  add_62 = None
    hidden_states_93 = hidden_states_92 * rsqrt_31;  hidden_states_92 = rsqrt_31 = None
    l__self___model_model_layers_7_post_attention_layernorm_weight = self.L__self___model_model_layers_7_post_attention_layernorm_weight
    to_66 = hidden_states_93.to(torch.float16);  hidden_states_93 = None
    hidden_states_94 = l__self___model_model_layers_7_post_attention_layernorm_weight * to_66;  l__self___model_model_layers_7_post_attention_layernorm_weight = to_66 = None
    l__self___model_model_layers_7_mlp_gate_proj_weight = self.L__self___model_model_layers_7_mlp_gate_proj_weight
    linear_53 = torch._C._nn.linear(hidden_states_94, l__self___model_model_layers_7_mlp_gate_proj_weight, None);  l__self___model_model_layers_7_mlp_gate_proj_weight = None
    silu_7 = torch.nn.functional.silu(linear_53);  linear_53 = None
    l__self___model_model_layers_7_mlp_up_proj_weight = self.L__self___model_model_layers_7_mlp_up_proj_weight
    linear_54 = torch._C._nn.linear(hidden_states_94, l__self___model_model_layers_7_mlp_up_proj_weight, None);  hidden_states_94 = l__self___model_model_layers_7_mlp_up_proj_weight = None
    mul_105 = silu_7 * linear_54;  silu_7 = linear_54 = None
    l__self___model_model_layers_7_mlp_down_proj_weight = self.L__self___model_model_layers_7_mlp_down_proj_weight
    down_proj_7 = torch._C._nn.linear(mul_105, l__self___model_model_layers_7_mlp_down_proj_weight, None);  mul_105 = l__self___model_model_layers_7_mlp_down_proj_weight = None
    hidden_states_95 = hidden_states_91 + down_proj_7;  hidden_states_91 = down_proj_7 = None
    hidden_states_96 = hidden_states_95.to(torch.float32)
    pow_33 = hidden_states_96.pow(2)
    variance_32 = pow_33.mean(-1, keepdim = True);  pow_33 = None
    add_64 = variance_32 + 1e-06;  variance_32 = None
    rsqrt_32 = torch.rsqrt(add_64);  add_64 = None
    hidden_states_97 = hidden_states_96 * rsqrt_32;  hidden_states_96 = rsqrt_32 = None
    l__self___model_model_layers_8_input_layernorm_weight = self.L__self___model_model_layers_8_input_layernorm_weight
    to_68 = hidden_states_97.to(torch.float16);  hidden_states_97 = None
    hidden_states_98 = l__self___model_model_layers_8_input_layernorm_weight * to_68;  l__self___model_model_layers_8_input_layernorm_weight = to_68 = None
    l__self___model_model_layers_8_self_attn_q_proj_weight = self.L__self___model_model_layers_8_self_attn_q_proj_weight
    linear_56 = torch._C._nn.linear(hidden_states_98, l__self___model_model_layers_8_self_attn_q_proj_weight, None);  l__self___model_model_layers_8_self_attn_q_proj_weight = None
    view_24 = linear_56.view((1, 2048, -1, 128));  linear_56 = None
    hidden_states_99 = view_24.to(torch.float32);  view_24 = None
    pow_34 = hidden_states_99.pow(2)
    variance_33 = pow_34.mean(-1, keepdim = True);  pow_34 = None
    add_65 = variance_33 + 1e-06;  variance_33 = None
    rsqrt_33 = torch.rsqrt(add_65);  add_65 = None
    hidden_states_100 = hidden_states_99 * rsqrt_33;  hidden_states_99 = rsqrt_33 = None
    l__self___model_model_layers_8_self_attn_q_norm_weight = self.L__self___model_model_layers_8_self_attn_q_norm_weight
    to_70 = hidden_states_100.to(torch.float16);  hidden_states_100 = None
    mul_109 = l__self___model_model_layers_8_self_attn_q_norm_weight * to_70;  l__self___model_model_layers_8_self_attn_q_norm_weight = to_70 = None
    query_states_8 = mul_109.transpose(1, 2);  mul_109 = None
    l__self___model_model_layers_8_self_attn_k_proj_weight = self.L__self___model_model_layers_8_self_attn_k_proj_weight
    linear_57 = torch._C._nn.linear(hidden_states_98, l__self___model_model_layers_8_self_attn_k_proj_weight, None);  l__self___model_model_layers_8_self_attn_k_proj_weight = None
    view_25 = linear_57.view((1, 2048, -1, 128));  linear_57 = None
    hidden_states_101 = view_25.to(torch.float32);  view_25 = None
    pow_35 = hidden_states_101.pow(2)
    variance_34 = pow_35.mean(-1, keepdim = True);  pow_35 = None
    add_66 = variance_34 + 1e-06;  variance_34 = None
    rsqrt_34 = torch.rsqrt(add_66);  add_66 = None
    hidden_states_102 = hidden_states_101 * rsqrt_34;  hidden_states_101 = rsqrt_34 = None
    l__self___model_model_layers_8_self_attn_k_norm_weight = self.L__self___model_model_layers_8_self_attn_k_norm_weight
    to_72 = hidden_states_102.to(torch.float16);  hidden_states_102 = None
    mul_111 = l__self___model_model_layers_8_self_attn_k_norm_weight * to_72;  l__self___model_model_layers_8_self_attn_k_norm_weight = to_72 = None
    key_states_8 = mul_111.transpose(1, 2);  mul_111 = None
    l__self___model_model_layers_8_self_attn_v_proj_weight = self.L__self___model_model_layers_8_self_attn_v_proj_weight
    linear_58 = torch._C._nn.linear(hidden_states_98, l__self___model_model_layers_8_self_attn_v_proj_weight, None);  hidden_states_98 = l__self___model_model_layers_8_self_attn_v_proj_weight = None
    view_26 = linear_58.view((1, 2048, -1, 128));  linear_58 = None
    value_states_8 = view_26.transpose(1, 2);  view_26 = None
    cos_11 = cos_2.unsqueeze(1)
    sin_11 = sin_2.unsqueeze(1)
    mul_112 = query_states_8 * cos_11
    x1_16 = query_states_8[(Ellipsis, slice(None, 64, None))]
    x2_16 = query_states_8[(Ellipsis, slice(64, None, None))];  query_states_8 = None
    neg_16 = -x2_16;  x2_16 = None
    cat_17 = torch.cat((neg_16, x1_16), dim = -1);  neg_16 = x1_16 = None
    mul_113 = cat_17 * sin_11;  cat_17 = None
    q_embed_8 = mul_112 + mul_113;  mul_112 = mul_113 = None
    mul_114 = key_states_8 * cos_11;  cos_11 = None
    x1_17 = key_states_8[(Ellipsis, slice(None, 64, None))]
    x2_17 = key_states_8[(Ellipsis, slice(64, None, None))];  key_states_8 = None
    neg_17 = -x2_17;  x2_17 = None
    cat_18 = torch.cat((neg_17, x1_17), dim = -1);  neg_17 = x1_17 = None
    mul_115 = cat_18 * sin_11;  cat_18 = sin_11 = None
    k_embed_8 = mul_114 + mul_115;  mul_114 = mul_115 = None
    q_8 = q_embed_8.contiguous();  q_embed_8 = None
    k_8 = k_embed_8.contiguous();  k_embed_8 = None
    v_8 = value_states_8.contiguous();  value_states_8 = None
    npu_fused_infer_attention_score_default_8 = torch.ops.npu.npu_fused_infer_attention_score.default(q_8, k_8, v_8, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_8 = npu_fused_infer_attention_score_default_8[0];  npu_fused_infer_attention_score_default_8 = None
    attn_output_24 = output_8.transpose(1, 2)
    reshape_8 = attn_output_24.reshape(1, 2048, -1);  attn_output_24 = None
    attn_output_25 = reshape_8.contiguous();  reshape_8 = None
    l__self___model_model_layers_8_self_attn_o_proj_weight = self.L__self___model_model_layers_8_self_attn_o_proj_weight
    attn_output_26 = torch._C._nn.linear(attn_output_25, l__self___model_model_layers_8_self_attn_o_proj_weight, None);  attn_output_25 = l__self___model_model_layers_8_self_attn_o_proj_weight = None
    hidden_states_103 = hidden_states_95 + attn_output_26;  hidden_states_95 = attn_output_26 = None
    hidden_states_104 = hidden_states_103.to(torch.float32)
    pow_36 = hidden_states_104.pow(2)
    variance_35 = pow_36.mean(-1, keepdim = True);  pow_36 = None
    add_70 = variance_35 + 1e-06;  variance_35 = None
    rsqrt_35 = torch.rsqrt(add_70);  add_70 = None
    hidden_states_105 = hidden_states_104 * rsqrt_35;  hidden_states_104 = rsqrt_35 = None
    l__self___model_model_layers_8_post_attention_layernorm_weight = self.L__self___model_model_layers_8_post_attention_layernorm_weight
    to_74 = hidden_states_105.to(torch.float16);  hidden_states_105 = None
    hidden_states_106 = l__self___model_model_layers_8_post_attention_layernorm_weight * to_74;  l__self___model_model_layers_8_post_attention_layernorm_weight = to_74 = None
    l__self___model_model_layers_8_mlp_gate_proj_weight = self.L__self___model_model_layers_8_mlp_gate_proj_weight
    linear_60 = torch._C._nn.linear(hidden_states_106, l__self___model_model_layers_8_mlp_gate_proj_weight, None);  l__self___model_model_layers_8_mlp_gate_proj_weight = None
    silu_8 = torch.nn.functional.silu(linear_60);  linear_60 = None
    l__self___model_model_layers_8_mlp_up_proj_weight = self.L__self___model_model_layers_8_mlp_up_proj_weight
    linear_61 = torch._C._nn.linear(hidden_states_106, l__self___model_model_layers_8_mlp_up_proj_weight, None);  hidden_states_106 = l__self___model_model_layers_8_mlp_up_proj_weight = None
    mul_118 = silu_8 * linear_61;  silu_8 = linear_61 = None
    l__self___model_model_layers_8_mlp_down_proj_weight = self.L__self___model_model_layers_8_mlp_down_proj_weight
    down_proj_8 = torch._C._nn.linear(mul_118, l__self___model_model_layers_8_mlp_down_proj_weight, None);  mul_118 = l__self___model_model_layers_8_mlp_down_proj_weight = None
    hidden_states_107 = hidden_states_103 + down_proj_8;  hidden_states_103 = down_proj_8 = None
    hidden_states_108 = hidden_states_107.to(torch.float32)
    pow_37 = hidden_states_108.pow(2)
    variance_36 = pow_37.mean(-1, keepdim = True);  pow_37 = None
    add_72 = variance_36 + 1e-06;  variance_36 = None
    rsqrt_36 = torch.rsqrt(add_72);  add_72 = None
    hidden_states_109 = hidden_states_108 * rsqrt_36;  hidden_states_108 = rsqrt_36 = None
    l__self___model_model_layers_9_input_layernorm_weight = self.L__self___model_model_layers_9_input_layernorm_weight
    to_76 = hidden_states_109.to(torch.float16);  hidden_states_109 = None
    hidden_states_110 = l__self___model_model_layers_9_input_layernorm_weight * to_76;  l__self___model_model_layers_9_input_layernorm_weight = to_76 = None
    l__self___model_model_layers_9_self_attn_q_proj_weight = self.L__self___model_model_layers_9_self_attn_q_proj_weight
    linear_63 = torch._C._nn.linear(hidden_states_110, l__self___model_model_layers_9_self_attn_q_proj_weight, None);  l__self___model_model_layers_9_self_attn_q_proj_weight = None
    view_27 = linear_63.view((1, 2048, -1, 128));  linear_63 = None
    hidden_states_111 = view_27.to(torch.float32);  view_27 = None
    pow_38 = hidden_states_111.pow(2)
    variance_37 = pow_38.mean(-1, keepdim = True);  pow_38 = None
    add_73 = variance_37 + 1e-06;  variance_37 = None
    rsqrt_37 = torch.rsqrt(add_73);  add_73 = None
    hidden_states_112 = hidden_states_111 * rsqrt_37;  hidden_states_111 = rsqrt_37 = None
    l__self___model_model_layers_9_self_attn_q_norm_weight = self.L__self___model_model_layers_9_self_attn_q_norm_weight
    to_78 = hidden_states_112.to(torch.float16);  hidden_states_112 = None
    mul_122 = l__self___model_model_layers_9_self_attn_q_norm_weight * to_78;  l__self___model_model_layers_9_self_attn_q_norm_weight = to_78 = None
    query_states_9 = mul_122.transpose(1, 2);  mul_122 = None
    l__self___model_model_layers_9_self_attn_k_proj_weight = self.L__self___model_model_layers_9_self_attn_k_proj_weight
    linear_64 = torch._C._nn.linear(hidden_states_110, l__self___model_model_layers_9_self_attn_k_proj_weight, None);  l__self___model_model_layers_9_self_attn_k_proj_weight = None
    view_28 = linear_64.view((1, 2048, -1, 128));  linear_64 = None
    hidden_states_113 = view_28.to(torch.float32);  view_28 = None
    pow_39 = hidden_states_113.pow(2)
    variance_38 = pow_39.mean(-1, keepdim = True);  pow_39 = None
    add_74 = variance_38 + 1e-06;  variance_38 = None
    rsqrt_38 = torch.rsqrt(add_74);  add_74 = None
    hidden_states_114 = hidden_states_113 * rsqrt_38;  hidden_states_113 = rsqrt_38 = None
    l__self___model_model_layers_9_self_attn_k_norm_weight = self.L__self___model_model_layers_9_self_attn_k_norm_weight
    to_80 = hidden_states_114.to(torch.float16);  hidden_states_114 = None
    mul_124 = l__self___model_model_layers_9_self_attn_k_norm_weight * to_80;  l__self___model_model_layers_9_self_attn_k_norm_weight = to_80 = None
    key_states_9 = mul_124.transpose(1, 2);  mul_124 = None
    l__self___model_model_layers_9_self_attn_v_proj_weight = self.L__self___model_model_layers_9_self_attn_v_proj_weight
    linear_65 = torch._C._nn.linear(hidden_states_110, l__self___model_model_layers_9_self_attn_v_proj_weight, None);  hidden_states_110 = l__self___model_model_layers_9_self_attn_v_proj_weight = None
    view_29 = linear_65.view((1, 2048, -1, 128));  linear_65 = None
    value_states_9 = view_29.transpose(1, 2);  view_29 = None
    cos_12 = cos_2.unsqueeze(1)
    sin_12 = sin_2.unsqueeze(1)
    mul_125 = query_states_9 * cos_12
    x1_18 = query_states_9[(Ellipsis, slice(None, 64, None))]
    x2_18 = query_states_9[(Ellipsis, slice(64, None, None))];  query_states_9 = None
    neg_18 = -x2_18;  x2_18 = None
    cat_19 = torch.cat((neg_18, x1_18), dim = -1);  neg_18 = x1_18 = None
    mul_126 = cat_19 * sin_12;  cat_19 = None
    q_embed_9 = mul_125 + mul_126;  mul_125 = mul_126 = None
    mul_127 = key_states_9 * cos_12;  cos_12 = None
    x1_19 = key_states_9[(Ellipsis, slice(None, 64, None))]
    x2_19 = key_states_9[(Ellipsis, slice(64, None, None))];  key_states_9 = None
    neg_19 = -x2_19;  x2_19 = None
    cat_20 = torch.cat((neg_19, x1_19), dim = -1);  neg_19 = x1_19 = None
    mul_128 = cat_20 * sin_12;  cat_20 = sin_12 = None
    k_embed_9 = mul_127 + mul_128;  mul_127 = mul_128 = None
    q_9 = q_embed_9.contiguous();  q_embed_9 = None
    k_9 = k_embed_9.contiguous();  k_embed_9 = None
    v_9 = value_states_9.contiguous();  value_states_9 = None
    npu_fused_infer_attention_score_default_9 = torch.ops.npu.npu_fused_infer_attention_score.default(q_9, k_9, v_9, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_9 = npu_fused_infer_attention_score_default_9[0];  npu_fused_infer_attention_score_default_9 = None
    attn_output_27 = output_9.transpose(1, 2)
    reshape_9 = attn_output_27.reshape(1, 2048, -1);  attn_output_27 = None
    attn_output_28 = reshape_9.contiguous();  reshape_9 = None
    l__self___model_model_layers_9_self_attn_o_proj_weight = self.L__self___model_model_layers_9_self_attn_o_proj_weight
    attn_output_29 = torch._C._nn.linear(attn_output_28, l__self___model_model_layers_9_self_attn_o_proj_weight, None);  attn_output_28 = l__self___model_model_layers_9_self_attn_o_proj_weight = None
    hidden_states_115 = hidden_states_107 + attn_output_29;  hidden_states_107 = attn_output_29 = None
    hidden_states_116 = hidden_states_115.to(torch.float32)
    pow_40 = hidden_states_116.pow(2)
    variance_39 = pow_40.mean(-1, keepdim = True);  pow_40 = None
    add_78 = variance_39 + 1e-06;  variance_39 = None
    rsqrt_39 = torch.rsqrt(add_78);  add_78 = None
    hidden_states_117 = hidden_states_116 * rsqrt_39;  hidden_states_116 = rsqrt_39 = None
    l__self___model_model_layers_9_post_attention_layernorm_weight = self.L__self___model_model_layers_9_post_attention_layernorm_weight
    to_82 = hidden_states_117.to(torch.float16);  hidden_states_117 = None
    hidden_states_118 = l__self___model_model_layers_9_post_attention_layernorm_weight * to_82;  l__self___model_model_layers_9_post_attention_layernorm_weight = to_82 = None
    l__self___model_model_layers_9_mlp_gate_proj_weight = self.L__self___model_model_layers_9_mlp_gate_proj_weight
    linear_67 = torch._C._nn.linear(hidden_states_118, l__self___model_model_layers_9_mlp_gate_proj_weight, None);  l__self___model_model_layers_9_mlp_gate_proj_weight = None
    silu_9 = torch.nn.functional.silu(linear_67);  linear_67 = None
    l__self___model_model_layers_9_mlp_up_proj_weight = self.L__self___model_model_layers_9_mlp_up_proj_weight
    linear_68 = torch._C._nn.linear(hidden_states_118, l__self___model_model_layers_9_mlp_up_proj_weight, None);  hidden_states_118 = l__self___model_model_layers_9_mlp_up_proj_weight = None
    mul_131 = silu_9 * linear_68;  silu_9 = linear_68 = None
    l__self___model_model_layers_9_mlp_down_proj_weight = self.L__self___model_model_layers_9_mlp_down_proj_weight
    down_proj_9 = torch._C._nn.linear(mul_131, l__self___model_model_layers_9_mlp_down_proj_weight, None);  mul_131 = l__self___model_model_layers_9_mlp_down_proj_weight = None
    hidden_states_119 = hidden_states_115 + down_proj_9;  hidden_states_115 = down_proj_9 = None
    hidden_states_120 = hidden_states_119.to(torch.float32)
    pow_41 = hidden_states_120.pow(2)
    variance_40 = pow_41.mean(-1, keepdim = True);  pow_41 = None
    add_80 = variance_40 + 1e-06;  variance_40 = None
    rsqrt_40 = torch.rsqrt(add_80);  add_80 = None
    hidden_states_121 = hidden_states_120 * rsqrt_40;  hidden_states_120 = rsqrt_40 = None
    l__self___model_model_layers_10_input_layernorm_weight = self.L__self___model_model_layers_10_input_layernorm_weight
    to_84 = hidden_states_121.to(torch.float16);  hidden_states_121 = None
    hidden_states_122 = l__self___model_model_layers_10_input_layernorm_weight * to_84;  l__self___model_model_layers_10_input_layernorm_weight = to_84 = None
    l__self___model_model_layers_10_self_attn_q_proj_weight = self.L__self___model_model_layers_10_self_attn_q_proj_weight
    linear_70 = torch._C._nn.linear(hidden_states_122, l__self___model_model_layers_10_self_attn_q_proj_weight, None);  l__self___model_model_layers_10_self_attn_q_proj_weight = None
    view_30 = linear_70.view((1, 2048, -1, 128));  linear_70 = None
    hidden_states_123 = view_30.to(torch.float32);  view_30 = None
    pow_42 = hidden_states_123.pow(2)
    variance_41 = pow_42.mean(-1, keepdim = True);  pow_42 = None
    add_81 = variance_41 + 1e-06;  variance_41 = None
    rsqrt_41 = torch.rsqrt(add_81);  add_81 = None
    hidden_states_124 = hidden_states_123 * rsqrt_41;  hidden_states_123 = rsqrt_41 = None
    l__self___model_model_layers_10_self_attn_q_norm_weight = self.L__self___model_model_layers_10_self_attn_q_norm_weight
    to_86 = hidden_states_124.to(torch.float16);  hidden_states_124 = None
    mul_135 = l__self___model_model_layers_10_self_attn_q_norm_weight * to_86;  l__self___model_model_layers_10_self_attn_q_norm_weight = to_86 = None
    query_states_10 = mul_135.transpose(1, 2);  mul_135 = None
    l__self___model_model_layers_10_self_attn_k_proj_weight = self.L__self___model_model_layers_10_self_attn_k_proj_weight
    linear_71 = torch._C._nn.linear(hidden_states_122, l__self___model_model_layers_10_self_attn_k_proj_weight, None);  l__self___model_model_layers_10_self_attn_k_proj_weight = None
    view_31 = linear_71.view((1, 2048, -1, 128));  linear_71 = None
    hidden_states_125 = view_31.to(torch.float32);  view_31 = None
    pow_43 = hidden_states_125.pow(2)
    variance_42 = pow_43.mean(-1, keepdim = True);  pow_43 = None
    add_82 = variance_42 + 1e-06;  variance_42 = None
    rsqrt_42 = torch.rsqrt(add_82);  add_82 = None
    hidden_states_126 = hidden_states_125 * rsqrt_42;  hidden_states_125 = rsqrt_42 = None
    l__self___model_model_layers_10_self_attn_k_norm_weight = self.L__self___model_model_layers_10_self_attn_k_norm_weight
    to_88 = hidden_states_126.to(torch.float16);  hidden_states_126 = None
    mul_137 = l__self___model_model_layers_10_self_attn_k_norm_weight * to_88;  l__self___model_model_layers_10_self_attn_k_norm_weight = to_88 = None
    key_states_10 = mul_137.transpose(1, 2);  mul_137 = None
    l__self___model_model_layers_10_self_attn_v_proj_weight = self.L__self___model_model_layers_10_self_attn_v_proj_weight
    linear_72 = torch._C._nn.linear(hidden_states_122, l__self___model_model_layers_10_self_attn_v_proj_weight, None);  hidden_states_122 = l__self___model_model_layers_10_self_attn_v_proj_weight = None
    view_32 = linear_72.view((1, 2048, -1, 128));  linear_72 = None
    value_states_10 = view_32.transpose(1, 2);  view_32 = None
    cos_13 = cos_2.unsqueeze(1)
    sin_13 = sin_2.unsqueeze(1)
    mul_138 = query_states_10 * cos_13
    x1_20 = query_states_10[(Ellipsis, slice(None, 64, None))]
    x2_20 = query_states_10[(Ellipsis, slice(64, None, None))];  query_states_10 = None
    neg_20 = -x2_20;  x2_20 = None
    cat_21 = torch.cat((neg_20, x1_20), dim = -1);  neg_20 = x1_20 = None
    mul_139 = cat_21 * sin_13;  cat_21 = None
    q_embed_10 = mul_138 + mul_139;  mul_138 = mul_139 = None
    mul_140 = key_states_10 * cos_13;  cos_13 = None
    x1_21 = key_states_10[(Ellipsis, slice(None, 64, None))]
    x2_21 = key_states_10[(Ellipsis, slice(64, None, None))];  key_states_10 = None
    neg_21 = -x2_21;  x2_21 = None
    cat_22 = torch.cat((neg_21, x1_21), dim = -1);  neg_21 = x1_21 = None
    mul_141 = cat_22 * sin_13;  cat_22 = sin_13 = None
    k_embed_10 = mul_140 + mul_141;  mul_140 = mul_141 = None
    q_10 = q_embed_10.contiguous();  q_embed_10 = None
    k_10 = k_embed_10.contiguous();  k_embed_10 = None
    v_10 = value_states_10.contiguous();  value_states_10 = None
    npu_fused_infer_attention_score_default_10 = torch.ops.npu.npu_fused_infer_attention_score.default(q_10, k_10, v_10, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_10 = npu_fused_infer_attention_score_default_10[0];  npu_fused_infer_attention_score_default_10 = None
    attn_output_30 = output_10.transpose(1, 2)
    reshape_10 = attn_output_30.reshape(1, 2048, -1);  attn_output_30 = None
    attn_output_31 = reshape_10.contiguous();  reshape_10 = None
    l__self___model_model_layers_10_self_attn_o_proj_weight = self.L__self___model_model_layers_10_self_attn_o_proj_weight
    attn_output_32 = torch._C._nn.linear(attn_output_31, l__self___model_model_layers_10_self_attn_o_proj_weight, None);  attn_output_31 = l__self___model_model_layers_10_self_attn_o_proj_weight = None
    hidden_states_127 = hidden_states_119 + attn_output_32;  hidden_states_119 = attn_output_32 = None
    hidden_states_128 = hidden_states_127.to(torch.float32)
    pow_44 = hidden_states_128.pow(2)
    variance_43 = pow_44.mean(-1, keepdim = True);  pow_44 = None
    add_86 = variance_43 + 1e-06;  variance_43 = None
    rsqrt_43 = torch.rsqrt(add_86);  add_86 = None
    hidden_states_129 = hidden_states_128 * rsqrt_43;  hidden_states_128 = rsqrt_43 = None
    l__self___model_model_layers_10_post_attention_layernorm_weight = self.L__self___model_model_layers_10_post_attention_layernorm_weight
    to_90 = hidden_states_129.to(torch.float16);  hidden_states_129 = None
    hidden_states_130 = l__self___model_model_layers_10_post_attention_layernorm_weight * to_90;  l__self___model_model_layers_10_post_attention_layernorm_weight = to_90 = None
    l__self___model_model_layers_10_mlp_gate_proj_weight = self.L__self___model_model_layers_10_mlp_gate_proj_weight
    linear_74 = torch._C._nn.linear(hidden_states_130, l__self___model_model_layers_10_mlp_gate_proj_weight, None);  l__self___model_model_layers_10_mlp_gate_proj_weight = None
    silu_10 = torch.nn.functional.silu(linear_74);  linear_74 = None
    l__self___model_model_layers_10_mlp_up_proj_weight = self.L__self___model_model_layers_10_mlp_up_proj_weight
    linear_75 = torch._C._nn.linear(hidden_states_130, l__self___model_model_layers_10_mlp_up_proj_weight, None);  hidden_states_130 = l__self___model_model_layers_10_mlp_up_proj_weight = None
    mul_144 = silu_10 * linear_75;  silu_10 = linear_75 = None
    l__self___model_model_layers_10_mlp_down_proj_weight = self.L__self___model_model_layers_10_mlp_down_proj_weight
    down_proj_10 = torch._C._nn.linear(mul_144, l__self___model_model_layers_10_mlp_down_proj_weight, None);  mul_144 = l__self___model_model_layers_10_mlp_down_proj_weight = None
    hidden_states_131 = hidden_states_127 + down_proj_10;  hidden_states_127 = down_proj_10 = None
    hidden_states_132 = hidden_states_131.to(torch.float32)
    pow_45 = hidden_states_132.pow(2)
    variance_44 = pow_45.mean(-1, keepdim = True);  pow_45 = None
    add_88 = variance_44 + 1e-06;  variance_44 = None
    rsqrt_44 = torch.rsqrt(add_88);  add_88 = None
    hidden_states_133 = hidden_states_132 * rsqrt_44;  hidden_states_132 = rsqrt_44 = None
    l__self___model_model_layers_11_input_layernorm_weight = self.L__self___model_model_layers_11_input_layernorm_weight
    to_92 = hidden_states_133.to(torch.float16);  hidden_states_133 = None
    hidden_states_134 = l__self___model_model_layers_11_input_layernorm_weight * to_92;  l__self___model_model_layers_11_input_layernorm_weight = to_92 = None
    l__self___model_model_layers_11_self_attn_q_proj_weight = self.L__self___model_model_layers_11_self_attn_q_proj_weight
    linear_77 = torch._C._nn.linear(hidden_states_134, l__self___model_model_layers_11_self_attn_q_proj_weight, None);  l__self___model_model_layers_11_self_attn_q_proj_weight = None
    view_33 = linear_77.view((1, 2048, -1, 128));  linear_77 = None
    hidden_states_135 = view_33.to(torch.float32);  view_33 = None
    pow_46 = hidden_states_135.pow(2)
    variance_45 = pow_46.mean(-1, keepdim = True);  pow_46 = None
    add_89 = variance_45 + 1e-06;  variance_45 = None
    rsqrt_45 = torch.rsqrt(add_89);  add_89 = None
    hidden_states_136 = hidden_states_135 * rsqrt_45;  hidden_states_135 = rsqrt_45 = None
    l__self___model_model_layers_11_self_attn_q_norm_weight = self.L__self___model_model_layers_11_self_attn_q_norm_weight
    to_94 = hidden_states_136.to(torch.float16);  hidden_states_136 = None
    mul_148 = l__self___model_model_layers_11_self_attn_q_norm_weight * to_94;  l__self___model_model_layers_11_self_attn_q_norm_weight = to_94 = None
    query_states_11 = mul_148.transpose(1, 2);  mul_148 = None
    l__self___model_model_layers_11_self_attn_k_proj_weight = self.L__self___model_model_layers_11_self_attn_k_proj_weight
    linear_78 = torch._C._nn.linear(hidden_states_134, l__self___model_model_layers_11_self_attn_k_proj_weight, None);  l__self___model_model_layers_11_self_attn_k_proj_weight = None
    view_34 = linear_78.view((1, 2048, -1, 128));  linear_78 = None
    hidden_states_137 = view_34.to(torch.float32);  view_34 = None
    pow_47 = hidden_states_137.pow(2)
    variance_46 = pow_47.mean(-1, keepdim = True);  pow_47 = None
    add_90 = variance_46 + 1e-06;  variance_46 = None
    rsqrt_46 = torch.rsqrt(add_90);  add_90 = None
    hidden_states_138 = hidden_states_137 * rsqrt_46;  hidden_states_137 = rsqrt_46 = None
    l__self___model_model_layers_11_self_attn_k_norm_weight = self.L__self___model_model_layers_11_self_attn_k_norm_weight
    to_96 = hidden_states_138.to(torch.float16);  hidden_states_138 = None
    mul_150 = l__self___model_model_layers_11_self_attn_k_norm_weight * to_96;  l__self___model_model_layers_11_self_attn_k_norm_weight = to_96 = None
    key_states_11 = mul_150.transpose(1, 2);  mul_150 = None
    l__self___model_model_layers_11_self_attn_v_proj_weight = self.L__self___model_model_layers_11_self_attn_v_proj_weight
    linear_79 = torch._C._nn.linear(hidden_states_134, l__self___model_model_layers_11_self_attn_v_proj_weight, None);  hidden_states_134 = l__self___model_model_layers_11_self_attn_v_proj_weight = None
    view_35 = linear_79.view((1, 2048, -1, 128));  linear_79 = None
    value_states_11 = view_35.transpose(1, 2);  view_35 = None
    cos_14 = cos_2.unsqueeze(1)
    sin_14 = sin_2.unsqueeze(1)
    mul_151 = query_states_11 * cos_14
    x1_22 = query_states_11[(Ellipsis, slice(None, 64, None))]
    x2_22 = query_states_11[(Ellipsis, slice(64, None, None))];  query_states_11 = None
    neg_22 = -x2_22;  x2_22 = None
    cat_23 = torch.cat((neg_22, x1_22), dim = -1);  neg_22 = x1_22 = None
    mul_152 = cat_23 * sin_14;  cat_23 = None
    q_embed_11 = mul_151 + mul_152;  mul_151 = mul_152 = None
    mul_153 = key_states_11 * cos_14;  cos_14 = None
    x1_23 = key_states_11[(Ellipsis, slice(None, 64, None))]
    x2_23 = key_states_11[(Ellipsis, slice(64, None, None))];  key_states_11 = None
    neg_23 = -x2_23;  x2_23 = None
    cat_24 = torch.cat((neg_23, x1_23), dim = -1);  neg_23 = x1_23 = None
    mul_154 = cat_24 * sin_14;  cat_24 = sin_14 = None
    k_embed_11 = mul_153 + mul_154;  mul_153 = mul_154 = None
    q_11 = q_embed_11.contiguous();  q_embed_11 = None
    k_11 = k_embed_11.contiguous();  k_embed_11 = None
    v_11 = value_states_11.contiguous();  value_states_11 = None
    npu_fused_infer_attention_score_default_11 = torch.ops.npu.npu_fused_infer_attention_score.default(q_11, k_11, v_11, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_11 = npu_fused_infer_attention_score_default_11[0];  npu_fused_infer_attention_score_default_11 = None
    attn_output_33 = output_11.transpose(1, 2)
    reshape_11 = attn_output_33.reshape(1, 2048, -1);  attn_output_33 = None
    attn_output_34 = reshape_11.contiguous();  reshape_11 = None
    l__self___model_model_layers_11_self_attn_o_proj_weight = self.L__self___model_model_layers_11_self_attn_o_proj_weight
    attn_output_35 = torch._C._nn.linear(attn_output_34, l__self___model_model_layers_11_self_attn_o_proj_weight, None);  attn_output_34 = l__self___model_model_layers_11_self_attn_o_proj_weight = None
    hidden_states_139 = hidden_states_131 + attn_output_35;  hidden_states_131 = attn_output_35 = None
    hidden_states_140 = hidden_states_139.to(torch.float32)
    pow_48 = hidden_states_140.pow(2)
    variance_47 = pow_48.mean(-1, keepdim = True);  pow_48 = None
    add_94 = variance_47 + 1e-06;  variance_47 = None
    rsqrt_47 = torch.rsqrt(add_94);  add_94 = None
    hidden_states_141 = hidden_states_140 * rsqrt_47;  hidden_states_140 = rsqrt_47 = None
    l__self___model_model_layers_11_post_attention_layernorm_weight = self.L__self___model_model_layers_11_post_attention_layernorm_weight
    to_98 = hidden_states_141.to(torch.float16);  hidden_states_141 = None
    hidden_states_142 = l__self___model_model_layers_11_post_attention_layernorm_weight * to_98;  l__self___model_model_layers_11_post_attention_layernorm_weight = to_98 = None
    l__self___model_model_layers_11_mlp_gate_proj_weight = self.L__self___model_model_layers_11_mlp_gate_proj_weight
    linear_81 = torch._C._nn.linear(hidden_states_142, l__self___model_model_layers_11_mlp_gate_proj_weight, None);  l__self___model_model_layers_11_mlp_gate_proj_weight = None
    silu_11 = torch.nn.functional.silu(linear_81);  linear_81 = None
    l__self___model_model_layers_11_mlp_up_proj_weight = self.L__self___model_model_layers_11_mlp_up_proj_weight
    linear_82 = torch._C._nn.linear(hidden_states_142, l__self___model_model_layers_11_mlp_up_proj_weight, None);  hidden_states_142 = l__self___model_model_layers_11_mlp_up_proj_weight = None
    mul_157 = silu_11 * linear_82;  silu_11 = linear_82 = None
    l__self___model_model_layers_11_mlp_down_proj_weight = self.L__self___model_model_layers_11_mlp_down_proj_weight
    down_proj_11 = torch._C._nn.linear(mul_157, l__self___model_model_layers_11_mlp_down_proj_weight, None);  mul_157 = l__self___model_model_layers_11_mlp_down_proj_weight = None
    hidden_states_143 = hidden_states_139 + down_proj_11;  hidden_states_139 = down_proj_11 = None
    hidden_states_144 = hidden_states_143.to(torch.float32)
    pow_49 = hidden_states_144.pow(2)
    variance_48 = pow_49.mean(-1, keepdim = True);  pow_49 = None
    add_96 = variance_48 + 1e-06;  variance_48 = None
    rsqrt_48 = torch.rsqrt(add_96);  add_96 = None
    hidden_states_145 = hidden_states_144 * rsqrt_48;  hidden_states_144 = rsqrt_48 = None
    l__self___model_model_layers_12_input_layernorm_weight = self.L__self___model_model_layers_12_input_layernorm_weight
    to_100 = hidden_states_145.to(torch.float16);  hidden_states_145 = None
    hidden_states_146 = l__self___model_model_layers_12_input_layernorm_weight * to_100;  l__self___model_model_layers_12_input_layernorm_weight = to_100 = None
    l__self___model_model_layers_12_self_attn_q_proj_weight = self.L__self___model_model_layers_12_self_attn_q_proj_weight
    linear_84 = torch._C._nn.linear(hidden_states_146, l__self___model_model_layers_12_self_attn_q_proj_weight, None);  l__self___model_model_layers_12_self_attn_q_proj_weight = None
    view_36 = linear_84.view((1, 2048, -1, 128));  linear_84 = None
    hidden_states_147 = view_36.to(torch.float32);  view_36 = None
    pow_50 = hidden_states_147.pow(2)
    variance_49 = pow_50.mean(-1, keepdim = True);  pow_50 = None
    add_97 = variance_49 + 1e-06;  variance_49 = None
    rsqrt_49 = torch.rsqrt(add_97);  add_97 = None
    hidden_states_148 = hidden_states_147 * rsqrt_49;  hidden_states_147 = rsqrt_49 = None
    l__self___model_model_layers_12_self_attn_q_norm_weight = self.L__self___model_model_layers_12_self_attn_q_norm_weight
    to_102 = hidden_states_148.to(torch.float16);  hidden_states_148 = None
    mul_161 = l__self___model_model_layers_12_self_attn_q_norm_weight * to_102;  l__self___model_model_layers_12_self_attn_q_norm_weight = to_102 = None
    query_states_12 = mul_161.transpose(1, 2);  mul_161 = None
    l__self___model_model_layers_12_self_attn_k_proj_weight = self.L__self___model_model_layers_12_self_attn_k_proj_weight
    linear_85 = torch._C._nn.linear(hidden_states_146, l__self___model_model_layers_12_self_attn_k_proj_weight, None);  l__self___model_model_layers_12_self_attn_k_proj_weight = None
    view_37 = linear_85.view((1, 2048, -1, 128));  linear_85 = None
    hidden_states_149 = view_37.to(torch.float32);  view_37 = None
    pow_51 = hidden_states_149.pow(2)
    variance_50 = pow_51.mean(-1, keepdim = True);  pow_51 = None
    add_98 = variance_50 + 1e-06;  variance_50 = None
    rsqrt_50 = torch.rsqrt(add_98);  add_98 = None
    hidden_states_150 = hidden_states_149 * rsqrt_50;  hidden_states_149 = rsqrt_50 = None
    l__self___model_model_layers_12_self_attn_k_norm_weight = self.L__self___model_model_layers_12_self_attn_k_norm_weight
    to_104 = hidden_states_150.to(torch.float16);  hidden_states_150 = None
    mul_163 = l__self___model_model_layers_12_self_attn_k_norm_weight * to_104;  l__self___model_model_layers_12_self_attn_k_norm_weight = to_104 = None
    key_states_12 = mul_163.transpose(1, 2);  mul_163 = None
    l__self___model_model_layers_12_self_attn_v_proj_weight = self.L__self___model_model_layers_12_self_attn_v_proj_weight
    linear_86 = torch._C._nn.linear(hidden_states_146, l__self___model_model_layers_12_self_attn_v_proj_weight, None);  hidden_states_146 = l__self___model_model_layers_12_self_attn_v_proj_weight = None
    view_38 = linear_86.view((1, 2048, -1, 128));  linear_86 = None
    value_states_12 = view_38.transpose(1, 2);  view_38 = None
    cos_15 = cos_2.unsqueeze(1)
    sin_15 = sin_2.unsqueeze(1)
    mul_164 = query_states_12 * cos_15
    x1_24 = query_states_12[(Ellipsis, slice(None, 64, None))]
    x2_24 = query_states_12[(Ellipsis, slice(64, None, None))];  query_states_12 = None
    neg_24 = -x2_24;  x2_24 = None
    cat_25 = torch.cat((neg_24, x1_24), dim = -1);  neg_24 = x1_24 = None
    mul_165 = cat_25 * sin_15;  cat_25 = None
    q_embed_12 = mul_164 + mul_165;  mul_164 = mul_165 = None
    mul_166 = key_states_12 * cos_15;  cos_15 = None
    x1_25 = key_states_12[(Ellipsis, slice(None, 64, None))]
    x2_25 = key_states_12[(Ellipsis, slice(64, None, None))];  key_states_12 = None
    neg_25 = -x2_25;  x2_25 = None
    cat_26 = torch.cat((neg_25, x1_25), dim = -1);  neg_25 = x1_25 = None
    mul_167 = cat_26 * sin_15;  cat_26 = sin_15 = None
    k_embed_12 = mul_166 + mul_167;  mul_166 = mul_167 = None
    q_12 = q_embed_12.contiguous();  q_embed_12 = None
    k_12 = k_embed_12.contiguous();  k_embed_12 = None
    v_12 = value_states_12.contiguous();  value_states_12 = None
    npu_fused_infer_attention_score_default_12 = torch.ops.npu.npu_fused_infer_attention_score.default(q_12, k_12, v_12, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_12 = npu_fused_infer_attention_score_default_12[0];  npu_fused_infer_attention_score_default_12 = None
    attn_output_36 = output_12.transpose(1, 2)
    reshape_12 = attn_output_36.reshape(1, 2048, -1);  attn_output_36 = None
    attn_output_37 = reshape_12.contiguous();  reshape_12 = None
    l__self___model_model_layers_12_self_attn_o_proj_weight = self.L__self___model_model_layers_12_self_attn_o_proj_weight
    attn_output_38 = torch._C._nn.linear(attn_output_37, l__self___model_model_layers_12_self_attn_o_proj_weight, None);  attn_output_37 = l__self___model_model_layers_12_self_attn_o_proj_weight = None
    hidden_states_151 = hidden_states_143 + attn_output_38;  hidden_states_143 = attn_output_38 = None
    hidden_states_152 = hidden_states_151.to(torch.float32)
    pow_52 = hidden_states_152.pow(2)
    variance_51 = pow_52.mean(-1, keepdim = True);  pow_52 = None
    add_102 = variance_51 + 1e-06;  variance_51 = None
    rsqrt_51 = torch.rsqrt(add_102);  add_102 = None
    hidden_states_153 = hidden_states_152 * rsqrt_51;  hidden_states_152 = rsqrt_51 = None
    l__self___model_model_layers_12_post_attention_layernorm_weight = self.L__self___model_model_layers_12_post_attention_layernorm_weight
    to_106 = hidden_states_153.to(torch.float16);  hidden_states_153 = None
    hidden_states_154 = l__self___model_model_layers_12_post_attention_layernorm_weight * to_106;  l__self___model_model_layers_12_post_attention_layernorm_weight = to_106 = None
    l__self___model_model_layers_12_mlp_gate_proj_weight = self.L__self___model_model_layers_12_mlp_gate_proj_weight
    linear_88 = torch._C._nn.linear(hidden_states_154, l__self___model_model_layers_12_mlp_gate_proj_weight, None);  l__self___model_model_layers_12_mlp_gate_proj_weight = None
    silu_12 = torch.nn.functional.silu(linear_88);  linear_88 = None
    l__self___model_model_layers_12_mlp_up_proj_weight = self.L__self___model_model_layers_12_mlp_up_proj_weight
    linear_89 = torch._C._nn.linear(hidden_states_154, l__self___model_model_layers_12_mlp_up_proj_weight, None);  hidden_states_154 = l__self___model_model_layers_12_mlp_up_proj_weight = None
    mul_170 = silu_12 * linear_89;  silu_12 = linear_89 = None
    l__self___model_model_layers_12_mlp_down_proj_weight = self.L__self___model_model_layers_12_mlp_down_proj_weight
    down_proj_12 = torch._C._nn.linear(mul_170, l__self___model_model_layers_12_mlp_down_proj_weight, None);  mul_170 = l__self___model_model_layers_12_mlp_down_proj_weight = None
    hidden_states_155 = hidden_states_151 + down_proj_12;  hidden_states_151 = down_proj_12 = None
    hidden_states_156 = hidden_states_155.to(torch.float32)
    pow_53 = hidden_states_156.pow(2)
    variance_52 = pow_53.mean(-1, keepdim = True);  pow_53 = None
    add_104 = variance_52 + 1e-06;  variance_52 = None
    rsqrt_52 = torch.rsqrt(add_104);  add_104 = None
    hidden_states_157 = hidden_states_156 * rsqrt_52;  hidden_states_156 = rsqrt_52 = None
    l__self___model_model_layers_13_input_layernorm_weight = self.L__self___model_model_layers_13_input_layernorm_weight
    to_108 = hidden_states_157.to(torch.float16);  hidden_states_157 = None
    hidden_states_158 = l__self___model_model_layers_13_input_layernorm_weight * to_108;  l__self___model_model_layers_13_input_layernorm_weight = to_108 = None
    l__self___model_model_layers_13_self_attn_q_proj_weight = self.L__self___model_model_layers_13_self_attn_q_proj_weight
    linear_91 = torch._C._nn.linear(hidden_states_158, l__self___model_model_layers_13_self_attn_q_proj_weight, None);  l__self___model_model_layers_13_self_attn_q_proj_weight = None
    view_39 = linear_91.view((1, 2048, -1, 128));  linear_91 = None
    hidden_states_159 = view_39.to(torch.float32);  view_39 = None
    pow_54 = hidden_states_159.pow(2)
    variance_53 = pow_54.mean(-1, keepdim = True);  pow_54 = None
    add_105 = variance_53 + 1e-06;  variance_53 = None
    rsqrt_53 = torch.rsqrt(add_105);  add_105 = None
    hidden_states_160 = hidden_states_159 * rsqrt_53;  hidden_states_159 = rsqrt_53 = None
    l__self___model_model_layers_13_self_attn_q_norm_weight = self.L__self___model_model_layers_13_self_attn_q_norm_weight
    to_110 = hidden_states_160.to(torch.float16);  hidden_states_160 = None
    mul_174 = l__self___model_model_layers_13_self_attn_q_norm_weight * to_110;  l__self___model_model_layers_13_self_attn_q_norm_weight = to_110 = None
    query_states_13 = mul_174.transpose(1, 2);  mul_174 = None
    l__self___model_model_layers_13_self_attn_k_proj_weight = self.L__self___model_model_layers_13_self_attn_k_proj_weight
    linear_92 = torch._C._nn.linear(hidden_states_158, l__self___model_model_layers_13_self_attn_k_proj_weight, None);  l__self___model_model_layers_13_self_attn_k_proj_weight = None
    view_40 = linear_92.view((1, 2048, -1, 128));  linear_92 = None
    hidden_states_161 = view_40.to(torch.float32);  view_40 = None
    pow_55 = hidden_states_161.pow(2)
    variance_54 = pow_55.mean(-1, keepdim = True);  pow_55 = None
    add_106 = variance_54 + 1e-06;  variance_54 = None
    rsqrt_54 = torch.rsqrt(add_106);  add_106 = None
    hidden_states_162 = hidden_states_161 * rsqrt_54;  hidden_states_161 = rsqrt_54 = None
    l__self___model_model_layers_13_self_attn_k_norm_weight = self.L__self___model_model_layers_13_self_attn_k_norm_weight
    to_112 = hidden_states_162.to(torch.float16);  hidden_states_162 = None
    mul_176 = l__self___model_model_layers_13_self_attn_k_norm_weight * to_112;  l__self___model_model_layers_13_self_attn_k_norm_weight = to_112 = None
    key_states_13 = mul_176.transpose(1, 2);  mul_176 = None
    l__self___model_model_layers_13_self_attn_v_proj_weight = self.L__self___model_model_layers_13_self_attn_v_proj_weight
    linear_93 = torch._C._nn.linear(hidden_states_158, l__self___model_model_layers_13_self_attn_v_proj_weight, None);  hidden_states_158 = l__self___model_model_layers_13_self_attn_v_proj_weight = None
    view_41 = linear_93.view((1, 2048, -1, 128));  linear_93 = None
    value_states_13 = view_41.transpose(1, 2);  view_41 = None
    cos_16 = cos_2.unsqueeze(1)
    sin_16 = sin_2.unsqueeze(1)
    mul_177 = query_states_13 * cos_16
    x1_26 = query_states_13[(Ellipsis, slice(None, 64, None))]
    x2_26 = query_states_13[(Ellipsis, slice(64, None, None))];  query_states_13 = None
    neg_26 = -x2_26;  x2_26 = None
    cat_27 = torch.cat((neg_26, x1_26), dim = -1);  neg_26 = x1_26 = None
    mul_178 = cat_27 * sin_16;  cat_27 = None
    q_embed_13 = mul_177 + mul_178;  mul_177 = mul_178 = None
    mul_179 = key_states_13 * cos_16;  cos_16 = None
    x1_27 = key_states_13[(Ellipsis, slice(None, 64, None))]
    x2_27 = key_states_13[(Ellipsis, slice(64, None, None))];  key_states_13 = None
    neg_27 = -x2_27;  x2_27 = None
    cat_28 = torch.cat((neg_27, x1_27), dim = -1);  neg_27 = x1_27 = None
    mul_180 = cat_28 * sin_16;  cat_28 = sin_16 = None
    k_embed_13 = mul_179 + mul_180;  mul_179 = mul_180 = None
    q_13 = q_embed_13.contiguous();  q_embed_13 = None
    k_13 = k_embed_13.contiguous();  k_embed_13 = None
    v_13 = value_states_13.contiguous();  value_states_13 = None
    npu_fused_infer_attention_score_default_13 = torch.ops.npu.npu_fused_infer_attention_score.default(q_13, k_13, v_13, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_13 = npu_fused_infer_attention_score_default_13[0];  npu_fused_infer_attention_score_default_13 = None
    attn_output_39 = output_13.transpose(1, 2)
    reshape_13 = attn_output_39.reshape(1, 2048, -1);  attn_output_39 = None
    attn_output_40 = reshape_13.contiguous();  reshape_13 = None
    l__self___model_model_layers_13_self_attn_o_proj_weight = self.L__self___model_model_layers_13_self_attn_o_proj_weight
    attn_output_41 = torch._C._nn.linear(attn_output_40, l__self___model_model_layers_13_self_attn_o_proj_weight, None);  attn_output_40 = l__self___model_model_layers_13_self_attn_o_proj_weight = None
    hidden_states_163 = hidden_states_155 + attn_output_41;  hidden_states_155 = attn_output_41 = None
    hidden_states_164 = hidden_states_163.to(torch.float32)
    pow_56 = hidden_states_164.pow(2)
    variance_55 = pow_56.mean(-1, keepdim = True);  pow_56 = None
    add_110 = variance_55 + 1e-06;  variance_55 = None
    rsqrt_55 = torch.rsqrt(add_110);  add_110 = None
    hidden_states_165 = hidden_states_164 * rsqrt_55;  hidden_states_164 = rsqrt_55 = None
    l__self___model_model_layers_13_post_attention_layernorm_weight = self.L__self___model_model_layers_13_post_attention_layernorm_weight
    to_114 = hidden_states_165.to(torch.float16);  hidden_states_165 = None
    hidden_states_166 = l__self___model_model_layers_13_post_attention_layernorm_weight * to_114;  l__self___model_model_layers_13_post_attention_layernorm_weight = to_114 = None
    l__self___model_model_layers_13_mlp_gate_proj_weight = self.L__self___model_model_layers_13_mlp_gate_proj_weight
    linear_95 = torch._C._nn.linear(hidden_states_166, l__self___model_model_layers_13_mlp_gate_proj_weight, None);  l__self___model_model_layers_13_mlp_gate_proj_weight = None
    silu_13 = torch.nn.functional.silu(linear_95);  linear_95 = None
    l__self___model_model_layers_13_mlp_up_proj_weight = self.L__self___model_model_layers_13_mlp_up_proj_weight
    linear_96 = torch._C._nn.linear(hidden_states_166, l__self___model_model_layers_13_mlp_up_proj_weight, None);  hidden_states_166 = l__self___model_model_layers_13_mlp_up_proj_weight = None
    mul_183 = silu_13 * linear_96;  silu_13 = linear_96 = None
    l__self___model_model_layers_13_mlp_down_proj_weight = self.L__self___model_model_layers_13_mlp_down_proj_weight
    down_proj_13 = torch._C._nn.linear(mul_183, l__self___model_model_layers_13_mlp_down_proj_weight, None);  mul_183 = l__self___model_model_layers_13_mlp_down_proj_weight = None
    hidden_states_167 = hidden_states_163 + down_proj_13;  hidden_states_163 = down_proj_13 = None
    hidden_states_168 = hidden_states_167.to(torch.float32)
    pow_57 = hidden_states_168.pow(2)
    variance_56 = pow_57.mean(-1, keepdim = True);  pow_57 = None
    add_112 = variance_56 + 1e-06;  variance_56 = None
    rsqrt_56 = torch.rsqrt(add_112);  add_112 = None
    hidden_states_169 = hidden_states_168 * rsqrt_56;  hidden_states_168 = rsqrt_56 = None
    l__self___model_model_layers_14_input_layernorm_weight = self.L__self___model_model_layers_14_input_layernorm_weight
    to_116 = hidden_states_169.to(torch.float16);  hidden_states_169 = None
    hidden_states_170 = l__self___model_model_layers_14_input_layernorm_weight * to_116;  l__self___model_model_layers_14_input_layernorm_weight = to_116 = None
    l__self___model_model_layers_14_self_attn_q_proj_weight = self.L__self___model_model_layers_14_self_attn_q_proj_weight
    linear_98 = torch._C._nn.linear(hidden_states_170, l__self___model_model_layers_14_self_attn_q_proj_weight, None);  l__self___model_model_layers_14_self_attn_q_proj_weight = None
    view_42 = linear_98.view((1, 2048, -1, 128));  linear_98 = None
    hidden_states_171 = view_42.to(torch.float32);  view_42 = None
    pow_58 = hidden_states_171.pow(2)
    variance_57 = pow_58.mean(-1, keepdim = True);  pow_58 = None
    add_113 = variance_57 + 1e-06;  variance_57 = None
    rsqrt_57 = torch.rsqrt(add_113);  add_113 = None
    hidden_states_172 = hidden_states_171 * rsqrt_57;  hidden_states_171 = rsqrt_57 = None
    l__self___model_model_layers_14_self_attn_q_norm_weight = self.L__self___model_model_layers_14_self_attn_q_norm_weight
    to_118 = hidden_states_172.to(torch.float16);  hidden_states_172 = None
    mul_187 = l__self___model_model_layers_14_self_attn_q_norm_weight * to_118;  l__self___model_model_layers_14_self_attn_q_norm_weight = to_118 = None
    query_states_14 = mul_187.transpose(1, 2);  mul_187 = None
    l__self___model_model_layers_14_self_attn_k_proj_weight = self.L__self___model_model_layers_14_self_attn_k_proj_weight
    linear_99 = torch._C._nn.linear(hidden_states_170, l__self___model_model_layers_14_self_attn_k_proj_weight, None);  l__self___model_model_layers_14_self_attn_k_proj_weight = None
    view_43 = linear_99.view((1, 2048, -1, 128));  linear_99 = None
    hidden_states_173 = view_43.to(torch.float32);  view_43 = None
    pow_59 = hidden_states_173.pow(2)
    variance_58 = pow_59.mean(-1, keepdim = True);  pow_59 = None
    add_114 = variance_58 + 1e-06;  variance_58 = None
    rsqrt_58 = torch.rsqrt(add_114);  add_114 = None
    hidden_states_174 = hidden_states_173 * rsqrt_58;  hidden_states_173 = rsqrt_58 = None
    l__self___model_model_layers_14_self_attn_k_norm_weight = self.L__self___model_model_layers_14_self_attn_k_norm_weight
    to_120 = hidden_states_174.to(torch.float16);  hidden_states_174 = None
    mul_189 = l__self___model_model_layers_14_self_attn_k_norm_weight * to_120;  l__self___model_model_layers_14_self_attn_k_norm_weight = to_120 = None
    key_states_14 = mul_189.transpose(1, 2);  mul_189 = None
    l__self___model_model_layers_14_self_attn_v_proj_weight = self.L__self___model_model_layers_14_self_attn_v_proj_weight
    linear_100 = torch._C._nn.linear(hidden_states_170, l__self___model_model_layers_14_self_attn_v_proj_weight, None);  hidden_states_170 = l__self___model_model_layers_14_self_attn_v_proj_weight = None
    view_44 = linear_100.view((1, 2048, -1, 128));  linear_100 = None
    value_states_14 = view_44.transpose(1, 2);  view_44 = None
    cos_17 = cos_2.unsqueeze(1)
    sin_17 = sin_2.unsqueeze(1)
    mul_190 = query_states_14 * cos_17
    x1_28 = query_states_14[(Ellipsis, slice(None, 64, None))]
    x2_28 = query_states_14[(Ellipsis, slice(64, None, None))];  query_states_14 = None
    neg_28 = -x2_28;  x2_28 = None
    cat_29 = torch.cat((neg_28, x1_28), dim = -1);  neg_28 = x1_28 = None
    mul_191 = cat_29 * sin_17;  cat_29 = None
    q_embed_14 = mul_190 + mul_191;  mul_190 = mul_191 = None
    mul_192 = key_states_14 * cos_17;  cos_17 = None
    x1_29 = key_states_14[(Ellipsis, slice(None, 64, None))]
    x2_29 = key_states_14[(Ellipsis, slice(64, None, None))];  key_states_14 = None
    neg_29 = -x2_29;  x2_29 = None
    cat_30 = torch.cat((neg_29, x1_29), dim = -1);  neg_29 = x1_29 = None
    mul_193 = cat_30 * sin_17;  cat_30 = sin_17 = None
    k_embed_14 = mul_192 + mul_193;  mul_192 = mul_193 = None
    q_14 = q_embed_14.contiguous();  q_embed_14 = None
    k_14 = k_embed_14.contiguous();  k_embed_14 = None
    v_14 = value_states_14.contiguous();  value_states_14 = None
    npu_fused_infer_attention_score_default_14 = torch.ops.npu.npu_fused_infer_attention_score.default(q_14, k_14, v_14, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_14 = npu_fused_infer_attention_score_default_14[0];  npu_fused_infer_attention_score_default_14 = None
    attn_output_42 = output_14.transpose(1, 2)
    reshape_14 = attn_output_42.reshape(1, 2048, -1);  attn_output_42 = None
    attn_output_43 = reshape_14.contiguous();  reshape_14 = None
    l__self___model_model_layers_14_self_attn_o_proj_weight = self.L__self___model_model_layers_14_self_attn_o_proj_weight
    attn_output_44 = torch._C._nn.linear(attn_output_43, l__self___model_model_layers_14_self_attn_o_proj_weight, None);  attn_output_43 = l__self___model_model_layers_14_self_attn_o_proj_weight = None
    hidden_states_175 = hidden_states_167 + attn_output_44;  hidden_states_167 = attn_output_44 = None
    hidden_states_176 = hidden_states_175.to(torch.float32)
    pow_60 = hidden_states_176.pow(2)
    variance_59 = pow_60.mean(-1, keepdim = True);  pow_60 = None
    add_118 = variance_59 + 1e-06;  variance_59 = None
    rsqrt_59 = torch.rsqrt(add_118);  add_118 = None
    hidden_states_177 = hidden_states_176 * rsqrt_59;  hidden_states_176 = rsqrt_59 = None
    l__self___model_model_layers_14_post_attention_layernorm_weight = self.L__self___model_model_layers_14_post_attention_layernorm_weight
    to_122 = hidden_states_177.to(torch.float16);  hidden_states_177 = None
    hidden_states_178 = l__self___model_model_layers_14_post_attention_layernorm_weight * to_122;  l__self___model_model_layers_14_post_attention_layernorm_weight = to_122 = None
    l__self___model_model_layers_14_mlp_gate_proj_weight = self.L__self___model_model_layers_14_mlp_gate_proj_weight
    linear_102 = torch._C._nn.linear(hidden_states_178, l__self___model_model_layers_14_mlp_gate_proj_weight, None);  l__self___model_model_layers_14_mlp_gate_proj_weight = None
    silu_14 = torch.nn.functional.silu(linear_102);  linear_102 = None
    l__self___model_model_layers_14_mlp_up_proj_weight = self.L__self___model_model_layers_14_mlp_up_proj_weight
    linear_103 = torch._C._nn.linear(hidden_states_178, l__self___model_model_layers_14_mlp_up_proj_weight, None);  hidden_states_178 = l__self___model_model_layers_14_mlp_up_proj_weight = None
    mul_196 = silu_14 * linear_103;  silu_14 = linear_103 = None
    l__self___model_model_layers_14_mlp_down_proj_weight = self.L__self___model_model_layers_14_mlp_down_proj_weight
    down_proj_14 = torch._C._nn.linear(mul_196, l__self___model_model_layers_14_mlp_down_proj_weight, None);  mul_196 = l__self___model_model_layers_14_mlp_down_proj_weight = None
    hidden_states_179 = hidden_states_175 + down_proj_14;  hidden_states_175 = down_proj_14 = None
    hidden_states_180 = hidden_states_179.to(torch.float32)
    pow_61 = hidden_states_180.pow(2)
    variance_60 = pow_61.mean(-1, keepdim = True);  pow_61 = None
    add_120 = variance_60 + 1e-06;  variance_60 = None
    rsqrt_60 = torch.rsqrt(add_120);  add_120 = None
    hidden_states_181 = hidden_states_180 * rsqrt_60;  hidden_states_180 = rsqrt_60 = None
    l__self___model_model_layers_15_input_layernorm_weight = self.L__self___model_model_layers_15_input_layernorm_weight
    to_124 = hidden_states_181.to(torch.float16);  hidden_states_181 = None
    hidden_states_182 = l__self___model_model_layers_15_input_layernorm_weight * to_124;  l__self___model_model_layers_15_input_layernorm_weight = to_124 = None
    l__self___model_model_layers_15_self_attn_q_proj_weight = self.L__self___model_model_layers_15_self_attn_q_proj_weight
    linear_105 = torch._C._nn.linear(hidden_states_182, l__self___model_model_layers_15_self_attn_q_proj_weight, None);  l__self___model_model_layers_15_self_attn_q_proj_weight = None
    view_45 = linear_105.view((1, 2048, -1, 128));  linear_105 = None
    hidden_states_183 = view_45.to(torch.float32);  view_45 = None
    pow_62 = hidden_states_183.pow(2)
    variance_61 = pow_62.mean(-1, keepdim = True);  pow_62 = None
    add_121 = variance_61 + 1e-06;  variance_61 = None
    rsqrt_61 = torch.rsqrt(add_121);  add_121 = None
    hidden_states_184 = hidden_states_183 * rsqrt_61;  hidden_states_183 = rsqrt_61 = None
    l__self___model_model_layers_15_self_attn_q_norm_weight = self.L__self___model_model_layers_15_self_attn_q_norm_weight
    to_126 = hidden_states_184.to(torch.float16);  hidden_states_184 = None
    mul_200 = l__self___model_model_layers_15_self_attn_q_norm_weight * to_126;  l__self___model_model_layers_15_self_attn_q_norm_weight = to_126 = None
    query_states_15 = mul_200.transpose(1, 2);  mul_200 = None
    l__self___model_model_layers_15_self_attn_k_proj_weight = self.L__self___model_model_layers_15_self_attn_k_proj_weight
    linear_106 = torch._C._nn.linear(hidden_states_182, l__self___model_model_layers_15_self_attn_k_proj_weight, None);  l__self___model_model_layers_15_self_attn_k_proj_weight = None
    view_46 = linear_106.view((1, 2048, -1, 128));  linear_106 = None
    hidden_states_185 = view_46.to(torch.float32);  view_46 = None
    pow_63 = hidden_states_185.pow(2)
    variance_62 = pow_63.mean(-1, keepdim = True);  pow_63 = None
    add_122 = variance_62 + 1e-06;  variance_62 = None
    rsqrt_62 = torch.rsqrt(add_122);  add_122 = None
    hidden_states_186 = hidden_states_185 * rsqrt_62;  hidden_states_185 = rsqrt_62 = None
    l__self___model_model_layers_15_self_attn_k_norm_weight = self.L__self___model_model_layers_15_self_attn_k_norm_weight
    to_128 = hidden_states_186.to(torch.float16);  hidden_states_186 = None
    mul_202 = l__self___model_model_layers_15_self_attn_k_norm_weight * to_128;  l__self___model_model_layers_15_self_attn_k_norm_weight = to_128 = None
    key_states_15 = mul_202.transpose(1, 2);  mul_202 = None
    l__self___model_model_layers_15_self_attn_v_proj_weight = self.L__self___model_model_layers_15_self_attn_v_proj_weight
    linear_107 = torch._C._nn.linear(hidden_states_182, l__self___model_model_layers_15_self_attn_v_proj_weight, None);  hidden_states_182 = l__self___model_model_layers_15_self_attn_v_proj_weight = None
    view_47 = linear_107.view((1, 2048, -1, 128));  linear_107 = None
    value_states_15 = view_47.transpose(1, 2);  view_47 = None
    cos_18 = cos_2.unsqueeze(1)
    sin_18 = sin_2.unsqueeze(1)
    mul_203 = query_states_15 * cos_18
    x1_30 = query_states_15[(Ellipsis, slice(None, 64, None))]
    x2_30 = query_states_15[(Ellipsis, slice(64, None, None))];  query_states_15 = None
    neg_30 = -x2_30;  x2_30 = None
    cat_31 = torch.cat((neg_30, x1_30), dim = -1);  neg_30 = x1_30 = None
    mul_204 = cat_31 * sin_18;  cat_31 = None
    q_embed_15 = mul_203 + mul_204;  mul_203 = mul_204 = None
    mul_205 = key_states_15 * cos_18;  cos_18 = None
    x1_31 = key_states_15[(Ellipsis, slice(None, 64, None))]
    x2_31 = key_states_15[(Ellipsis, slice(64, None, None))];  key_states_15 = None
    neg_31 = -x2_31;  x2_31 = None
    cat_32 = torch.cat((neg_31, x1_31), dim = -1);  neg_31 = x1_31 = None
    mul_206 = cat_32 * sin_18;  cat_32 = sin_18 = None
    k_embed_15 = mul_205 + mul_206;  mul_205 = mul_206 = None
    q_15 = q_embed_15.contiguous();  q_embed_15 = None
    k_15 = k_embed_15.contiguous();  k_embed_15 = None
    v_15 = value_states_15.contiguous();  value_states_15 = None
    npu_fused_infer_attention_score_default_15 = torch.ops.npu.npu_fused_infer_attention_score.default(q_15, k_15, v_15, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_15 = npu_fused_infer_attention_score_default_15[0];  npu_fused_infer_attention_score_default_15 = None
    attn_output_45 = output_15.transpose(1, 2)
    reshape_15 = attn_output_45.reshape(1, 2048, -1);  attn_output_45 = None
    attn_output_46 = reshape_15.contiguous();  reshape_15 = None
    l__self___model_model_layers_15_self_attn_o_proj_weight = self.L__self___model_model_layers_15_self_attn_o_proj_weight
    attn_output_47 = torch._C._nn.linear(attn_output_46, l__self___model_model_layers_15_self_attn_o_proj_weight, None);  attn_output_46 = l__self___model_model_layers_15_self_attn_o_proj_weight = None
    hidden_states_187 = hidden_states_179 + attn_output_47;  hidden_states_179 = attn_output_47 = None
    hidden_states_188 = hidden_states_187.to(torch.float32)
    pow_64 = hidden_states_188.pow(2)
    variance_63 = pow_64.mean(-1, keepdim = True);  pow_64 = None
    add_126 = variance_63 + 1e-06;  variance_63 = None
    rsqrt_63 = torch.rsqrt(add_126);  add_126 = None
    hidden_states_189 = hidden_states_188 * rsqrt_63;  hidden_states_188 = rsqrt_63 = None
    l__self___model_model_layers_15_post_attention_layernorm_weight = self.L__self___model_model_layers_15_post_attention_layernorm_weight
    to_130 = hidden_states_189.to(torch.float16);  hidden_states_189 = None
    hidden_states_190 = l__self___model_model_layers_15_post_attention_layernorm_weight * to_130;  l__self___model_model_layers_15_post_attention_layernorm_weight = to_130 = None
    l__self___model_model_layers_15_mlp_gate_proj_weight = self.L__self___model_model_layers_15_mlp_gate_proj_weight
    linear_109 = torch._C._nn.linear(hidden_states_190, l__self___model_model_layers_15_mlp_gate_proj_weight, None);  l__self___model_model_layers_15_mlp_gate_proj_weight = None
    silu_15 = torch.nn.functional.silu(linear_109);  linear_109 = None
    l__self___model_model_layers_15_mlp_up_proj_weight = self.L__self___model_model_layers_15_mlp_up_proj_weight
    linear_110 = torch._C._nn.linear(hidden_states_190, l__self___model_model_layers_15_mlp_up_proj_weight, None);  hidden_states_190 = l__self___model_model_layers_15_mlp_up_proj_weight = None
    mul_209 = silu_15 * linear_110;  silu_15 = linear_110 = None
    l__self___model_model_layers_15_mlp_down_proj_weight = self.L__self___model_model_layers_15_mlp_down_proj_weight
    down_proj_15 = torch._C._nn.linear(mul_209, l__self___model_model_layers_15_mlp_down_proj_weight, None);  mul_209 = l__self___model_model_layers_15_mlp_down_proj_weight = None
    hidden_states_191 = hidden_states_187 + down_proj_15;  hidden_states_187 = down_proj_15 = None
    hidden_states_192 = hidden_states_191.to(torch.float32)
    pow_65 = hidden_states_192.pow(2)
    variance_64 = pow_65.mean(-1, keepdim = True);  pow_65 = None
    add_128 = variance_64 + 1e-06;  variance_64 = None
    rsqrt_64 = torch.rsqrt(add_128);  add_128 = None
    hidden_states_193 = hidden_states_192 * rsqrt_64;  hidden_states_192 = rsqrt_64 = None
    l__self___model_model_layers_16_input_layernorm_weight = self.L__self___model_model_layers_16_input_layernorm_weight
    to_132 = hidden_states_193.to(torch.float16);  hidden_states_193 = None
    hidden_states_194 = l__self___model_model_layers_16_input_layernorm_weight * to_132;  l__self___model_model_layers_16_input_layernorm_weight = to_132 = None
    l__self___model_model_layers_16_self_attn_q_proj_weight = self.L__self___model_model_layers_16_self_attn_q_proj_weight
    linear_112 = torch._C._nn.linear(hidden_states_194, l__self___model_model_layers_16_self_attn_q_proj_weight, None);  l__self___model_model_layers_16_self_attn_q_proj_weight = None
    view_48 = linear_112.view((1, 2048, -1, 128));  linear_112 = None
    hidden_states_195 = view_48.to(torch.float32);  view_48 = None
    pow_66 = hidden_states_195.pow(2)
    variance_65 = pow_66.mean(-1, keepdim = True);  pow_66 = None
    add_129 = variance_65 + 1e-06;  variance_65 = None
    rsqrt_65 = torch.rsqrt(add_129);  add_129 = None
    hidden_states_196 = hidden_states_195 * rsqrt_65;  hidden_states_195 = rsqrt_65 = None
    l__self___model_model_layers_16_self_attn_q_norm_weight = self.L__self___model_model_layers_16_self_attn_q_norm_weight
    to_134 = hidden_states_196.to(torch.float16);  hidden_states_196 = None
    mul_213 = l__self___model_model_layers_16_self_attn_q_norm_weight * to_134;  l__self___model_model_layers_16_self_attn_q_norm_weight = to_134 = None
    query_states_16 = mul_213.transpose(1, 2);  mul_213 = None
    l__self___model_model_layers_16_self_attn_k_proj_weight = self.L__self___model_model_layers_16_self_attn_k_proj_weight
    linear_113 = torch._C._nn.linear(hidden_states_194, l__self___model_model_layers_16_self_attn_k_proj_weight, None);  l__self___model_model_layers_16_self_attn_k_proj_weight = None
    view_49 = linear_113.view((1, 2048, -1, 128));  linear_113 = None
    hidden_states_197 = view_49.to(torch.float32);  view_49 = None
    pow_67 = hidden_states_197.pow(2)
    variance_66 = pow_67.mean(-1, keepdim = True);  pow_67 = None
    add_130 = variance_66 + 1e-06;  variance_66 = None
    rsqrt_66 = torch.rsqrt(add_130);  add_130 = None
    hidden_states_198 = hidden_states_197 * rsqrt_66;  hidden_states_197 = rsqrt_66 = None
    l__self___model_model_layers_16_self_attn_k_norm_weight = self.L__self___model_model_layers_16_self_attn_k_norm_weight
    to_136 = hidden_states_198.to(torch.float16);  hidden_states_198 = None
    mul_215 = l__self___model_model_layers_16_self_attn_k_norm_weight * to_136;  l__self___model_model_layers_16_self_attn_k_norm_weight = to_136 = None
    key_states_16 = mul_215.transpose(1, 2);  mul_215 = None
    l__self___model_model_layers_16_self_attn_v_proj_weight = self.L__self___model_model_layers_16_self_attn_v_proj_weight
    linear_114 = torch._C._nn.linear(hidden_states_194, l__self___model_model_layers_16_self_attn_v_proj_weight, None);  hidden_states_194 = l__self___model_model_layers_16_self_attn_v_proj_weight = None
    view_50 = linear_114.view((1, 2048, -1, 128));  linear_114 = None
    value_states_16 = view_50.transpose(1, 2);  view_50 = None
    cos_19 = cos_2.unsqueeze(1)
    sin_19 = sin_2.unsqueeze(1)
    mul_216 = query_states_16 * cos_19
    x1_32 = query_states_16[(Ellipsis, slice(None, 64, None))]
    x2_32 = query_states_16[(Ellipsis, slice(64, None, None))];  query_states_16 = None
    neg_32 = -x2_32;  x2_32 = None
    cat_33 = torch.cat((neg_32, x1_32), dim = -1);  neg_32 = x1_32 = None
    mul_217 = cat_33 * sin_19;  cat_33 = None
    q_embed_16 = mul_216 + mul_217;  mul_216 = mul_217 = None
    mul_218 = key_states_16 * cos_19;  cos_19 = None
    x1_33 = key_states_16[(Ellipsis, slice(None, 64, None))]
    x2_33 = key_states_16[(Ellipsis, slice(64, None, None))];  key_states_16 = None
    neg_33 = -x2_33;  x2_33 = None
    cat_34 = torch.cat((neg_33, x1_33), dim = -1);  neg_33 = x1_33 = None
    mul_219 = cat_34 * sin_19;  cat_34 = sin_19 = None
    k_embed_16 = mul_218 + mul_219;  mul_218 = mul_219 = None
    q_16 = q_embed_16.contiguous();  q_embed_16 = None
    k_16 = k_embed_16.contiguous();  k_embed_16 = None
    v_16 = value_states_16.contiguous();  value_states_16 = None
    npu_fused_infer_attention_score_default_16 = torch.ops.npu.npu_fused_infer_attention_score.default(q_16, k_16, v_16, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_16 = npu_fused_infer_attention_score_default_16[0];  npu_fused_infer_attention_score_default_16 = None
    attn_output_48 = output_16.transpose(1, 2)
    reshape_16 = attn_output_48.reshape(1, 2048, -1);  attn_output_48 = None
    attn_output_49 = reshape_16.contiguous();  reshape_16 = None
    l__self___model_model_layers_16_self_attn_o_proj_weight = self.L__self___model_model_layers_16_self_attn_o_proj_weight
    attn_output_50 = torch._C._nn.linear(attn_output_49, l__self___model_model_layers_16_self_attn_o_proj_weight, None);  attn_output_49 = l__self___model_model_layers_16_self_attn_o_proj_weight = None
    hidden_states_199 = hidden_states_191 + attn_output_50;  hidden_states_191 = attn_output_50 = None
    hidden_states_200 = hidden_states_199.to(torch.float32)
    pow_68 = hidden_states_200.pow(2)
    variance_67 = pow_68.mean(-1, keepdim = True);  pow_68 = None
    add_134 = variance_67 + 1e-06;  variance_67 = None
    rsqrt_67 = torch.rsqrt(add_134);  add_134 = None
    hidden_states_201 = hidden_states_200 * rsqrt_67;  hidden_states_200 = rsqrt_67 = None
    l__self___model_model_layers_16_post_attention_layernorm_weight = self.L__self___model_model_layers_16_post_attention_layernorm_weight
    to_138 = hidden_states_201.to(torch.float16);  hidden_states_201 = None
    hidden_states_202 = l__self___model_model_layers_16_post_attention_layernorm_weight * to_138;  l__self___model_model_layers_16_post_attention_layernorm_weight = to_138 = None
    l__self___model_model_layers_16_mlp_gate_proj_weight = self.L__self___model_model_layers_16_mlp_gate_proj_weight
    linear_116 = torch._C._nn.linear(hidden_states_202, l__self___model_model_layers_16_mlp_gate_proj_weight, None);  l__self___model_model_layers_16_mlp_gate_proj_weight = None
    silu_16 = torch.nn.functional.silu(linear_116);  linear_116 = None
    l__self___model_model_layers_16_mlp_up_proj_weight = self.L__self___model_model_layers_16_mlp_up_proj_weight
    linear_117 = torch._C._nn.linear(hidden_states_202, l__self___model_model_layers_16_mlp_up_proj_weight, None);  hidden_states_202 = l__self___model_model_layers_16_mlp_up_proj_weight = None
    mul_222 = silu_16 * linear_117;  silu_16 = linear_117 = None
    l__self___model_model_layers_16_mlp_down_proj_weight = self.L__self___model_model_layers_16_mlp_down_proj_weight
    down_proj_16 = torch._C._nn.linear(mul_222, l__self___model_model_layers_16_mlp_down_proj_weight, None);  mul_222 = l__self___model_model_layers_16_mlp_down_proj_weight = None
    hidden_states_203 = hidden_states_199 + down_proj_16;  hidden_states_199 = down_proj_16 = None
    hidden_states_204 = hidden_states_203.to(torch.float32)
    pow_69 = hidden_states_204.pow(2)
    variance_68 = pow_69.mean(-1, keepdim = True);  pow_69 = None
    add_136 = variance_68 + 1e-06;  variance_68 = None
    rsqrt_68 = torch.rsqrt(add_136);  add_136 = None
    hidden_states_205 = hidden_states_204 * rsqrt_68;  hidden_states_204 = rsqrt_68 = None
    l__self___model_model_layers_17_input_layernorm_weight = self.L__self___model_model_layers_17_input_layernorm_weight
    to_140 = hidden_states_205.to(torch.float16);  hidden_states_205 = None
    hidden_states_206 = l__self___model_model_layers_17_input_layernorm_weight * to_140;  l__self___model_model_layers_17_input_layernorm_weight = to_140 = None
    l__self___model_model_layers_17_self_attn_q_proj_weight = self.L__self___model_model_layers_17_self_attn_q_proj_weight
    linear_119 = torch._C._nn.linear(hidden_states_206, l__self___model_model_layers_17_self_attn_q_proj_weight, None);  l__self___model_model_layers_17_self_attn_q_proj_weight = None
    view_51 = linear_119.view((1, 2048, -1, 128));  linear_119 = None
    hidden_states_207 = view_51.to(torch.float32);  view_51 = None
    pow_70 = hidden_states_207.pow(2)
    variance_69 = pow_70.mean(-1, keepdim = True);  pow_70 = None
    add_137 = variance_69 + 1e-06;  variance_69 = None
    rsqrt_69 = torch.rsqrt(add_137);  add_137 = None
    hidden_states_208 = hidden_states_207 * rsqrt_69;  hidden_states_207 = rsqrt_69 = None
    l__self___model_model_layers_17_self_attn_q_norm_weight = self.L__self___model_model_layers_17_self_attn_q_norm_weight
    to_142 = hidden_states_208.to(torch.float16);  hidden_states_208 = None
    mul_226 = l__self___model_model_layers_17_self_attn_q_norm_weight * to_142;  l__self___model_model_layers_17_self_attn_q_norm_weight = to_142 = None
    query_states_17 = mul_226.transpose(1, 2);  mul_226 = None
    l__self___model_model_layers_17_self_attn_k_proj_weight = self.L__self___model_model_layers_17_self_attn_k_proj_weight
    linear_120 = torch._C._nn.linear(hidden_states_206, l__self___model_model_layers_17_self_attn_k_proj_weight, None);  l__self___model_model_layers_17_self_attn_k_proj_weight = None
    view_52 = linear_120.view((1, 2048, -1, 128));  linear_120 = None
    hidden_states_209 = view_52.to(torch.float32);  view_52 = None
    pow_71 = hidden_states_209.pow(2)
    variance_70 = pow_71.mean(-1, keepdim = True);  pow_71 = None
    add_138 = variance_70 + 1e-06;  variance_70 = None
    rsqrt_70 = torch.rsqrt(add_138);  add_138 = None
    hidden_states_210 = hidden_states_209 * rsqrt_70;  hidden_states_209 = rsqrt_70 = None
    l__self___model_model_layers_17_self_attn_k_norm_weight = self.L__self___model_model_layers_17_self_attn_k_norm_weight
    to_144 = hidden_states_210.to(torch.float16);  hidden_states_210 = None
    mul_228 = l__self___model_model_layers_17_self_attn_k_norm_weight * to_144;  l__self___model_model_layers_17_self_attn_k_norm_weight = to_144 = None
    key_states_17 = mul_228.transpose(1, 2);  mul_228 = None
    l__self___model_model_layers_17_self_attn_v_proj_weight = self.L__self___model_model_layers_17_self_attn_v_proj_weight
    linear_121 = torch._C._nn.linear(hidden_states_206, l__self___model_model_layers_17_self_attn_v_proj_weight, None);  hidden_states_206 = l__self___model_model_layers_17_self_attn_v_proj_weight = None
    view_53 = linear_121.view((1, 2048, -1, 128));  linear_121 = None
    value_states_17 = view_53.transpose(1, 2);  view_53 = None
    cos_20 = cos_2.unsqueeze(1)
    sin_20 = sin_2.unsqueeze(1)
    mul_229 = query_states_17 * cos_20
    x1_34 = query_states_17[(Ellipsis, slice(None, 64, None))]
    x2_34 = query_states_17[(Ellipsis, slice(64, None, None))];  query_states_17 = None
    neg_34 = -x2_34;  x2_34 = None
    cat_35 = torch.cat((neg_34, x1_34), dim = -1);  neg_34 = x1_34 = None
    mul_230 = cat_35 * sin_20;  cat_35 = None
    q_embed_17 = mul_229 + mul_230;  mul_229 = mul_230 = None
    mul_231 = key_states_17 * cos_20;  cos_20 = None
    x1_35 = key_states_17[(Ellipsis, slice(None, 64, None))]
    x2_35 = key_states_17[(Ellipsis, slice(64, None, None))];  key_states_17 = None
    neg_35 = -x2_35;  x2_35 = None
    cat_36 = torch.cat((neg_35, x1_35), dim = -1);  neg_35 = x1_35 = None
    mul_232 = cat_36 * sin_20;  cat_36 = sin_20 = None
    k_embed_17 = mul_231 + mul_232;  mul_231 = mul_232 = None
    q_17 = q_embed_17.contiguous();  q_embed_17 = None
    k_17 = k_embed_17.contiguous();  k_embed_17 = None
    v_17 = value_states_17.contiguous();  value_states_17 = None
    npu_fused_infer_attention_score_default_17 = torch.ops.npu.npu_fused_infer_attention_score.default(q_17, k_17, v_17, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_17 = npu_fused_infer_attention_score_default_17[0];  npu_fused_infer_attention_score_default_17 = None
    attn_output_51 = output_17.transpose(1, 2)
    reshape_17 = attn_output_51.reshape(1, 2048, -1);  attn_output_51 = None
    attn_output_52 = reshape_17.contiguous();  reshape_17 = None
    l__self___model_model_layers_17_self_attn_o_proj_weight = self.L__self___model_model_layers_17_self_attn_o_proj_weight
    attn_output_53 = torch._C._nn.linear(attn_output_52, l__self___model_model_layers_17_self_attn_o_proj_weight, None);  attn_output_52 = l__self___model_model_layers_17_self_attn_o_proj_weight = None
    hidden_states_211 = hidden_states_203 + attn_output_53;  hidden_states_203 = attn_output_53 = None
    hidden_states_212 = hidden_states_211.to(torch.float32)
    pow_72 = hidden_states_212.pow(2)
    variance_71 = pow_72.mean(-1, keepdim = True);  pow_72 = None
    add_142 = variance_71 + 1e-06;  variance_71 = None
    rsqrt_71 = torch.rsqrt(add_142);  add_142 = None
    hidden_states_213 = hidden_states_212 * rsqrt_71;  hidden_states_212 = rsqrt_71 = None
    l__self___model_model_layers_17_post_attention_layernorm_weight = self.L__self___model_model_layers_17_post_attention_layernorm_weight
    to_146 = hidden_states_213.to(torch.float16);  hidden_states_213 = None
    hidden_states_214 = l__self___model_model_layers_17_post_attention_layernorm_weight * to_146;  l__self___model_model_layers_17_post_attention_layernorm_weight = to_146 = None
    l__self___model_model_layers_17_mlp_gate_proj_weight = self.L__self___model_model_layers_17_mlp_gate_proj_weight
    linear_123 = torch._C._nn.linear(hidden_states_214, l__self___model_model_layers_17_mlp_gate_proj_weight, None);  l__self___model_model_layers_17_mlp_gate_proj_weight = None
    silu_17 = torch.nn.functional.silu(linear_123);  linear_123 = None
    l__self___model_model_layers_17_mlp_up_proj_weight = self.L__self___model_model_layers_17_mlp_up_proj_weight
    linear_124 = torch._C._nn.linear(hidden_states_214, l__self___model_model_layers_17_mlp_up_proj_weight, None);  hidden_states_214 = l__self___model_model_layers_17_mlp_up_proj_weight = None
    mul_235 = silu_17 * linear_124;  silu_17 = linear_124 = None
    l__self___model_model_layers_17_mlp_down_proj_weight = self.L__self___model_model_layers_17_mlp_down_proj_weight
    down_proj_17 = torch._C._nn.linear(mul_235, l__self___model_model_layers_17_mlp_down_proj_weight, None);  mul_235 = l__self___model_model_layers_17_mlp_down_proj_weight = None
    hidden_states_215 = hidden_states_211 + down_proj_17;  hidden_states_211 = down_proj_17 = None
    hidden_states_216 = hidden_states_215.to(torch.float32)
    pow_73 = hidden_states_216.pow(2)
    variance_72 = pow_73.mean(-1, keepdim = True);  pow_73 = None
    add_144 = variance_72 + 1e-06;  variance_72 = None
    rsqrt_72 = torch.rsqrt(add_144);  add_144 = None
    hidden_states_217 = hidden_states_216 * rsqrt_72;  hidden_states_216 = rsqrt_72 = None
    l__self___model_model_layers_18_input_layernorm_weight = self.L__self___model_model_layers_18_input_layernorm_weight
    to_148 = hidden_states_217.to(torch.float16);  hidden_states_217 = None
    hidden_states_218 = l__self___model_model_layers_18_input_layernorm_weight * to_148;  l__self___model_model_layers_18_input_layernorm_weight = to_148 = None
    l__self___model_model_layers_18_self_attn_q_proj_weight = self.L__self___model_model_layers_18_self_attn_q_proj_weight
    linear_126 = torch._C._nn.linear(hidden_states_218, l__self___model_model_layers_18_self_attn_q_proj_weight, None);  l__self___model_model_layers_18_self_attn_q_proj_weight = None
    view_54 = linear_126.view((1, 2048, -1, 128));  linear_126 = None
    hidden_states_219 = view_54.to(torch.float32);  view_54 = None
    pow_74 = hidden_states_219.pow(2)
    variance_73 = pow_74.mean(-1, keepdim = True);  pow_74 = None
    add_145 = variance_73 + 1e-06;  variance_73 = None
    rsqrt_73 = torch.rsqrt(add_145);  add_145 = None
    hidden_states_220 = hidden_states_219 * rsqrt_73;  hidden_states_219 = rsqrt_73 = None
    l__self___model_model_layers_18_self_attn_q_norm_weight = self.L__self___model_model_layers_18_self_attn_q_norm_weight
    to_150 = hidden_states_220.to(torch.float16);  hidden_states_220 = None
    mul_239 = l__self___model_model_layers_18_self_attn_q_norm_weight * to_150;  l__self___model_model_layers_18_self_attn_q_norm_weight = to_150 = None
    query_states_18 = mul_239.transpose(1, 2);  mul_239 = None
    l__self___model_model_layers_18_self_attn_k_proj_weight = self.L__self___model_model_layers_18_self_attn_k_proj_weight
    linear_127 = torch._C._nn.linear(hidden_states_218, l__self___model_model_layers_18_self_attn_k_proj_weight, None);  l__self___model_model_layers_18_self_attn_k_proj_weight = None
    view_55 = linear_127.view((1, 2048, -1, 128));  linear_127 = None
    hidden_states_221 = view_55.to(torch.float32);  view_55 = None
    pow_75 = hidden_states_221.pow(2)
    variance_74 = pow_75.mean(-1, keepdim = True);  pow_75 = None
    add_146 = variance_74 + 1e-06;  variance_74 = None
    rsqrt_74 = torch.rsqrt(add_146);  add_146 = None
    hidden_states_222 = hidden_states_221 * rsqrt_74;  hidden_states_221 = rsqrt_74 = None
    l__self___model_model_layers_18_self_attn_k_norm_weight = self.L__self___model_model_layers_18_self_attn_k_norm_weight
    to_152 = hidden_states_222.to(torch.float16);  hidden_states_222 = None
    mul_241 = l__self___model_model_layers_18_self_attn_k_norm_weight * to_152;  l__self___model_model_layers_18_self_attn_k_norm_weight = to_152 = None
    key_states_18 = mul_241.transpose(1, 2);  mul_241 = None
    l__self___model_model_layers_18_self_attn_v_proj_weight = self.L__self___model_model_layers_18_self_attn_v_proj_weight
    linear_128 = torch._C._nn.linear(hidden_states_218, l__self___model_model_layers_18_self_attn_v_proj_weight, None);  hidden_states_218 = l__self___model_model_layers_18_self_attn_v_proj_weight = None
    view_56 = linear_128.view((1, 2048, -1, 128));  linear_128 = None
    value_states_18 = view_56.transpose(1, 2);  view_56 = None
    cos_21 = cos_2.unsqueeze(1)
    sin_21 = sin_2.unsqueeze(1)
    mul_242 = query_states_18 * cos_21
    x1_36 = query_states_18[(Ellipsis, slice(None, 64, None))]
    x2_36 = query_states_18[(Ellipsis, slice(64, None, None))];  query_states_18 = None
    neg_36 = -x2_36;  x2_36 = None
    cat_37 = torch.cat((neg_36, x1_36), dim = -1);  neg_36 = x1_36 = None
    mul_243 = cat_37 * sin_21;  cat_37 = None
    q_embed_18 = mul_242 + mul_243;  mul_242 = mul_243 = None
    mul_244 = key_states_18 * cos_21;  cos_21 = None
    x1_37 = key_states_18[(Ellipsis, slice(None, 64, None))]
    x2_37 = key_states_18[(Ellipsis, slice(64, None, None))];  key_states_18 = None
    neg_37 = -x2_37;  x2_37 = None
    cat_38 = torch.cat((neg_37, x1_37), dim = -1);  neg_37 = x1_37 = None
    mul_245 = cat_38 * sin_21;  cat_38 = sin_21 = None
    k_embed_18 = mul_244 + mul_245;  mul_244 = mul_245 = None
    q_18 = q_embed_18.contiguous();  q_embed_18 = None
    k_18 = k_embed_18.contiguous();  k_embed_18 = None
    v_18 = value_states_18.contiguous();  value_states_18 = None
    npu_fused_infer_attention_score_default_18 = torch.ops.npu.npu_fused_infer_attention_score.default(q_18, k_18, v_18, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_18 = npu_fused_infer_attention_score_default_18[0];  npu_fused_infer_attention_score_default_18 = None
    attn_output_54 = output_18.transpose(1, 2)
    reshape_18 = attn_output_54.reshape(1, 2048, -1);  attn_output_54 = None
    attn_output_55 = reshape_18.contiguous();  reshape_18 = None
    l__self___model_model_layers_18_self_attn_o_proj_weight = self.L__self___model_model_layers_18_self_attn_o_proj_weight
    attn_output_56 = torch._C._nn.linear(attn_output_55, l__self___model_model_layers_18_self_attn_o_proj_weight, None);  attn_output_55 = l__self___model_model_layers_18_self_attn_o_proj_weight = None
    hidden_states_223 = hidden_states_215 + attn_output_56;  hidden_states_215 = attn_output_56 = None
    hidden_states_224 = hidden_states_223.to(torch.float32)
    pow_76 = hidden_states_224.pow(2)
    variance_75 = pow_76.mean(-1, keepdim = True);  pow_76 = None
    add_150 = variance_75 + 1e-06;  variance_75 = None
    rsqrt_75 = torch.rsqrt(add_150);  add_150 = None
    hidden_states_225 = hidden_states_224 * rsqrt_75;  hidden_states_224 = rsqrt_75 = None
    l__self___model_model_layers_18_post_attention_layernorm_weight = self.L__self___model_model_layers_18_post_attention_layernorm_weight
    to_154 = hidden_states_225.to(torch.float16);  hidden_states_225 = None
    hidden_states_226 = l__self___model_model_layers_18_post_attention_layernorm_weight * to_154;  l__self___model_model_layers_18_post_attention_layernorm_weight = to_154 = None
    l__self___model_model_layers_18_mlp_gate_proj_weight = self.L__self___model_model_layers_18_mlp_gate_proj_weight
    linear_130 = torch._C._nn.linear(hidden_states_226, l__self___model_model_layers_18_mlp_gate_proj_weight, None);  l__self___model_model_layers_18_mlp_gate_proj_weight = None
    silu_18 = torch.nn.functional.silu(linear_130);  linear_130 = None
    l__self___model_model_layers_18_mlp_up_proj_weight = self.L__self___model_model_layers_18_mlp_up_proj_weight
    linear_131 = torch._C._nn.linear(hidden_states_226, l__self___model_model_layers_18_mlp_up_proj_weight, None);  hidden_states_226 = l__self___model_model_layers_18_mlp_up_proj_weight = None
    mul_248 = silu_18 * linear_131;  silu_18 = linear_131 = None
    l__self___model_model_layers_18_mlp_down_proj_weight = self.L__self___model_model_layers_18_mlp_down_proj_weight
    down_proj_18 = torch._C._nn.linear(mul_248, l__self___model_model_layers_18_mlp_down_proj_weight, None);  mul_248 = l__self___model_model_layers_18_mlp_down_proj_weight = None
    hidden_states_227 = hidden_states_223 + down_proj_18;  hidden_states_223 = down_proj_18 = None
    hidden_states_228 = hidden_states_227.to(torch.float32)
    pow_77 = hidden_states_228.pow(2)
    variance_76 = pow_77.mean(-1, keepdim = True);  pow_77 = None
    add_152 = variance_76 + 1e-06;  variance_76 = None
    rsqrt_76 = torch.rsqrt(add_152);  add_152 = None
    hidden_states_229 = hidden_states_228 * rsqrt_76;  hidden_states_228 = rsqrt_76 = None
    l__self___model_model_layers_19_input_layernorm_weight = self.L__self___model_model_layers_19_input_layernorm_weight
    to_156 = hidden_states_229.to(torch.float16);  hidden_states_229 = None
    hidden_states_230 = l__self___model_model_layers_19_input_layernorm_weight * to_156;  l__self___model_model_layers_19_input_layernorm_weight = to_156 = None
    l__self___model_model_layers_19_self_attn_q_proj_weight = self.L__self___model_model_layers_19_self_attn_q_proj_weight
    linear_133 = torch._C._nn.linear(hidden_states_230, l__self___model_model_layers_19_self_attn_q_proj_weight, None);  l__self___model_model_layers_19_self_attn_q_proj_weight = None
    view_57 = linear_133.view((1, 2048, -1, 128));  linear_133 = None
    hidden_states_231 = view_57.to(torch.float32);  view_57 = None
    pow_78 = hidden_states_231.pow(2)
    variance_77 = pow_78.mean(-1, keepdim = True);  pow_78 = None
    add_153 = variance_77 + 1e-06;  variance_77 = None
    rsqrt_77 = torch.rsqrt(add_153);  add_153 = None
    hidden_states_232 = hidden_states_231 * rsqrt_77;  hidden_states_231 = rsqrt_77 = None
    l__self___model_model_layers_19_self_attn_q_norm_weight = self.L__self___model_model_layers_19_self_attn_q_norm_weight
    to_158 = hidden_states_232.to(torch.float16);  hidden_states_232 = None
    mul_252 = l__self___model_model_layers_19_self_attn_q_norm_weight * to_158;  l__self___model_model_layers_19_self_attn_q_norm_weight = to_158 = None
    query_states_19 = mul_252.transpose(1, 2);  mul_252 = None
    l__self___model_model_layers_19_self_attn_k_proj_weight = self.L__self___model_model_layers_19_self_attn_k_proj_weight
    linear_134 = torch._C._nn.linear(hidden_states_230, l__self___model_model_layers_19_self_attn_k_proj_weight, None);  l__self___model_model_layers_19_self_attn_k_proj_weight = None
    view_58 = linear_134.view((1, 2048, -1, 128));  linear_134 = None
    hidden_states_233 = view_58.to(torch.float32);  view_58 = None
    pow_79 = hidden_states_233.pow(2)
    variance_78 = pow_79.mean(-1, keepdim = True);  pow_79 = None
    add_154 = variance_78 + 1e-06;  variance_78 = None
    rsqrt_78 = torch.rsqrt(add_154);  add_154 = None
    hidden_states_234 = hidden_states_233 * rsqrt_78;  hidden_states_233 = rsqrt_78 = None
    l__self___model_model_layers_19_self_attn_k_norm_weight = self.L__self___model_model_layers_19_self_attn_k_norm_weight
    to_160 = hidden_states_234.to(torch.float16);  hidden_states_234 = None
    mul_254 = l__self___model_model_layers_19_self_attn_k_norm_weight * to_160;  l__self___model_model_layers_19_self_attn_k_norm_weight = to_160 = None
    key_states_19 = mul_254.transpose(1, 2);  mul_254 = None
    l__self___model_model_layers_19_self_attn_v_proj_weight = self.L__self___model_model_layers_19_self_attn_v_proj_weight
    linear_135 = torch._C._nn.linear(hidden_states_230, l__self___model_model_layers_19_self_attn_v_proj_weight, None);  hidden_states_230 = l__self___model_model_layers_19_self_attn_v_proj_weight = None
    view_59 = linear_135.view((1, 2048, -1, 128));  linear_135 = None
    value_states_19 = view_59.transpose(1, 2);  view_59 = None
    cos_22 = cos_2.unsqueeze(1)
    sin_22 = sin_2.unsqueeze(1)
    mul_255 = query_states_19 * cos_22
    x1_38 = query_states_19[(Ellipsis, slice(None, 64, None))]
    x2_38 = query_states_19[(Ellipsis, slice(64, None, None))];  query_states_19 = None
    neg_38 = -x2_38;  x2_38 = None
    cat_39 = torch.cat((neg_38, x1_38), dim = -1);  neg_38 = x1_38 = None
    mul_256 = cat_39 * sin_22;  cat_39 = None
    q_embed_19 = mul_255 + mul_256;  mul_255 = mul_256 = None
    mul_257 = key_states_19 * cos_22;  cos_22 = None
    x1_39 = key_states_19[(Ellipsis, slice(None, 64, None))]
    x2_39 = key_states_19[(Ellipsis, slice(64, None, None))];  key_states_19 = None
    neg_39 = -x2_39;  x2_39 = None
    cat_40 = torch.cat((neg_39, x1_39), dim = -1);  neg_39 = x1_39 = None
    mul_258 = cat_40 * sin_22;  cat_40 = sin_22 = None
    k_embed_19 = mul_257 + mul_258;  mul_257 = mul_258 = None
    q_19 = q_embed_19.contiguous();  q_embed_19 = None
    k_19 = k_embed_19.contiguous();  k_embed_19 = None
    v_19 = value_states_19.contiguous();  value_states_19 = None
    npu_fused_infer_attention_score_default_19 = torch.ops.npu.npu_fused_infer_attention_score.default(q_19, k_19, v_19, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_19 = npu_fused_infer_attention_score_default_19[0];  npu_fused_infer_attention_score_default_19 = None
    attn_output_57 = output_19.transpose(1, 2)
    reshape_19 = attn_output_57.reshape(1, 2048, -1);  attn_output_57 = None
    attn_output_58 = reshape_19.contiguous();  reshape_19 = None
    l__self___model_model_layers_19_self_attn_o_proj_weight = self.L__self___model_model_layers_19_self_attn_o_proj_weight
    attn_output_59 = torch._C._nn.linear(attn_output_58, l__self___model_model_layers_19_self_attn_o_proj_weight, None);  attn_output_58 = l__self___model_model_layers_19_self_attn_o_proj_weight = None
    hidden_states_235 = hidden_states_227 + attn_output_59;  hidden_states_227 = attn_output_59 = None
    hidden_states_236 = hidden_states_235.to(torch.float32)
    pow_80 = hidden_states_236.pow(2)
    variance_79 = pow_80.mean(-1, keepdim = True);  pow_80 = None
    add_158 = variance_79 + 1e-06;  variance_79 = None
    rsqrt_79 = torch.rsqrt(add_158);  add_158 = None
    hidden_states_237 = hidden_states_236 * rsqrt_79;  hidden_states_236 = rsqrt_79 = None
    l__self___model_model_layers_19_post_attention_layernorm_weight = self.L__self___model_model_layers_19_post_attention_layernorm_weight
    to_162 = hidden_states_237.to(torch.float16);  hidden_states_237 = None
    hidden_states_238 = l__self___model_model_layers_19_post_attention_layernorm_weight * to_162;  l__self___model_model_layers_19_post_attention_layernorm_weight = to_162 = None
    l__self___model_model_layers_19_mlp_gate_proj_weight = self.L__self___model_model_layers_19_mlp_gate_proj_weight
    linear_137 = torch._C._nn.linear(hidden_states_238, l__self___model_model_layers_19_mlp_gate_proj_weight, None);  l__self___model_model_layers_19_mlp_gate_proj_weight = None
    silu_19 = torch.nn.functional.silu(linear_137);  linear_137 = None
    l__self___model_model_layers_19_mlp_up_proj_weight = self.L__self___model_model_layers_19_mlp_up_proj_weight
    linear_138 = torch._C._nn.linear(hidden_states_238, l__self___model_model_layers_19_mlp_up_proj_weight, None);  hidden_states_238 = l__self___model_model_layers_19_mlp_up_proj_weight = None
    mul_261 = silu_19 * linear_138;  silu_19 = linear_138 = None
    l__self___model_model_layers_19_mlp_down_proj_weight = self.L__self___model_model_layers_19_mlp_down_proj_weight
    down_proj_19 = torch._C._nn.linear(mul_261, l__self___model_model_layers_19_mlp_down_proj_weight, None);  mul_261 = l__self___model_model_layers_19_mlp_down_proj_weight = None
    hidden_states_239 = hidden_states_235 + down_proj_19;  hidden_states_235 = down_proj_19 = None
    hidden_states_240 = hidden_states_239.to(torch.float32)
    pow_81 = hidden_states_240.pow(2)
    variance_80 = pow_81.mean(-1, keepdim = True);  pow_81 = None
    add_160 = variance_80 + 1e-06;  variance_80 = None
    rsqrt_80 = torch.rsqrt(add_160);  add_160 = None
    hidden_states_241 = hidden_states_240 * rsqrt_80;  hidden_states_240 = rsqrt_80 = None
    l__self___model_model_layers_20_input_layernorm_weight = self.L__self___model_model_layers_20_input_layernorm_weight
    to_164 = hidden_states_241.to(torch.float16);  hidden_states_241 = None
    hidden_states_242 = l__self___model_model_layers_20_input_layernorm_weight * to_164;  l__self___model_model_layers_20_input_layernorm_weight = to_164 = None
    l__self___model_model_layers_20_self_attn_q_proj_weight = self.L__self___model_model_layers_20_self_attn_q_proj_weight
    linear_140 = torch._C._nn.linear(hidden_states_242, l__self___model_model_layers_20_self_attn_q_proj_weight, None);  l__self___model_model_layers_20_self_attn_q_proj_weight = None
    view_60 = linear_140.view((1, 2048, -1, 128));  linear_140 = None
    hidden_states_243 = view_60.to(torch.float32);  view_60 = None
    pow_82 = hidden_states_243.pow(2)
    variance_81 = pow_82.mean(-1, keepdim = True);  pow_82 = None
    add_161 = variance_81 + 1e-06;  variance_81 = None
    rsqrt_81 = torch.rsqrt(add_161);  add_161 = None
    hidden_states_244 = hidden_states_243 * rsqrt_81;  hidden_states_243 = rsqrt_81 = None
    l__self___model_model_layers_20_self_attn_q_norm_weight = self.L__self___model_model_layers_20_self_attn_q_norm_weight
    to_166 = hidden_states_244.to(torch.float16);  hidden_states_244 = None
    mul_265 = l__self___model_model_layers_20_self_attn_q_norm_weight * to_166;  l__self___model_model_layers_20_self_attn_q_norm_weight = to_166 = None
    query_states_20 = mul_265.transpose(1, 2);  mul_265 = None
    l__self___model_model_layers_20_self_attn_k_proj_weight = self.L__self___model_model_layers_20_self_attn_k_proj_weight
    linear_141 = torch._C._nn.linear(hidden_states_242, l__self___model_model_layers_20_self_attn_k_proj_weight, None);  l__self___model_model_layers_20_self_attn_k_proj_weight = None
    view_61 = linear_141.view((1, 2048, -1, 128));  linear_141 = None
    hidden_states_245 = view_61.to(torch.float32);  view_61 = None
    pow_83 = hidden_states_245.pow(2)
    variance_82 = pow_83.mean(-1, keepdim = True);  pow_83 = None
    add_162 = variance_82 + 1e-06;  variance_82 = None
    rsqrt_82 = torch.rsqrt(add_162);  add_162 = None
    hidden_states_246 = hidden_states_245 * rsqrt_82;  hidden_states_245 = rsqrt_82 = None
    l__self___model_model_layers_20_self_attn_k_norm_weight = self.L__self___model_model_layers_20_self_attn_k_norm_weight
    to_168 = hidden_states_246.to(torch.float16);  hidden_states_246 = None
    mul_267 = l__self___model_model_layers_20_self_attn_k_norm_weight * to_168;  l__self___model_model_layers_20_self_attn_k_norm_weight = to_168 = None
    key_states_20 = mul_267.transpose(1, 2);  mul_267 = None
    l__self___model_model_layers_20_self_attn_v_proj_weight = self.L__self___model_model_layers_20_self_attn_v_proj_weight
    linear_142 = torch._C._nn.linear(hidden_states_242, l__self___model_model_layers_20_self_attn_v_proj_weight, None);  hidden_states_242 = l__self___model_model_layers_20_self_attn_v_proj_weight = None
    view_62 = linear_142.view((1, 2048, -1, 128));  linear_142 = None
    value_states_20 = view_62.transpose(1, 2);  view_62 = None
    cos_23 = cos_2.unsqueeze(1)
    sin_23 = sin_2.unsqueeze(1)
    mul_268 = query_states_20 * cos_23
    x1_40 = query_states_20[(Ellipsis, slice(None, 64, None))]
    x2_40 = query_states_20[(Ellipsis, slice(64, None, None))];  query_states_20 = None
    neg_40 = -x2_40;  x2_40 = None
    cat_41 = torch.cat((neg_40, x1_40), dim = -1);  neg_40 = x1_40 = None
    mul_269 = cat_41 * sin_23;  cat_41 = None
    q_embed_20 = mul_268 + mul_269;  mul_268 = mul_269 = None
    mul_270 = key_states_20 * cos_23;  cos_23 = None
    x1_41 = key_states_20[(Ellipsis, slice(None, 64, None))]
    x2_41 = key_states_20[(Ellipsis, slice(64, None, None))];  key_states_20 = None
    neg_41 = -x2_41;  x2_41 = None
    cat_42 = torch.cat((neg_41, x1_41), dim = -1);  neg_41 = x1_41 = None
    mul_271 = cat_42 * sin_23;  cat_42 = sin_23 = None
    k_embed_20 = mul_270 + mul_271;  mul_270 = mul_271 = None
    q_20 = q_embed_20.contiguous();  q_embed_20 = None
    k_20 = k_embed_20.contiguous();  k_embed_20 = None
    v_20 = value_states_20.contiguous();  value_states_20 = None
    npu_fused_infer_attention_score_default_20 = torch.ops.npu.npu_fused_infer_attention_score.default(q_20, k_20, v_20, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_20 = npu_fused_infer_attention_score_default_20[0];  npu_fused_infer_attention_score_default_20 = None
    attn_output_60 = output_20.transpose(1, 2)
    reshape_20 = attn_output_60.reshape(1, 2048, -1);  attn_output_60 = None
    attn_output_61 = reshape_20.contiguous();  reshape_20 = None
    l__self___model_model_layers_20_self_attn_o_proj_weight = self.L__self___model_model_layers_20_self_attn_o_proj_weight
    attn_output_62 = torch._C._nn.linear(attn_output_61, l__self___model_model_layers_20_self_attn_o_proj_weight, None);  attn_output_61 = l__self___model_model_layers_20_self_attn_o_proj_weight = None
    hidden_states_247 = hidden_states_239 + attn_output_62;  hidden_states_239 = attn_output_62 = None
    hidden_states_248 = hidden_states_247.to(torch.float32)
    pow_84 = hidden_states_248.pow(2)
    variance_83 = pow_84.mean(-1, keepdim = True);  pow_84 = None
    add_166 = variance_83 + 1e-06;  variance_83 = None
    rsqrt_83 = torch.rsqrt(add_166);  add_166 = None
    hidden_states_249 = hidden_states_248 * rsqrt_83;  hidden_states_248 = rsqrt_83 = None
    l__self___model_model_layers_20_post_attention_layernorm_weight = self.L__self___model_model_layers_20_post_attention_layernorm_weight
    to_170 = hidden_states_249.to(torch.float16);  hidden_states_249 = None
    hidden_states_250 = l__self___model_model_layers_20_post_attention_layernorm_weight * to_170;  l__self___model_model_layers_20_post_attention_layernorm_weight = to_170 = None
    l__self___model_model_layers_20_mlp_gate_proj_weight = self.L__self___model_model_layers_20_mlp_gate_proj_weight
    linear_144 = torch._C._nn.linear(hidden_states_250, l__self___model_model_layers_20_mlp_gate_proj_weight, None);  l__self___model_model_layers_20_mlp_gate_proj_weight = None
    silu_20 = torch.nn.functional.silu(linear_144);  linear_144 = None
    l__self___model_model_layers_20_mlp_up_proj_weight = self.L__self___model_model_layers_20_mlp_up_proj_weight
    linear_145 = torch._C._nn.linear(hidden_states_250, l__self___model_model_layers_20_mlp_up_proj_weight, None);  hidden_states_250 = l__self___model_model_layers_20_mlp_up_proj_weight = None
    mul_274 = silu_20 * linear_145;  silu_20 = linear_145 = None
    l__self___model_model_layers_20_mlp_down_proj_weight = self.L__self___model_model_layers_20_mlp_down_proj_weight
    down_proj_20 = torch._C._nn.linear(mul_274, l__self___model_model_layers_20_mlp_down_proj_weight, None);  mul_274 = l__self___model_model_layers_20_mlp_down_proj_weight = None
    hidden_states_251 = hidden_states_247 + down_proj_20;  hidden_states_247 = down_proj_20 = None
    hidden_states_252 = hidden_states_251.to(torch.float32)
    pow_85 = hidden_states_252.pow(2)
    variance_84 = pow_85.mean(-1, keepdim = True);  pow_85 = None
    add_168 = variance_84 + 1e-06;  variance_84 = None
    rsqrt_84 = torch.rsqrt(add_168);  add_168 = None
    hidden_states_253 = hidden_states_252 * rsqrt_84;  hidden_states_252 = rsqrt_84 = None
    l__self___model_model_layers_21_input_layernorm_weight = self.L__self___model_model_layers_21_input_layernorm_weight
    to_172 = hidden_states_253.to(torch.float16);  hidden_states_253 = None
    hidden_states_254 = l__self___model_model_layers_21_input_layernorm_weight * to_172;  l__self___model_model_layers_21_input_layernorm_weight = to_172 = None
    l__self___model_model_layers_21_self_attn_q_proj_weight = self.L__self___model_model_layers_21_self_attn_q_proj_weight
    linear_147 = torch._C._nn.linear(hidden_states_254, l__self___model_model_layers_21_self_attn_q_proj_weight, None);  l__self___model_model_layers_21_self_attn_q_proj_weight = None
    view_63 = linear_147.view((1, 2048, -1, 128));  linear_147 = None
    hidden_states_255 = view_63.to(torch.float32);  view_63 = None
    pow_86 = hidden_states_255.pow(2)
    variance_85 = pow_86.mean(-1, keepdim = True);  pow_86 = None
    add_169 = variance_85 + 1e-06;  variance_85 = None
    rsqrt_85 = torch.rsqrt(add_169);  add_169 = None
    hidden_states_256 = hidden_states_255 * rsqrt_85;  hidden_states_255 = rsqrt_85 = None
    l__self___model_model_layers_21_self_attn_q_norm_weight = self.L__self___model_model_layers_21_self_attn_q_norm_weight
    to_174 = hidden_states_256.to(torch.float16);  hidden_states_256 = None
    mul_278 = l__self___model_model_layers_21_self_attn_q_norm_weight * to_174;  l__self___model_model_layers_21_self_attn_q_norm_weight = to_174 = None
    query_states_21 = mul_278.transpose(1, 2);  mul_278 = None
    l__self___model_model_layers_21_self_attn_k_proj_weight = self.L__self___model_model_layers_21_self_attn_k_proj_weight
    linear_148 = torch._C._nn.linear(hidden_states_254, l__self___model_model_layers_21_self_attn_k_proj_weight, None);  l__self___model_model_layers_21_self_attn_k_proj_weight = None
    view_64 = linear_148.view((1, 2048, -1, 128));  linear_148 = None
    hidden_states_257 = view_64.to(torch.float32);  view_64 = None
    pow_87 = hidden_states_257.pow(2)
    variance_86 = pow_87.mean(-1, keepdim = True);  pow_87 = None
    add_170 = variance_86 + 1e-06;  variance_86 = None
    rsqrt_86 = torch.rsqrt(add_170);  add_170 = None
    hidden_states_258 = hidden_states_257 * rsqrt_86;  hidden_states_257 = rsqrt_86 = None
    l__self___model_model_layers_21_self_attn_k_norm_weight = self.L__self___model_model_layers_21_self_attn_k_norm_weight
    to_176 = hidden_states_258.to(torch.float16);  hidden_states_258 = None
    mul_280 = l__self___model_model_layers_21_self_attn_k_norm_weight * to_176;  l__self___model_model_layers_21_self_attn_k_norm_weight = to_176 = None
    key_states_21 = mul_280.transpose(1, 2);  mul_280 = None
    l__self___model_model_layers_21_self_attn_v_proj_weight = self.L__self___model_model_layers_21_self_attn_v_proj_weight
    linear_149 = torch._C._nn.linear(hidden_states_254, l__self___model_model_layers_21_self_attn_v_proj_weight, None);  hidden_states_254 = l__self___model_model_layers_21_self_attn_v_proj_weight = None
    view_65 = linear_149.view((1, 2048, -1, 128));  linear_149 = None
    value_states_21 = view_65.transpose(1, 2);  view_65 = None
    cos_24 = cos_2.unsqueeze(1)
    sin_24 = sin_2.unsqueeze(1)
    mul_281 = query_states_21 * cos_24
    x1_42 = query_states_21[(Ellipsis, slice(None, 64, None))]
    x2_42 = query_states_21[(Ellipsis, slice(64, None, None))];  query_states_21 = None
    neg_42 = -x2_42;  x2_42 = None
    cat_43 = torch.cat((neg_42, x1_42), dim = -1);  neg_42 = x1_42 = None
    mul_282 = cat_43 * sin_24;  cat_43 = None
    q_embed_21 = mul_281 + mul_282;  mul_281 = mul_282 = None
    mul_283 = key_states_21 * cos_24;  cos_24 = None
    x1_43 = key_states_21[(Ellipsis, slice(None, 64, None))]
    x2_43 = key_states_21[(Ellipsis, slice(64, None, None))];  key_states_21 = None
    neg_43 = -x2_43;  x2_43 = None
    cat_44 = torch.cat((neg_43, x1_43), dim = -1);  neg_43 = x1_43 = None
    mul_284 = cat_44 * sin_24;  cat_44 = sin_24 = None
    k_embed_21 = mul_283 + mul_284;  mul_283 = mul_284 = None
    q_21 = q_embed_21.contiguous();  q_embed_21 = None
    k_21 = k_embed_21.contiguous();  k_embed_21 = None
    v_21 = value_states_21.contiguous();  value_states_21 = None
    npu_fused_infer_attention_score_default_21 = torch.ops.npu.npu_fused_infer_attention_score.default(q_21, k_21, v_21, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_21 = npu_fused_infer_attention_score_default_21[0];  npu_fused_infer_attention_score_default_21 = None
    attn_output_63 = output_21.transpose(1, 2)
    reshape_21 = attn_output_63.reshape(1, 2048, -1);  attn_output_63 = None
    attn_output_64 = reshape_21.contiguous();  reshape_21 = None
    l__self___model_model_layers_21_self_attn_o_proj_weight = self.L__self___model_model_layers_21_self_attn_o_proj_weight
    attn_output_65 = torch._C._nn.linear(attn_output_64, l__self___model_model_layers_21_self_attn_o_proj_weight, None);  attn_output_64 = l__self___model_model_layers_21_self_attn_o_proj_weight = None
    hidden_states_259 = hidden_states_251 + attn_output_65;  hidden_states_251 = attn_output_65 = None
    hidden_states_260 = hidden_states_259.to(torch.float32)
    pow_88 = hidden_states_260.pow(2)
    variance_87 = pow_88.mean(-1, keepdim = True);  pow_88 = None
    add_174 = variance_87 + 1e-06;  variance_87 = None
    rsqrt_87 = torch.rsqrt(add_174);  add_174 = None
    hidden_states_261 = hidden_states_260 * rsqrt_87;  hidden_states_260 = rsqrt_87 = None
    l__self___model_model_layers_21_post_attention_layernorm_weight = self.L__self___model_model_layers_21_post_attention_layernorm_weight
    to_178 = hidden_states_261.to(torch.float16);  hidden_states_261 = None
    hidden_states_262 = l__self___model_model_layers_21_post_attention_layernorm_weight * to_178;  l__self___model_model_layers_21_post_attention_layernorm_weight = to_178 = None
    l__self___model_model_layers_21_mlp_gate_proj_weight = self.L__self___model_model_layers_21_mlp_gate_proj_weight
    linear_151 = torch._C._nn.linear(hidden_states_262, l__self___model_model_layers_21_mlp_gate_proj_weight, None);  l__self___model_model_layers_21_mlp_gate_proj_weight = None
    silu_21 = torch.nn.functional.silu(linear_151);  linear_151 = None
    l__self___model_model_layers_21_mlp_up_proj_weight = self.L__self___model_model_layers_21_mlp_up_proj_weight
    linear_152 = torch._C._nn.linear(hidden_states_262, l__self___model_model_layers_21_mlp_up_proj_weight, None);  hidden_states_262 = l__self___model_model_layers_21_mlp_up_proj_weight = None
    mul_287 = silu_21 * linear_152;  silu_21 = linear_152 = None
    l__self___model_model_layers_21_mlp_down_proj_weight = self.L__self___model_model_layers_21_mlp_down_proj_weight
    down_proj_21 = torch._C._nn.linear(mul_287, l__self___model_model_layers_21_mlp_down_proj_weight, None);  mul_287 = l__self___model_model_layers_21_mlp_down_proj_weight = None
    hidden_states_263 = hidden_states_259 + down_proj_21;  hidden_states_259 = down_proj_21 = None
    hidden_states_264 = hidden_states_263.to(torch.float32)
    pow_89 = hidden_states_264.pow(2)
    variance_88 = pow_89.mean(-1, keepdim = True);  pow_89 = None
    add_176 = variance_88 + 1e-06;  variance_88 = None
    rsqrt_88 = torch.rsqrt(add_176);  add_176 = None
    hidden_states_265 = hidden_states_264 * rsqrt_88;  hidden_states_264 = rsqrt_88 = None
    l__self___model_model_layers_22_input_layernorm_weight = self.L__self___model_model_layers_22_input_layernorm_weight
    to_180 = hidden_states_265.to(torch.float16);  hidden_states_265 = None
    hidden_states_266 = l__self___model_model_layers_22_input_layernorm_weight * to_180;  l__self___model_model_layers_22_input_layernorm_weight = to_180 = None
    l__self___model_model_layers_22_self_attn_q_proj_weight = self.L__self___model_model_layers_22_self_attn_q_proj_weight
    linear_154 = torch._C._nn.linear(hidden_states_266, l__self___model_model_layers_22_self_attn_q_proj_weight, None);  l__self___model_model_layers_22_self_attn_q_proj_weight = None
    view_66 = linear_154.view((1, 2048, -1, 128));  linear_154 = None
    hidden_states_267 = view_66.to(torch.float32);  view_66 = None
    pow_90 = hidden_states_267.pow(2)
    variance_89 = pow_90.mean(-1, keepdim = True);  pow_90 = None
    add_177 = variance_89 + 1e-06;  variance_89 = None
    rsqrt_89 = torch.rsqrt(add_177);  add_177 = None
    hidden_states_268 = hidden_states_267 * rsqrt_89;  hidden_states_267 = rsqrt_89 = None
    l__self___model_model_layers_22_self_attn_q_norm_weight = self.L__self___model_model_layers_22_self_attn_q_norm_weight
    to_182 = hidden_states_268.to(torch.float16);  hidden_states_268 = None
    mul_291 = l__self___model_model_layers_22_self_attn_q_norm_weight * to_182;  l__self___model_model_layers_22_self_attn_q_norm_weight = to_182 = None
    query_states_22 = mul_291.transpose(1, 2);  mul_291 = None
    l__self___model_model_layers_22_self_attn_k_proj_weight = self.L__self___model_model_layers_22_self_attn_k_proj_weight
    linear_155 = torch._C._nn.linear(hidden_states_266, l__self___model_model_layers_22_self_attn_k_proj_weight, None);  l__self___model_model_layers_22_self_attn_k_proj_weight = None
    view_67 = linear_155.view((1, 2048, -1, 128));  linear_155 = None
    hidden_states_269 = view_67.to(torch.float32);  view_67 = None
    pow_91 = hidden_states_269.pow(2)
    variance_90 = pow_91.mean(-1, keepdim = True);  pow_91 = None
    add_178 = variance_90 + 1e-06;  variance_90 = None
    rsqrt_90 = torch.rsqrt(add_178);  add_178 = None
    hidden_states_270 = hidden_states_269 * rsqrt_90;  hidden_states_269 = rsqrt_90 = None
    l__self___model_model_layers_22_self_attn_k_norm_weight = self.L__self___model_model_layers_22_self_attn_k_norm_weight
    to_184 = hidden_states_270.to(torch.float16);  hidden_states_270 = None
    mul_293 = l__self___model_model_layers_22_self_attn_k_norm_weight * to_184;  l__self___model_model_layers_22_self_attn_k_norm_weight = to_184 = None
    key_states_22 = mul_293.transpose(1, 2);  mul_293 = None
    l__self___model_model_layers_22_self_attn_v_proj_weight = self.L__self___model_model_layers_22_self_attn_v_proj_weight
    linear_156 = torch._C._nn.linear(hidden_states_266, l__self___model_model_layers_22_self_attn_v_proj_weight, None);  hidden_states_266 = l__self___model_model_layers_22_self_attn_v_proj_weight = None
    view_68 = linear_156.view((1, 2048, -1, 128));  linear_156 = None
    value_states_22 = view_68.transpose(1, 2);  view_68 = None
    cos_25 = cos_2.unsqueeze(1)
    sin_25 = sin_2.unsqueeze(1)
    mul_294 = query_states_22 * cos_25
    x1_44 = query_states_22[(Ellipsis, slice(None, 64, None))]
    x2_44 = query_states_22[(Ellipsis, slice(64, None, None))];  query_states_22 = None
    neg_44 = -x2_44;  x2_44 = None
    cat_45 = torch.cat((neg_44, x1_44), dim = -1);  neg_44 = x1_44 = None
    mul_295 = cat_45 * sin_25;  cat_45 = None
    q_embed_22 = mul_294 + mul_295;  mul_294 = mul_295 = None
    mul_296 = key_states_22 * cos_25;  cos_25 = None
    x1_45 = key_states_22[(Ellipsis, slice(None, 64, None))]
    x2_45 = key_states_22[(Ellipsis, slice(64, None, None))];  key_states_22 = None
    neg_45 = -x2_45;  x2_45 = None
    cat_46 = torch.cat((neg_45, x1_45), dim = -1);  neg_45 = x1_45 = None
    mul_297 = cat_46 * sin_25;  cat_46 = sin_25 = None
    k_embed_22 = mul_296 + mul_297;  mul_296 = mul_297 = None
    q_22 = q_embed_22.contiguous();  q_embed_22 = None
    k_22 = k_embed_22.contiguous();  k_embed_22 = None
    v_22 = value_states_22.contiguous();  value_states_22 = None
    npu_fused_infer_attention_score_default_22 = torch.ops.npu.npu_fused_infer_attention_score.default(q_22, k_22, v_22, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_22 = npu_fused_infer_attention_score_default_22[0];  npu_fused_infer_attention_score_default_22 = None
    attn_output_66 = output_22.transpose(1, 2)
    reshape_22 = attn_output_66.reshape(1, 2048, -1);  attn_output_66 = None
    attn_output_67 = reshape_22.contiguous();  reshape_22 = None
    l__self___model_model_layers_22_self_attn_o_proj_weight = self.L__self___model_model_layers_22_self_attn_o_proj_weight
    attn_output_68 = torch._C._nn.linear(attn_output_67, l__self___model_model_layers_22_self_attn_o_proj_weight, None);  attn_output_67 = l__self___model_model_layers_22_self_attn_o_proj_weight = None
    hidden_states_271 = hidden_states_263 + attn_output_68;  hidden_states_263 = attn_output_68 = None
    hidden_states_272 = hidden_states_271.to(torch.float32)
    pow_92 = hidden_states_272.pow(2)
    variance_91 = pow_92.mean(-1, keepdim = True);  pow_92 = None
    add_182 = variance_91 + 1e-06;  variance_91 = None
    rsqrt_91 = torch.rsqrt(add_182);  add_182 = None
    hidden_states_273 = hidden_states_272 * rsqrt_91;  hidden_states_272 = rsqrt_91 = None
    l__self___model_model_layers_22_post_attention_layernorm_weight = self.L__self___model_model_layers_22_post_attention_layernorm_weight
    to_186 = hidden_states_273.to(torch.float16);  hidden_states_273 = None
    hidden_states_274 = l__self___model_model_layers_22_post_attention_layernorm_weight * to_186;  l__self___model_model_layers_22_post_attention_layernorm_weight = to_186 = None
    l__self___model_model_layers_22_mlp_gate_proj_weight = self.L__self___model_model_layers_22_mlp_gate_proj_weight
    linear_158 = torch._C._nn.linear(hidden_states_274, l__self___model_model_layers_22_mlp_gate_proj_weight, None);  l__self___model_model_layers_22_mlp_gate_proj_weight = None
    silu_22 = torch.nn.functional.silu(linear_158);  linear_158 = None
    l__self___model_model_layers_22_mlp_up_proj_weight = self.L__self___model_model_layers_22_mlp_up_proj_weight
    linear_159 = torch._C._nn.linear(hidden_states_274, l__self___model_model_layers_22_mlp_up_proj_weight, None);  hidden_states_274 = l__self___model_model_layers_22_mlp_up_proj_weight = None
    mul_300 = silu_22 * linear_159;  silu_22 = linear_159 = None
    l__self___model_model_layers_22_mlp_down_proj_weight = self.L__self___model_model_layers_22_mlp_down_proj_weight
    down_proj_22 = torch._C._nn.linear(mul_300, l__self___model_model_layers_22_mlp_down_proj_weight, None);  mul_300 = l__self___model_model_layers_22_mlp_down_proj_weight = None
    hidden_states_275 = hidden_states_271 + down_proj_22;  hidden_states_271 = down_proj_22 = None
    hidden_states_276 = hidden_states_275.to(torch.float32)
    pow_93 = hidden_states_276.pow(2)
    variance_92 = pow_93.mean(-1, keepdim = True);  pow_93 = None
    add_184 = variance_92 + 1e-06;  variance_92 = None
    rsqrt_92 = torch.rsqrt(add_184);  add_184 = None
    hidden_states_277 = hidden_states_276 * rsqrt_92;  hidden_states_276 = rsqrt_92 = None
    l__self___model_model_layers_23_input_layernorm_weight = self.L__self___model_model_layers_23_input_layernorm_weight
    to_188 = hidden_states_277.to(torch.float16);  hidden_states_277 = None
    hidden_states_278 = l__self___model_model_layers_23_input_layernorm_weight * to_188;  l__self___model_model_layers_23_input_layernorm_weight = to_188 = None
    l__self___model_model_layers_23_self_attn_q_proj_weight = self.L__self___model_model_layers_23_self_attn_q_proj_weight
    linear_161 = torch._C._nn.linear(hidden_states_278, l__self___model_model_layers_23_self_attn_q_proj_weight, None);  l__self___model_model_layers_23_self_attn_q_proj_weight = None
    view_69 = linear_161.view((1, 2048, -1, 128));  linear_161 = None
    hidden_states_279 = view_69.to(torch.float32);  view_69 = None
    pow_94 = hidden_states_279.pow(2)
    variance_93 = pow_94.mean(-1, keepdim = True);  pow_94 = None
    add_185 = variance_93 + 1e-06;  variance_93 = None
    rsqrt_93 = torch.rsqrt(add_185);  add_185 = None
    hidden_states_280 = hidden_states_279 * rsqrt_93;  hidden_states_279 = rsqrt_93 = None
    l__self___model_model_layers_23_self_attn_q_norm_weight = self.L__self___model_model_layers_23_self_attn_q_norm_weight
    to_190 = hidden_states_280.to(torch.float16);  hidden_states_280 = None
    mul_304 = l__self___model_model_layers_23_self_attn_q_norm_weight * to_190;  l__self___model_model_layers_23_self_attn_q_norm_weight = to_190 = None
    query_states_23 = mul_304.transpose(1, 2);  mul_304 = None
    l__self___model_model_layers_23_self_attn_k_proj_weight = self.L__self___model_model_layers_23_self_attn_k_proj_weight
    linear_162 = torch._C._nn.linear(hidden_states_278, l__self___model_model_layers_23_self_attn_k_proj_weight, None);  l__self___model_model_layers_23_self_attn_k_proj_weight = None
    view_70 = linear_162.view((1, 2048, -1, 128));  linear_162 = None
    hidden_states_281 = view_70.to(torch.float32);  view_70 = None
    pow_95 = hidden_states_281.pow(2)
    variance_94 = pow_95.mean(-1, keepdim = True);  pow_95 = None
    add_186 = variance_94 + 1e-06;  variance_94 = None
    rsqrt_94 = torch.rsqrt(add_186);  add_186 = None
    hidden_states_282 = hidden_states_281 * rsqrt_94;  hidden_states_281 = rsqrt_94 = None
    l__self___model_model_layers_23_self_attn_k_norm_weight = self.L__self___model_model_layers_23_self_attn_k_norm_weight
    to_192 = hidden_states_282.to(torch.float16);  hidden_states_282 = None
    mul_306 = l__self___model_model_layers_23_self_attn_k_norm_weight * to_192;  l__self___model_model_layers_23_self_attn_k_norm_weight = to_192 = None
    key_states_23 = mul_306.transpose(1, 2);  mul_306 = None
    l__self___model_model_layers_23_self_attn_v_proj_weight = self.L__self___model_model_layers_23_self_attn_v_proj_weight
    linear_163 = torch._C._nn.linear(hidden_states_278, l__self___model_model_layers_23_self_attn_v_proj_weight, None);  hidden_states_278 = l__self___model_model_layers_23_self_attn_v_proj_weight = None
    view_71 = linear_163.view((1, 2048, -1, 128));  linear_163 = None
    value_states_23 = view_71.transpose(1, 2);  view_71 = None
    cos_26 = cos_2.unsqueeze(1)
    sin_26 = sin_2.unsqueeze(1)
    mul_307 = query_states_23 * cos_26
    x1_46 = query_states_23[(Ellipsis, slice(None, 64, None))]
    x2_46 = query_states_23[(Ellipsis, slice(64, None, None))];  query_states_23 = None
    neg_46 = -x2_46;  x2_46 = None
    cat_47 = torch.cat((neg_46, x1_46), dim = -1);  neg_46 = x1_46 = None
    mul_308 = cat_47 * sin_26;  cat_47 = None
    q_embed_23 = mul_307 + mul_308;  mul_307 = mul_308 = None
    mul_309 = key_states_23 * cos_26;  cos_26 = None
    x1_47 = key_states_23[(Ellipsis, slice(None, 64, None))]
    x2_47 = key_states_23[(Ellipsis, slice(64, None, None))];  key_states_23 = None
    neg_47 = -x2_47;  x2_47 = None
    cat_48 = torch.cat((neg_47, x1_47), dim = -1);  neg_47 = x1_47 = None
    mul_310 = cat_48 * sin_26;  cat_48 = sin_26 = None
    k_embed_23 = mul_309 + mul_310;  mul_309 = mul_310 = None
    q_23 = q_embed_23.contiguous();  q_embed_23 = None
    k_23 = k_embed_23.contiguous();  k_embed_23 = None
    v_23 = value_states_23.contiguous();  value_states_23 = None
    npu_fused_infer_attention_score_default_23 = torch.ops.npu.npu_fused_infer_attention_score.default(q_23, k_23, v_23, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_23 = npu_fused_infer_attention_score_default_23[0];  npu_fused_infer_attention_score_default_23 = None
    attn_output_69 = output_23.transpose(1, 2)
    reshape_23 = attn_output_69.reshape(1, 2048, -1);  attn_output_69 = None
    attn_output_70 = reshape_23.contiguous();  reshape_23 = None
    l__self___model_model_layers_23_self_attn_o_proj_weight = self.L__self___model_model_layers_23_self_attn_o_proj_weight
    attn_output_71 = torch._C._nn.linear(attn_output_70, l__self___model_model_layers_23_self_attn_o_proj_weight, None);  attn_output_70 = l__self___model_model_layers_23_self_attn_o_proj_weight = None
    hidden_states_283 = hidden_states_275 + attn_output_71;  hidden_states_275 = attn_output_71 = None
    hidden_states_284 = hidden_states_283.to(torch.float32)
    pow_96 = hidden_states_284.pow(2)
    variance_95 = pow_96.mean(-1, keepdim = True);  pow_96 = None
    add_190 = variance_95 + 1e-06;  variance_95 = None
    rsqrt_95 = torch.rsqrt(add_190);  add_190 = None
    hidden_states_285 = hidden_states_284 * rsqrt_95;  hidden_states_284 = rsqrt_95 = None
    l__self___model_model_layers_23_post_attention_layernorm_weight = self.L__self___model_model_layers_23_post_attention_layernorm_weight
    to_194 = hidden_states_285.to(torch.float16);  hidden_states_285 = None
    hidden_states_286 = l__self___model_model_layers_23_post_attention_layernorm_weight * to_194;  l__self___model_model_layers_23_post_attention_layernorm_weight = to_194 = None
    l__self___model_model_layers_23_mlp_gate_proj_weight = self.L__self___model_model_layers_23_mlp_gate_proj_weight
    linear_165 = torch._C._nn.linear(hidden_states_286, l__self___model_model_layers_23_mlp_gate_proj_weight, None);  l__self___model_model_layers_23_mlp_gate_proj_weight = None
    silu_23 = torch.nn.functional.silu(linear_165);  linear_165 = None
    l__self___model_model_layers_23_mlp_up_proj_weight = self.L__self___model_model_layers_23_mlp_up_proj_weight
    linear_166 = torch._C._nn.linear(hidden_states_286, l__self___model_model_layers_23_mlp_up_proj_weight, None);  hidden_states_286 = l__self___model_model_layers_23_mlp_up_proj_weight = None
    mul_313 = silu_23 * linear_166;  silu_23 = linear_166 = None
    l__self___model_model_layers_23_mlp_down_proj_weight = self.L__self___model_model_layers_23_mlp_down_proj_weight
    down_proj_23 = torch._C._nn.linear(mul_313, l__self___model_model_layers_23_mlp_down_proj_weight, None);  mul_313 = l__self___model_model_layers_23_mlp_down_proj_weight = None
    hidden_states_287 = hidden_states_283 + down_proj_23;  hidden_states_283 = down_proj_23 = None
    hidden_states_288 = hidden_states_287.to(torch.float32)
    pow_97 = hidden_states_288.pow(2)
    variance_96 = pow_97.mean(-1, keepdim = True);  pow_97 = None
    add_192 = variance_96 + 1e-06;  variance_96 = None
    rsqrt_96 = torch.rsqrt(add_192);  add_192 = None
    hidden_states_289 = hidden_states_288 * rsqrt_96;  hidden_states_288 = rsqrt_96 = None
    l__self___model_model_layers_24_input_layernorm_weight = self.L__self___model_model_layers_24_input_layernorm_weight
    to_196 = hidden_states_289.to(torch.float16);  hidden_states_289 = None
    hidden_states_290 = l__self___model_model_layers_24_input_layernorm_weight * to_196;  l__self___model_model_layers_24_input_layernorm_weight = to_196 = None
    l__self___model_model_layers_24_self_attn_q_proj_weight = self.L__self___model_model_layers_24_self_attn_q_proj_weight
    linear_168 = torch._C._nn.linear(hidden_states_290, l__self___model_model_layers_24_self_attn_q_proj_weight, None);  l__self___model_model_layers_24_self_attn_q_proj_weight = None
    view_72 = linear_168.view((1, 2048, -1, 128));  linear_168 = None
    hidden_states_291 = view_72.to(torch.float32);  view_72 = None
    pow_98 = hidden_states_291.pow(2)
    variance_97 = pow_98.mean(-1, keepdim = True);  pow_98 = None
    add_193 = variance_97 + 1e-06;  variance_97 = None
    rsqrt_97 = torch.rsqrt(add_193);  add_193 = None
    hidden_states_292 = hidden_states_291 * rsqrt_97;  hidden_states_291 = rsqrt_97 = None
    l__self___model_model_layers_24_self_attn_q_norm_weight = self.L__self___model_model_layers_24_self_attn_q_norm_weight
    to_198 = hidden_states_292.to(torch.float16);  hidden_states_292 = None
    mul_317 = l__self___model_model_layers_24_self_attn_q_norm_weight * to_198;  l__self___model_model_layers_24_self_attn_q_norm_weight = to_198 = None
    query_states_24 = mul_317.transpose(1, 2);  mul_317 = None
    l__self___model_model_layers_24_self_attn_k_proj_weight = self.L__self___model_model_layers_24_self_attn_k_proj_weight
    linear_169 = torch._C._nn.linear(hidden_states_290, l__self___model_model_layers_24_self_attn_k_proj_weight, None);  l__self___model_model_layers_24_self_attn_k_proj_weight = None
    view_73 = linear_169.view((1, 2048, -1, 128));  linear_169 = None
    hidden_states_293 = view_73.to(torch.float32);  view_73 = None
    pow_99 = hidden_states_293.pow(2)
    variance_98 = pow_99.mean(-1, keepdim = True);  pow_99 = None
    add_194 = variance_98 + 1e-06;  variance_98 = None
    rsqrt_98 = torch.rsqrt(add_194);  add_194 = None
    hidden_states_294 = hidden_states_293 * rsqrt_98;  hidden_states_293 = rsqrt_98 = None
    l__self___model_model_layers_24_self_attn_k_norm_weight = self.L__self___model_model_layers_24_self_attn_k_norm_weight
    to_200 = hidden_states_294.to(torch.float16);  hidden_states_294 = None
    mul_319 = l__self___model_model_layers_24_self_attn_k_norm_weight * to_200;  l__self___model_model_layers_24_self_attn_k_norm_weight = to_200 = None
    key_states_24 = mul_319.transpose(1, 2);  mul_319 = None
    l__self___model_model_layers_24_self_attn_v_proj_weight = self.L__self___model_model_layers_24_self_attn_v_proj_weight
    linear_170 = torch._C._nn.linear(hidden_states_290, l__self___model_model_layers_24_self_attn_v_proj_weight, None);  hidden_states_290 = l__self___model_model_layers_24_self_attn_v_proj_weight = None
    view_74 = linear_170.view((1, 2048, -1, 128));  linear_170 = None
    value_states_24 = view_74.transpose(1, 2);  view_74 = None
    cos_27 = cos_2.unsqueeze(1)
    sin_27 = sin_2.unsqueeze(1)
    mul_320 = query_states_24 * cos_27
    x1_48 = query_states_24[(Ellipsis, slice(None, 64, None))]
    x2_48 = query_states_24[(Ellipsis, slice(64, None, None))];  query_states_24 = None
    neg_48 = -x2_48;  x2_48 = None
    cat_49 = torch.cat((neg_48, x1_48), dim = -1);  neg_48 = x1_48 = None
    mul_321 = cat_49 * sin_27;  cat_49 = None
    q_embed_24 = mul_320 + mul_321;  mul_320 = mul_321 = None
    mul_322 = key_states_24 * cos_27;  cos_27 = None
    x1_49 = key_states_24[(Ellipsis, slice(None, 64, None))]
    x2_49 = key_states_24[(Ellipsis, slice(64, None, None))];  key_states_24 = None
    neg_49 = -x2_49;  x2_49 = None
    cat_50 = torch.cat((neg_49, x1_49), dim = -1);  neg_49 = x1_49 = None
    mul_323 = cat_50 * sin_27;  cat_50 = sin_27 = None
    k_embed_24 = mul_322 + mul_323;  mul_322 = mul_323 = None
    q_24 = q_embed_24.contiguous();  q_embed_24 = None
    k_24 = k_embed_24.contiguous();  k_embed_24 = None
    v_24 = value_states_24.contiguous();  value_states_24 = None
    npu_fused_infer_attention_score_default_24 = torch.ops.npu.npu_fused_infer_attention_score.default(q_24, k_24, v_24, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_24 = npu_fused_infer_attention_score_default_24[0];  npu_fused_infer_attention_score_default_24 = None
    attn_output_72 = output_24.transpose(1, 2)
    reshape_24 = attn_output_72.reshape(1, 2048, -1);  attn_output_72 = None
    attn_output_73 = reshape_24.contiguous();  reshape_24 = None
    l__self___model_model_layers_24_self_attn_o_proj_weight = self.L__self___model_model_layers_24_self_attn_o_proj_weight
    attn_output_74 = torch._C._nn.linear(attn_output_73, l__self___model_model_layers_24_self_attn_o_proj_weight, None);  attn_output_73 = l__self___model_model_layers_24_self_attn_o_proj_weight = None
    hidden_states_295 = hidden_states_287 + attn_output_74;  hidden_states_287 = attn_output_74 = None
    hidden_states_296 = hidden_states_295.to(torch.float32)
    pow_100 = hidden_states_296.pow(2)
    variance_99 = pow_100.mean(-1, keepdim = True);  pow_100 = None
    add_198 = variance_99 + 1e-06;  variance_99 = None
    rsqrt_99 = torch.rsqrt(add_198);  add_198 = None
    hidden_states_297 = hidden_states_296 * rsqrt_99;  hidden_states_296 = rsqrt_99 = None
    l__self___model_model_layers_24_post_attention_layernorm_weight = self.L__self___model_model_layers_24_post_attention_layernorm_weight
    to_202 = hidden_states_297.to(torch.float16);  hidden_states_297 = None
    hidden_states_298 = l__self___model_model_layers_24_post_attention_layernorm_weight * to_202;  l__self___model_model_layers_24_post_attention_layernorm_weight = to_202 = None
    l__self___model_model_layers_24_mlp_gate_proj_weight = self.L__self___model_model_layers_24_mlp_gate_proj_weight
    linear_172 = torch._C._nn.linear(hidden_states_298, l__self___model_model_layers_24_mlp_gate_proj_weight, None);  l__self___model_model_layers_24_mlp_gate_proj_weight = None
    silu_24 = torch.nn.functional.silu(linear_172);  linear_172 = None
    l__self___model_model_layers_24_mlp_up_proj_weight = self.L__self___model_model_layers_24_mlp_up_proj_weight
    linear_173 = torch._C._nn.linear(hidden_states_298, l__self___model_model_layers_24_mlp_up_proj_weight, None);  hidden_states_298 = l__self___model_model_layers_24_mlp_up_proj_weight = None
    mul_326 = silu_24 * linear_173;  silu_24 = linear_173 = None
    l__self___model_model_layers_24_mlp_down_proj_weight = self.L__self___model_model_layers_24_mlp_down_proj_weight
    down_proj_24 = torch._C._nn.linear(mul_326, l__self___model_model_layers_24_mlp_down_proj_weight, None);  mul_326 = l__self___model_model_layers_24_mlp_down_proj_weight = None
    hidden_states_299 = hidden_states_295 + down_proj_24;  hidden_states_295 = down_proj_24 = None
    hidden_states_300 = hidden_states_299.to(torch.float32)
    pow_101 = hidden_states_300.pow(2)
    variance_100 = pow_101.mean(-1, keepdim = True);  pow_101 = None
    add_200 = variance_100 + 1e-06;  variance_100 = None
    rsqrt_100 = torch.rsqrt(add_200);  add_200 = None
    hidden_states_301 = hidden_states_300 * rsqrt_100;  hidden_states_300 = rsqrt_100 = None
    l__self___model_model_layers_25_input_layernorm_weight = self.L__self___model_model_layers_25_input_layernorm_weight
    to_204 = hidden_states_301.to(torch.float16);  hidden_states_301 = None
    hidden_states_302 = l__self___model_model_layers_25_input_layernorm_weight * to_204;  l__self___model_model_layers_25_input_layernorm_weight = to_204 = None
    l__self___model_model_layers_25_self_attn_q_proj_weight = self.L__self___model_model_layers_25_self_attn_q_proj_weight
    linear_175 = torch._C._nn.linear(hidden_states_302, l__self___model_model_layers_25_self_attn_q_proj_weight, None);  l__self___model_model_layers_25_self_attn_q_proj_weight = None
    view_75 = linear_175.view((1, 2048, -1, 128));  linear_175 = None
    hidden_states_303 = view_75.to(torch.float32);  view_75 = None
    pow_102 = hidden_states_303.pow(2)
    variance_101 = pow_102.mean(-1, keepdim = True);  pow_102 = None
    add_201 = variance_101 + 1e-06;  variance_101 = None
    rsqrt_101 = torch.rsqrt(add_201);  add_201 = None
    hidden_states_304 = hidden_states_303 * rsqrt_101;  hidden_states_303 = rsqrt_101 = None
    l__self___model_model_layers_25_self_attn_q_norm_weight = self.L__self___model_model_layers_25_self_attn_q_norm_weight
    to_206 = hidden_states_304.to(torch.float16);  hidden_states_304 = None
    mul_330 = l__self___model_model_layers_25_self_attn_q_norm_weight * to_206;  l__self___model_model_layers_25_self_attn_q_norm_weight = to_206 = None
    query_states_25 = mul_330.transpose(1, 2);  mul_330 = None
    l__self___model_model_layers_25_self_attn_k_proj_weight = self.L__self___model_model_layers_25_self_attn_k_proj_weight
    linear_176 = torch._C._nn.linear(hidden_states_302, l__self___model_model_layers_25_self_attn_k_proj_weight, None);  l__self___model_model_layers_25_self_attn_k_proj_weight = None
    view_76 = linear_176.view((1, 2048, -1, 128));  linear_176 = None
    hidden_states_305 = view_76.to(torch.float32);  view_76 = None
    pow_103 = hidden_states_305.pow(2)
    variance_102 = pow_103.mean(-1, keepdim = True);  pow_103 = None
    add_202 = variance_102 + 1e-06;  variance_102 = None
    rsqrt_102 = torch.rsqrt(add_202);  add_202 = None
    hidden_states_306 = hidden_states_305 * rsqrt_102;  hidden_states_305 = rsqrt_102 = None
    l__self___model_model_layers_25_self_attn_k_norm_weight = self.L__self___model_model_layers_25_self_attn_k_norm_weight
    to_208 = hidden_states_306.to(torch.float16);  hidden_states_306 = None
    mul_332 = l__self___model_model_layers_25_self_attn_k_norm_weight * to_208;  l__self___model_model_layers_25_self_attn_k_norm_weight = to_208 = None
    key_states_25 = mul_332.transpose(1, 2);  mul_332 = None
    l__self___model_model_layers_25_self_attn_v_proj_weight = self.L__self___model_model_layers_25_self_attn_v_proj_weight
    linear_177 = torch._C._nn.linear(hidden_states_302, l__self___model_model_layers_25_self_attn_v_proj_weight, None);  hidden_states_302 = l__self___model_model_layers_25_self_attn_v_proj_weight = None
    view_77 = linear_177.view((1, 2048, -1, 128));  linear_177 = None
    value_states_25 = view_77.transpose(1, 2);  view_77 = None
    cos_28 = cos_2.unsqueeze(1)
    sin_28 = sin_2.unsqueeze(1)
    mul_333 = query_states_25 * cos_28
    x1_50 = query_states_25[(Ellipsis, slice(None, 64, None))]
    x2_50 = query_states_25[(Ellipsis, slice(64, None, None))];  query_states_25 = None
    neg_50 = -x2_50;  x2_50 = None
    cat_51 = torch.cat((neg_50, x1_50), dim = -1);  neg_50 = x1_50 = None
    mul_334 = cat_51 * sin_28;  cat_51 = None
    q_embed_25 = mul_333 + mul_334;  mul_333 = mul_334 = None
    mul_335 = key_states_25 * cos_28;  cos_28 = None
    x1_51 = key_states_25[(Ellipsis, slice(None, 64, None))]
    x2_51 = key_states_25[(Ellipsis, slice(64, None, None))];  key_states_25 = None
    neg_51 = -x2_51;  x2_51 = None
    cat_52 = torch.cat((neg_51, x1_51), dim = -1);  neg_51 = x1_51 = None
    mul_336 = cat_52 * sin_28;  cat_52 = sin_28 = None
    k_embed_25 = mul_335 + mul_336;  mul_335 = mul_336 = None
    q_25 = q_embed_25.contiguous();  q_embed_25 = None
    k_25 = k_embed_25.contiguous();  k_embed_25 = None
    v_25 = value_states_25.contiguous();  value_states_25 = None
    npu_fused_infer_attention_score_default_25 = torch.ops.npu.npu_fused_infer_attention_score.default(q_25, k_25, v_25, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_25 = npu_fused_infer_attention_score_default_25[0];  npu_fused_infer_attention_score_default_25 = None
    attn_output_75 = output_25.transpose(1, 2)
    reshape_25 = attn_output_75.reshape(1, 2048, -1);  attn_output_75 = None
    attn_output_76 = reshape_25.contiguous();  reshape_25 = None
    l__self___model_model_layers_25_self_attn_o_proj_weight = self.L__self___model_model_layers_25_self_attn_o_proj_weight
    attn_output_77 = torch._C._nn.linear(attn_output_76, l__self___model_model_layers_25_self_attn_o_proj_weight, None);  attn_output_76 = l__self___model_model_layers_25_self_attn_o_proj_weight = None
    hidden_states_307 = hidden_states_299 + attn_output_77;  hidden_states_299 = attn_output_77 = None
    hidden_states_308 = hidden_states_307.to(torch.float32)
    pow_104 = hidden_states_308.pow(2)
    variance_103 = pow_104.mean(-1, keepdim = True);  pow_104 = None
    add_206 = variance_103 + 1e-06;  variance_103 = None
    rsqrt_103 = torch.rsqrt(add_206);  add_206 = None
    hidden_states_309 = hidden_states_308 * rsqrt_103;  hidden_states_308 = rsqrt_103 = None
    l__self___model_model_layers_25_post_attention_layernorm_weight = self.L__self___model_model_layers_25_post_attention_layernorm_weight
    to_210 = hidden_states_309.to(torch.float16);  hidden_states_309 = None
    hidden_states_310 = l__self___model_model_layers_25_post_attention_layernorm_weight * to_210;  l__self___model_model_layers_25_post_attention_layernorm_weight = to_210 = None
    l__self___model_model_layers_25_mlp_gate_proj_weight = self.L__self___model_model_layers_25_mlp_gate_proj_weight
    linear_179 = torch._C._nn.linear(hidden_states_310, l__self___model_model_layers_25_mlp_gate_proj_weight, None);  l__self___model_model_layers_25_mlp_gate_proj_weight = None
    silu_25 = torch.nn.functional.silu(linear_179);  linear_179 = None
    l__self___model_model_layers_25_mlp_up_proj_weight = self.L__self___model_model_layers_25_mlp_up_proj_weight
    linear_180 = torch._C._nn.linear(hidden_states_310, l__self___model_model_layers_25_mlp_up_proj_weight, None);  hidden_states_310 = l__self___model_model_layers_25_mlp_up_proj_weight = None
    mul_339 = silu_25 * linear_180;  silu_25 = linear_180 = None
    l__self___model_model_layers_25_mlp_down_proj_weight = self.L__self___model_model_layers_25_mlp_down_proj_weight
    down_proj_25 = torch._C._nn.linear(mul_339, l__self___model_model_layers_25_mlp_down_proj_weight, None);  mul_339 = l__self___model_model_layers_25_mlp_down_proj_weight = None
    hidden_states_311 = hidden_states_307 + down_proj_25;  hidden_states_307 = down_proj_25 = None
    hidden_states_312 = hidden_states_311.to(torch.float32)
    pow_105 = hidden_states_312.pow(2)
    variance_104 = pow_105.mean(-1, keepdim = True);  pow_105 = None
    add_208 = variance_104 + 1e-06;  variance_104 = None
    rsqrt_104 = torch.rsqrt(add_208);  add_208 = None
    hidden_states_313 = hidden_states_312 * rsqrt_104;  hidden_states_312 = rsqrt_104 = None
    l__self___model_model_layers_26_input_layernorm_weight = self.L__self___model_model_layers_26_input_layernorm_weight
    to_212 = hidden_states_313.to(torch.float16);  hidden_states_313 = None
    hidden_states_314 = l__self___model_model_layers_26_input_layernorm_weight * to_212;  l__self___model_model_layers_26_input_layernorm_weight = to_212 = None
    l__self___model_model_layers_26_self_attn_q_proj_weight = self.L__self___model_model_layers_26_self_attn_q_proj_weight
    linear_182 = torch._C._nn.linear(hidden_states_314, l__self___model_model_layers_26_self_attn_q_proj_weight, None);  l__self___model_model_layers_26_self_attn_q_proj_weight = None
    view_78 = linear_182.view((1, 2048, -1, 128));  linear_182 = None
    hidden_states_315 = view_78.to(torch.float32);  view_78 = None
    pow_106 = hidden_states_315.pow(2)
    variance_105 = pow_106.mean(-1, keepdim = True);  pow_106 = None
    add_209 = variance_105 + 1e-06;  variance_105 = None
    rsqrt_105 = torch.rsqrt(add_209);  add_209 = None
    hidden_states_316 = hidden_states_315 * rsqrt_105;  hidden_states_315 = rsqrt_105 = None
    l__self___model_model_layers_26_self_attn_q_norm_weight = self.L__self___model_model_layers_26_self_attn_q_norm_weight
    to_214 = hidden_states_316.to(torch.float16);  hidden_states_316 = None
    mul_343 = l__self___model_model_layers_26_self_attn_q_norm_weight * to_214;  l__self___model_model_layers_26_self_attn_q_norm_weight = to_214 = None
    query_states_26 = mul_343.transpose(1, 2);  mul_343 = None
    l__self___model_model_layers_26_self_attn_k_proj_weight = self.L__self___model_model_layers_26_self_attn_k_proj_weight
    linear_183 = torch._C._nn.linear(hidden_states_314, l__self___model_model_layers_26_self_attn_k_proj_weight, None);  l__self___model_model_layers_26_self_attn_k_proj_weight = None
    view_79 = linear_183.view((1, 2048, -1, 128));  linear_183 = None
    hidden_states_317 = view_79.to(torch.float32);  view_79 = None
    pow_107 = hidden_states_317.pow(2)
    variance_106 = pow_107.mean(-1, keepdim = True);  pow_107 = None
    add_210 = variance_106 + 1e-06;  variance_106 = None
    rsqrt_106 = torch.rsqrt(add_210);  add_210 = None
    hidden_states_318 = hidden_states_317 * rsqrt_106;  hidden_states_317 = rsqrt_106 = None
    l__self___model_model_layers_26_self_attn_k_norm_weight = self.L__self___model_model_layers_26_self_attn_k_norm_weight
    to_216 = hidden_states_318.to(torch.float16);  hidden_states_318 = None
    mul_345 = l__self___model_model_layers_26_self_attn_k_norm_weight * to_216;  l__self___model_model_layers_26_self_attn_k_norm_weight = to_216 = None
    key_states_26 = mul_345.transpose(1, 2);  mul_345 = None
    l__self___model_model_layers_26_self_attn_v_proj_weight = self.L__self___model_model_layers_26_self_attn_v_proj_weight
    linear_184 = torch._C._nn.linear(hidden_states_314, l__self___model_model_layers_26_self_attn_v_proj_weight, None);  hidden_states_314 = l__self___model_model_layers_26_self_attn_v_proj_weight = None
    view_80 = linear_184.view((1, 2048, -1, 128));  linear_184 = None
    value_states_26 = view_80.transpose(1, 2);  view_80 = None
    cos_29 = cos_2.unsqueeze(1)
    sin_29 = sin_2.unsqueeze(1)
    mul_346 = query_states_26 * cos_29
    x1_52 = query_states_26[(Ellipsis, slice(None, 64, None))]
    x2_52 = query_states_26[(Ellipsis, slice(64, None, None))];  query_states_26 = None
    neg_52 = -x2_52;  x2_52 = None
    cat_53 = torch.cat((neg_52, x1_52), dim = -1);  neg_52 = x1_52 = None
    mul_347 = cat_53 * sin_29;  cat_53 = None
    q_embed_26 = mul_346 + mul_347;  mul_346 = mul_347 = None
    mul_348 = key_states_26 * cos_29;  cos_29 = None
    x1_53 = key_states_26[(Ellipsis, slice(None, 64, None))]
    x2_53 = key_states_26[(Ellipsis, slice(64, None, None))];  key_states_26 = None
    neg_53 = -x2_53;  x2_53 = None
    cat_54 = torch.cat((neg_53, x1_53), dim = -1);  neg_53 = x1_53 = None
    mul_349 = cat_54 * sin_29;  cat_54 = sin_29 = None
    k_embed_26 = mul_348 + mul_349;  mul_348 = mul_349 = None
    q_26 = q_embed_26.contiguous();  q_embed_26 = None
    k_26 = k_embed_26.contiguous();  k_embed_26 = None
    v_26 = value_states_26.contiguous();  value_states_26 = None
    npu_fused_infer_attention_score_default_26 = torch.ops.npu.npu_fused_infer_attention_score.default(q_26, k_26, v_26, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0)
    output_26 = npu_fused_infer_attention_score_default_26[0];  npu_fused_infer_attention_score_default_26 = None
    attn_output_78 = output_26.transpose(1, 2)
    reshape_26 = attn_output_78.reshape(1, 2048, -1);  attn_output_78 = None
    attn_output_79 = reshape_26.contiguous();  reshape_26 = None
    l__self___model_model_layers_26_self_attn_o_proj_weight = self.L__self___model_model_layers_26_self_attn_o_proj_weight
    attn_output_80 = torch._C._nn.linear(attn_output_79, l__self___model_model_layers_26_self_attn_o_proj_weight, None);  attn_output_79 = l__self___model_model_layers_26_self_attn_o_proj_weight = None
    hidden_states_319 = hidden_states_311 + attn_output_80;  hidden_states_311 = attn_output_80 = None
    hidden_states_320 = hidden_states_319.to(torch.float32)
    pow_108 = hidden_states_320.pow(2)
    variance_107 = pow_108.mean(-1, keepdim = True);  pow_108 = None
    add_214 = variance_107 + 1e-06;  variance_107 = None
    rsqrt_107 = torch.rsqrt(add_214);  add_214 = None
    hidden_states_321 = hidden_states_320 * rsqrt_107;  hidden_states_320 = rsqrt_107 = None
    l__self___model_model_layers_26_post_attention_layernorm_weight = self.L__self___model_model_layers_26_post_attention_layernorm_weight
    to_218 = hidden_states_321.to(torch.float16);  hidden_states_321 = None
    hidden_states_322 = l__self___model_model_layers_26_post_attention_layernorm_weight * to_218;  l__self___model_model_layers_26_post_attention_layernorm_weight = to_218 = None
    l__self___model_model_layers_26_mlp_gate_proj_weight = self.L__self___model_model_layers_26_mlp_gate_proj_weight
    linear_186 = torch._C._nn.linear(hidden_states_322, l__self___model_model_layers_26_mlp_gate_proj_weight, None);  l__self___model_model_layers_26_mlp_gate_proj_weight = None
    silu_26 = torch.nn.functional.silu(linear_186);  linear_186 = None
    l__self___model_model_layers_26_mlp_up_proj_weight = self.L__self___model_model_layers_26_mlp_up_proj_weight
    linear_187 = torch._C._nn.linear(hidden_states_322, l__self___model_model_layers_26_mlp_up_proj_weight, None);  hidden_states_322 = l__self___model_model_layers_26_mlp_up_proj_weight = None
    mul_352 = silu_26 * linear_187;  silu_26 = linear_187 = None
    l__self___model_model_layers_26_mlp_down_proj_weight = self.L__self___model_model_layers_26_mlp_down_proj_weight
    down_proj_26 = torch._C._nn.linear(mul_352, l__self___model_model_layers_26_mlp_down_proj_weight, None);  mul_352 = l__self___model_model_layers_26_mlp_down_proj_weight = None
    hidden_states_323 = hidden_states_319 + down_proj_26;  hidden_states_319 = down_proj_26 = None
    hidden_states_324 = hidden_states_323.to(torch.float32)
    pow_109 = hidden_states_324.pow(2)
    variance_108 = pow_109.mean(-1, keepdim = True);  pow_109 = None
    add_216 = variance_108 + 1e-06;  variance_108 = None
    rsqrt_108 = torch.rsqrt(add_216);  add_216 = None
    hidden_states_325 = hidden_states_324 * rsqrt_108;  hidden_states_324 = rsqrt_108 = None
    l__self___model_model_layers_27_input_layernorm_weight = self.L__self___model_model_layers_27_input_layernorm_weight
    to_220 = hidden_states_325.to(torch.float16);  hidden_states_325 = None
    hidden_states_326 = l__self___model_model_layers_27_input_layernorm_weight * to_220;  l__self___model_model_layers_27_input_layernorm_weight = to_220 = None
    l__self___model_model_layers_27_self_attn_q_proj_weight = self.L__self___model_model_layers_27_self_attn_q_proj_weight
    linear_189 = torch._C._nn.linear(hidden_states_326, l__self___model_model_layers_27_self_attn_q_proj_weight, None);  l__self___model_model_layers_27_self_attn_q_proj_weight = None
    view_81 = linear_189.view((1, 2048, -1, 128));  linear_189 = None
    hidden_states_327 = view_81.to(torch.float32);  view_81 = None
    pow_110 = hidden_states_327.pow(2)
    variance_109 = pow_110.mean(-1, keepdim = True);  pow_110 = None
    add_217 = variance_109 + 1e-06;  variance_109 = None
    rsqrt_109 = torch.rsqrt(add_217);  add_217 = None
    hidden_states_328 = hidden_states_327 * rsqrt_109;  hidden_states_327 = rsqrt_109 = None
    l__self___model_model_layers_27_self_attn_q_norm_weight = self.L__self___model_model_layers_27_self_attn_q_norm_weight
    to_222 = hidden_states_328.to(torch.float16);  hidden_states_328 = None
    mul_356 = l__self___model_model_layers_27_self_attn_q_norm_weight * to_222;  l__self___model_model_layers_27_self_attn_q_norm_weight = to_222 = None
    query_states_27 = mul_356.transpose(1, 2);  mul_356 = None
    l__self___model_model_layers_27_self_attn_k_proj_weight = self.L__self___model_model_layers_27_self_attn_k_proj_weight
    linear_190 = torch._C._nn.linear(hidden_states_326, l__self___model_model_layers_27_self_attn_k_proj_weight, None);  l__self___model_model_layers_27_self_attn_k_proj_weight = None
    view_82 = linear_190.view((1, 2048, -1, 128));  linear_190 = None
    hidden_states_329 = view_82.to(torch.float32);  view_82 = None
    pow_111 = hidden_states_329.pow(2)
    variance_110 = pow_111.mean(-1, keepdim = True);  pow_111 = None
    add_218 = variance_110 + 1e-06;  variance_110 = None
    rsqrt_110 = torch.rsqrt(add_218);  add_218 = None
    hidden_states_330 = hidden_states_329 * rsqrt_110;  hidden_states_329 = rsqrt_110 = None
    l__self___model_model_layers_27_self_attn_k_norm_weight = self.L__self___model_model_layers_27_self_attn_k_norm_weight
    to_224 = hidden_states_330.to(torch.float16);  hidden_states_330 = None
    mul_358 = l__self___model_model_layers_27_self_attn_k_norm_weight * to_224;  l__self___model_model_layers_27_self_attn_k_norm_weight = to_224 = None
    key_states_27 = mul_358.transpose(1, 2);  mul_358 = None
    l__self___model_model_layers_27_self_attn_v_proj_weight = self.L__self___model_model_layers_27_self_attn_v_proj_weight
    linear_191 = torch._C._nn.linear(hidden_states_326, l__self___model_model_layers_27_self_attn_v_proj_weight, None);  hidden_states_326 = l__self___model_model_layers_27_self_attn_v_proj_weight = None
    view_83 = linear_191.view((1, 2048, -1, 128));  linear_191 = None
    value_states_27 = view_83.transpose(1, 2);  view_83 = None
    cos_30 = cos_2.unsqueeze(1);  cos_2 = None
    sin_30 = sin_2.unsqueeze(1);  sin_2 = None
    mul_359 = query_states_27 * cos_30
    x1_54 = query_states_27[(Ellipsis, slice(None, 64, None))]
    x2_54 = query_states_27[(Ellipsis, slice(64, None, None))];  query_states_27 = None
    neg_54 = -x2_54;  x2_54 = None
    cat_55 = torch.cat((neg_54, x1_54), dim = -1);  neg_54 = x1_54 = None
    mul_360 = cat_55 * sin_30;  cat_55 = None
    q_embed_27 = mul_359 + mul_360;  mul_359 = mul_360 = None
    mul_361 = key_states_27 * cos_30;  cos_30 = None
    x1_55 = key_states_27[(Ellipsis, slice(None, 64, None))]
    x2_55 = key_states_27[(Ellipsis, slice(64, None, None))];  key_states_27 = None
    neg_55 = -x2_55;  x2_55 = None
    cat_56 = torch.cat((neg_55, x1_55), dim = -1);  neg_55 = x1_55 = None
    mul_362 = cat_56 * sin_30;  cat_56 = sin_30 = None
    k_embed_27 = mul_361 + mul_362;  mul_361 = mul_362 = None
    q_27 = q_embed_27.contiguous();  q_embed_27 = None
    k_27 = k_embed_27.contiguous();  k_embed_27 = None
    v_27 = value_states_27.contiguous();  value_states_27 = None
    npu_fused_infer_attention_score_default_27 = torch.ops.npu.npu_fused_infer_attention_score.default(q_27, k_27, v_27, atten_mask = l_causal_mask_, actual_seq_lengths = [2048], actual_seq_lengths_kv = [2048], num_heads = 16, num_key_value_heads = 8, scale = 0.08838834764831845, input_layout = 'BNSD', pre_tokens = 2147483647, next_tokens = 0, sparse_mode = 2, inner_precise = 0);  l_causal_mask_ = None
    output_27 = npu_fused_infer_attention_score_default_27[0];  npu_fused_infer_attention_score_default_27 = None
    attn_output_81 = output_27.transpose(1, 2)
    reshape_27 = attn_output_81.reshape(1, 2048, -1);  attn_output_81 = None
    attn_output_82 = reshape_27.contiguous();  reshape_27 = None
    l__self___model_model_layers_27_self_attn_o_proj_weight = self.L__self___model_model_layers_27_self_attn_o_proj_weight
    attn_output_83 = torch._C._nn.linear(attn_output_82, l__self___model_model_layers_27_self_attn_o_proj_weight, None);  attn_output_82 = l__self___model_model_layers_27_self_attn_o_proj_weight = None
    hidden_states_331 = hidden_states_323 + attn_output_83;  hidden_states_323 = attn_output_83 = None
    hidden_states_332 = hidden_states_331.to(torch.float32)
    pow_112 = hidden_states_332.pow(2)
    variance_111 = pow_112.mean(-1, keepdim = True);  pow_112 = None
    add_222 = variance_111 + 1e-06;  variance_111 = None
    rsqrt_111 = torch.rsqrt(add_222);  add_222 = None
    hidden_states_333 = hidden_states_332 * rsqrt_111;  hidden_states_332 = rsqrt_111 = None
    l__self___model_model_layers_27_post_attention_layernorm_weight = self.L__self___model_model_layers_27_post_attention_layernorm_weight
    to_226 = hidden_states_333.to(torch.float16);  hidden_states_333 = None
    hidden_states_334 = l__self___model_model_layers_27_post_attention_layernorm_weight * to_226;  l__self___model_model_layers_27_post_attention_layernorm_weight = to_226 = None
    l__self___model_model_layers_27_mlp_gate_proj_weight = self.L__self___model_model_layers_27_mlp_gate_proj_weight
    linear_193 = torch._C._nn.linear(hidden_states_334, l__self___model_model_layers_27_mlp_gate_proj_weight, None);  l__self___model_model_layers_27_mlp_gate_proj_weight = None
    silu_27 = torch.nn.functional.silu(linear_193);  linear_193 = None
    l__self___model_model_layers_27_mlp_up_proj_weight = self.L__self___model_model_layers_27_mlp_up_proj_weight
    linear_194 = torch._C._nn.linear(hidden_states_334, l__self___model_model_layers_27_mlp_up_proj_weight, None);  hidden_states_334 = l__self___model_model_layers_27_mlp_up_proj_weight = None
    mul_365 = silu_27 * linear_194;  silu_27 = linear_194 = None
    l__self___model_model_layers_27_mlp_down_proj_weight = self.L__self___model_model_layers_27_mlp_down_proj_weight
    down_proj_27 = torch._C._nn.linear(mul_365, l__self___model_model_layers_27_mlp_down_proj_weight, None);  mul_365 = l__self___model_model_layers_27_mlp_down_proj_weight = None
    hidden_states_335 = hidden_states_331 + down_proj_27;  hidden_states_331 = down_proj_27 = None
    hidden_states_336 = hidden_states_335.to(torch.float32);  hidden_states_335 = None
    pow_113 = hidden_states_336.pow(2)
    variance_112 = pow_113.mean(-1, keepdim = True);  pow_113 = None
    add_224 = variance_112 + 1e-06;  variance_112 = None
    rsqrt_112 = torch.rsqrt(add_224);  add_224 = None
    hidden_states_337 = hidden_states_336 * rsqrt_112;  hidden_states_336 = rsqrt_112 = None
    l__self___model_model_norm_weight = self.L__self___model_model_norm_weight
    to_228 = hidden_states_337.to(torch.float16);  hidden_states_337 = None
    hidden_states_338 = l__self___model_model_norm_weight * to_228;  l__self___model_model_norm_weight = to_228 = None
    getitem_195 = hidden_states_338[(slice(None, None, None), slice(-1, None, None), slice(None, None, None))];  hidden_states_338 = None
    logits = torch._C._nn.linear(getitem_195, l__self___model_model_embed_tokens_weight, None);  getitem_195 = l__self___model_model_embed_tokens_weight = None
    return pytree.tree_unflatten([logits, q, k, v, output, q_1, k_1, v_1, output_1, q_2, k_2, v_2, output_2, q_3, k_3, v_3, output_3, q_4, k_4, v_4, output_4, q_5, k_5, v_5, output_5, q_6, k_6, v_6, output_6, q_7, k_7, v_7, output_7, q_8, k_8, v_8, output_8, q_9, k_9, v_9, output_9, q_10, k_10, v_10, output_10, q_11, k_11, v_11, output_11, q_12, k_12, v_12, output_12, q_13, k_13, v_13, output_13, q_14, k_14, v_14, output_14, q_15, k_15, v_15, output_15, q_16, k_16, v_16, output_16, q_17, k_17, v_17, output_17, q_18, k_18, v_18, output_18, q_19, k_19, v_19, output_19, q_20, k_20, v_20, output_20, q_21, k_21, v_21, output_21, q_22, k_22, v_22, output_22, q_23, k_23, v_23, output_23, q_24, k_24, v_24, output_24, q_25, k_25, v_25, output_25, q_26, k_26, v_26, output_26, q_27, k_27, v_27, output_27], self._out_spec)
    