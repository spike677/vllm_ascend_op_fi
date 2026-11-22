# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team. Copyright 2023 DeepSeek-AI.
# Exact two method excerpts from vllm-ascend v0.32.0rc1 commit
# 24487f4f48ee99cf45d86d1c504e645549b8c1ef, token_dispatcher.py SHA
# 43e57927ed54a2d2a1193ccf8bbcd56d1751417e460954bc0b8b967d2a23b228.
# Licensed under Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
# CPU fixture scaffolding below substitutes result dataclasses and native API.
from __future__ import annotations
from types import SimpleNamespace
MoETokenDispatchOutput=MoEMC2CombineMetadata=SimpleNamespace
torch_npu=SimpleNamespace()

class TokenDispatcherWithMC2:
    def token_dispatch(
            self,
            token_dispatch_input: MoETokenDispatchInput,
        ):
            kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
            output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
            # comm_stream.wait_stream(torch.npu.current_stream())
            (
                expand_x,
                dynamic_scale,
                assist_info_for_combine,
                expert_token_nums,
                ep_recv_counts,
                tp_recv_counts,
                expand_scales,
            ) = output[0:7]
    
            group_list_type = kwargs_mc2["expert_token_nums_type"]
            return MoETokenDispatchOutput(
                hidden_states=expand_x,
                dynamic_scale=dynamic_scale,
                group_list=expert_token_nums,
                group_list_type=group_list_type,
                combine_metadata=MoEMC2CombineMetadata(
                    topk_ids=token_dispatch_input.topk_ids,
                    topk_weights=token_dispatch_input.topk_weights,
                    expert_map=token_dispatch_input.routing.expert_map,
                    ep_recv_counts=ep_recv_counts,
                    tp_recv_counts=tp_recv_counts,
                    assist_info_for_combine=assist_info_for_combine,
                    expand_scales=expand_scales,
                    quant=token_dispatch_input.quant,
                    mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
                ),
            )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
            assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."
    
            kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
            combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
    
            return combined_output
