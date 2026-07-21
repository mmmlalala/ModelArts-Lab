#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Patch GatedDeltaNetAttention._forward_core to use cloud_causal_conv1d_update.

When VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO is 0 (default), the decode-path
``torch.ops._C_ascend.npu_causal_conv1d_custom`` calls (both the spec
multi-query branch and the non-spec decode branch) are routed to the fused
``cloud_ops_turbo.cloud_causal_conv1d_update`` AscendC operator, matching
the ascend-vllm reference. The prefill path keeps ``npu_causal_conv1d_custom``
because the cloud operator does not provide a run_mode=0 (varlen prefill)
equivalent. Setting the env var to 1 keeps the legacy implementation.

Parameter call pattern (kwarg names, conv_state / bias / query_start_loc /
cache_indices / num_accepted_tokens / activation_mode / pad_slot_id) follows
ascend-vllm/patch/worker/patch_gdn.py:cloud_ops_turbo_forward_core, adapted
to the current vllm-ascend GDNAttentionMetadata layout
(``attn_metadata.spec_decode_metadata.spec_causal_conv1d`` and
``attn_metadata.non_spec_decode_metadata.causal_conv1d``).
"""

import torch
import vllm_ascend.envs as envs
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention as _GDN_PATCH_TARGET,
)
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width

# Capture the _forward_core that vllm-ascend's patch_qwen3_5 already installed
# (AscendGatedDeltaNetAttention._forward_core). The meta-path hook fires after
# vllm_ascend.ops is imported, which happens after adapt_patch() applies the
# worker patches in Worker.__init__.
_orig_forward_core = _GDN_PATCH_TARGET._forward_core


def _forward_core_cloud(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    """Core attention computation with cloud_causal_conv1d_update on decode paths.

    Mirrors AscendGatedDeltaNetAttention._forward_core but routes the spec and
    non-spec decode conv1d updates through cloud_ops_turbo. The prefill path
    is unchanged because the cloud operator has no varlen-prefill equivalent.
    """
    import cloud_ops_turbo  # noqa: F401

    forward_context = get_forward_context()
    attn_metadata: AttentionMetadata = forward_context.attn_metadata

    if attn_metadata is None:
        # V1 profile run
        return

    assert isinstance(attn_metadata, dict)
    attn_metadata = attn_metadata[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    self_kv_cache = self.kv_cache
    ssm_state = self_kv_cache[1]
    num_actual_tokens = attn_metadata.num_actual_tokens

    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    # 1. Convolution sequence transformation
    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            mixed_qkv_non_spec = None
        else:
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    # 1.1: Process the multi-query part (spec decode) — cloud op
    if spec_sequence_masks is not None:
        conv_weights_T = conv_weights.transpose(0, 1)
        activation_num = 1 if self.activation else 0
        spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
        spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc
        # Reference: ascend-vllm cloud_causal_conv1d_update call. The cloud op
        # returns the output tensor directly (no separate output arg, no
        # run_mode / initial_state_mode_opt).
        mixed_qkv_spec = torch.ops.cloud_ops_turbo.cloud_causal_conv1d_update(
            mixed_qkv_spec,
            conv_weights_T,
            conv_state=self_kv_cache[0],
            bias=self.conv1d.bias,
            query_start_loc=spec_query_start_loc_device,
            cache_indices=spec_causal_conv1d_meta.cache_indices[:, 0],
            num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(
                torch.int32
            ),
            activation_mode=activation_num,
            pad_slot_id=PAD_SLOT_ID,
        )

    # 1.2: Process the remaining part
    if attn_metadata.num_prefills > 0:
        if mixed_qkv_non_spec is not None:
            non_spec_causal_conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
            query_start_loc_opt = non_spec_causal_conv1d_meta.query_start_loc
            cache_indices_opt = non_spec_causal_conv1d_meta.cache_indices
            initial_state_mode_opt = non_spec_causal_conv1d_meta.initial_state_mode
            if get_pcp_group().world_size > 1:
                conv_weights_T = conv_weights.transpose(0, 1)
                activation_num = 1 if self.activation else 0
                non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                assert non_spec_query_start_loc is not None
                non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                width = conv_weights.shape[1]
                state_len = width - 1
                num_seqs = non_spec_query_start_loc.shape[0] - 1
                prefill_seq_offset = max(0, num_seqs - attn_metadata.num_prefills)
                prefill_cache_indices = non_spec_state_indices_tensor[prefill_seq_offset:]
                mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                last_width_prefill_x = extract_last_width(
                    mixed_qkv_non_spec_T,
                    non_spec_query_start_loc[prefill_seq_offset:],
                    state_len,
                )
                pcp_rank = get_pcp_group().rank_in_group
                all_last_width_prefill_x = get_pcp_group().all_gather(
                    last_width_prefill_x.unsqueeze(0).contiguous(), 0
                )
                if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
                    self_kv_cache[0][prefill_cache_indices, :state_len, :] = (
                        all_last_width_prefill_x[pcp_rank - 1, ...].transpose(-1, -2)
                    )
                mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                # Prefill path keeps npu_causal_conv1d_custom (no cloud equivalent).
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    mixed_qkv_non_spec_output,
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=query_start_loc_opt,
                    cache_indices_opt=cache_indices_opt,
                    initial_state_mode_opt=initial_state_mode_opt,
                    num_accepted_tokens_opt=None,
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )
                mixed_qkv_non_spec = mixed_qkv_non_spec_output
                if prefill_cache_indices.shape[0] > 0:
                    self_kv_cache[0][prefill_cache_indices, :state_len, :] = (
                        all_last_width_prefill_x[-1, ...].transpose(-1, -2)
                    )
            else:
                conv_weights_T = conv_weights.transpose(0, 1)
                activation_num = 1 if self.activation else 0
                mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                # Prefill path keeps npu_causal_conv1d_custom (no cloud equivalent).
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    mixed_qkv_non_spec_output,
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=query_start_loc_opt,
                    cache_indices_opt=cache_indices_opt,
                    initial_state_mode_opt=initial_state_mode_opt,
                    num_accepted_tokens_opt=None,
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )
                mixed_qkv_non_spec = mixed_qkv_non_spec_output
    elif attn_metadata.num_decodes > 0:
        conv_weights_T = conv_weights.transpose(0, 1)
        activation_num = 1 if self.activation else 0
        non_spec_causal_conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
        non_spec_query_start_loc_device = non_spec_causal_conv1d_meta.query_start_loc
        # Non-spec decode path — cloud op (reference: ascend-vllm).
        mixed_qkv_non_spec = torch.ops.cloud_ops_turbo.cloud_causal_conv1d_update(
            mixed_qkv_non_spec,
            conv_weights_T,
            conv_state=self_kv_cache[0],
            bias=self.conv1d.bias,
            query_start_loc=non_spec_query_start_loc_device,
            cache_indices=non_spec_causal_conv1d_meta.cache_indices[:, 0]
            if non_spec_causal_conv1d_meta.cache_indices.dim() > 1
            else non_spec_causal_conv1d_meta.cache_indices,
            num_accepted_tokens=None,
            activation_mode=activation_num,
            pad_slot_id=PAD_SLOT_ID,
        )
    else:
        mixed_qkv_non_spec = None

    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
        mixed_qkv_non_spec
    )

    # 2. Recurrent attention
    g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            g_spec = g
            beta_spec = beta
            g_non_spec = None
            beta_non_spec = None
        else:
            g_spec = g.index_select(1, spec_token_indx)
            beta_spec = beta.index_select(1, spec_token_indx)
            g_non_spec = g.index_select(1, non_spec_token_indx)
            beta_non_spec = beta.index_select(1, non_spec_token_indx)
    else:
        g_spec = None
        beta_spec = None
        g_non_spec = g
        beta_non_spec = beta

    split_non_spec = (
        spec_sequence_masks is None
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes > 0
    )
    num_decode_tokens = attn_metadata.num_decode_tokens

    # 2.1: Process the multi-query part
    if spec_sequence_masks is not None:
        actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
        query_spec = l2norm_fwd(query_spec)
        key_spec = l2norm_fwd(key_spec)
        core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query_spec.squeeze(0),
            key=key_spec.squeeze(0),
            value=value_spec.squeeze(0),
            g=g_spec.squeeze(0),
            beta=beta_spec.squeeze(0),
            state=ssm_state,
            scale=key_spec.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=spec_state_indices_tensor.flatten(),
            num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(
                torch.int32
            ),
        ).unsqueeze(0)
    else:
        core_attn_out_spec, last_recurrent_state = None, None

    # 2.2: Process non-spec-decode part in mixed non-spec batches
    if split_non_spec:
        assert mixed_qkv_non_spec is not None
        assert g_non_spec is not None
        assert beta_non_spec is not None
        query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec[:num_decode_tokens]
        )
        actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
        query_decode = l2norm_fwd(query_decode)
        key_decode = l2norm_fwd(key_decode)
        core_attn_out_decode = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query_decode.squeeze(0),
            key=key_decode.squeeze(0),
            value=value_decode.squeeze(0),
            g=g_non_spec[:, :num_decode_tokens].squeeze(0),
            beta=beta_non_spec[:, :num_decode_tokens].squeeze(0),
            state=ssm_state,
            scale=key_decode.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
        ).unsqueeze(0)
    else:
        core_attn_out_decode = None

    # 2.3: Process the remaining part
    if attn_metadata.num_prefills > 0:
        prefill_query_start_loc = attn_metadata.prefill_query_start_loc
        prefill_state_indices = attn_metadata.prefill_state_indices
        prefill_has_initial_state = attn_metadata.prefill_has_initial_state
        assert prefill_query_start_loc is not None
        assert prefill_state_indices is not None
        assert prefill_has_initial_state is not None
        assert g_non_spec is not None
        assert beta_non_spec is not None
        if split_non_spec:
            query_non_spec = query_non_spec[:, num_decode_tokens:]
            key_non_spec = key_non_spec[:, num_decode_tokens:]
            value_non_spec = value_non_spec[:, num_decode_tokens:]
            g_non_spec = g_non_spec[:, num_decode_tokens:]
            beta_non_spec = beta_non_spec[:, num_decode_tokens:]

        initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
        clear_ssm_states(initial_state, prefill_has_initial_state)
        (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
            q=query_non_spec,
            k=key_non_spec,
            v=value_non_spec,
            g=g_non_spec,
            beta=beta_non_spec,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[prefill_state_indices] = (
            last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
        )
        if split_non_spec:
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_decode, core_attn_out_non_spec],
                dim=1,
            )
    elif attn_metadata.num_decodes > 0:
        actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
        query_non_spec = l2norm_fwd(query_non_spec)
        key_non_spec = l2norm_fwd(key_non_spec)
        core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query_non_spec.squeeze(0),
            key=key_non_spec.squeeze(0),
            value=value_non_spec.squeeze(0),
            g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
            beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
            state=ssm_state,
            scale=key_non_spec.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=non_spec_state_indices_tensor,
        ).unsqueeze(0)
    else:
        core_attn_out_non_spec, last_recurrent_state = None, None

    # 3. Merge core attention output
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


def _patched_forward_core(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    """Dispatch to the cloud variant unless VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO=1.

    Reading the env var at call time (matching patch_chunk_fla.py) lets the
    operator be toggled between requests without re-importing the module.
    """
    if envs.VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO:
        _orig_forward_core(self, mixed_qkv, b, a, core_attn_out)
        return
    _forward_core_cloud(self, mixed_qkv, b, a, core_attn_out)


_GDN_PATCH_TARGET._forward_core = _patched_forward_core
