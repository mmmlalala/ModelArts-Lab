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

"""Patch Qwen3NextAttention.forward to use cloud cloud_split_rms_mrope_gate.

When VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO is 0 (default) and the model is a
qwen3_5 variant, the fused ``cloud_ops_turbo.cloud_split_rms_mrope_gate``
AscendC operator replaces the Triton ``triton_split_qkv_rmsnorm_mrope``
path used by vllm-ascend's ``AscendQwen3NextAttention.forward``. For any
other case (env var set to 1, or non-qwen3_5 model) we delegate to the
original forward unchanged.

The cloud operator ``cloud_split_rms_mrope_gate`` fuses rmsnorm + mrope + qkv split
and internally applies the ``1.0 + weight`` correction that the Triton path does in
Python, so we pass ``self.q_norm.weight`` / ``self.k_norm.weight`` directly (matching
the ascend-vllm reference). The call also takes the raw ``cos_sin_cache`` and
``positions`` instead of a pre-indexed ``cos_sin`` tensor.

The env var is read at call time (matching patch_chunk_fla.py), so toggling
it between requests takes effect without re-importing the module.
"""

import torch
import vllm_ascend.envs as envs
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

# Capture the forward that vllm-ascend's patch_qwen3_5 already installed
# (AscendQwen3NextAttention.forward). The meta-path hook fires after
# vllm_ascend.ops is imported, which happens after adapt_patch() applies the
# worker patches in Worker.__init__.
_orig_forward = Qwen3NextAttention.forward


def _patched_forward(
    self,
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
):
    use_cloud_ops = not envs.VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO

    if not use_cloud_ops or "qwen3_5" not in self.config.model_type:
        # Legacy / non-qwen3_5 path — delegate to the vllm-ascend forward
        # unchanged. This re-runs qkv_proj inside _orig_forward, which is
        # acceptable because this branch is only the debug fallback or the
        # non-fused model path.
        _orig_forward(self, positions, output, hidden_states)
        return

    # cloud_ops_turbo is imported lazily: a module-level import triggers
    # device property queries before init_device_properties_triton() runs
    # during worker setup ("Device properties not initialized").
    import cloud_ops_turbo  # noqa: F401

    qkv, _ = self.qkv_proj(hidden_states)

    if not positions.is_contiguous():
        positions = positions.contiguous()
    # Reference: ascend-vllm/patch/worker/patch_qwen3_5.py
    # cloud op applies the (1.0 + weight) correction internally, so we pass
    # the raw norm weights and the cos_sin_cache + positions directly (no
    # pre-indexing, no +1 in Python).
    q, k, v, gate = torch.ops.cloud_ops_turbo.cloud_split_rms_mrope_gate(
        qkv,
        self.q_norm.weight,
        self.k_norm.weight,
        self.rotary_emb.cos_sin_cache,
        positions.to(torch.int64),
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
    )

    attn_output = self.attn(q, k, v)

    if self.attn_output_gate:
        gate = torch.sigmoid(gate)
        attn_output = attn_output * gate

    output[:], _ = self.o_proj(attn_output)


Qwen3NextAttention.forward = _patched_forward
