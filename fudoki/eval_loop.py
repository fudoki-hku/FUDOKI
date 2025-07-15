# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from torch.nn.modules import Module

from flow_matching.utils import ModelWrapper

PRINT_FREQUENCY = 50


logger = logging.getLogger(__name__)


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module, g_or_u):
        super().__init__(model)
        self.nfe_counter = 0
        self.g_or_u = g_or_u
        assert self.g_or_u in ['understanding', 'generation', 'generation and understanding']

    def forward(
        self, x: torch.Tensor, cfg_scale: float, datainfo, uncond_id=100015
    ):
        with torch.no_grad():
            conditional_img_logits, conditional_txt_logits = self.model(x, datainfo)
            if self.g_or_u == 'understanding':
                conditional_logits = conditional_txt_logits
                result_txt = conditional_logits
                result_txt = top_k_logits(result_txt, top_k=1) 
                result_img = None
            elif self.g_or_u == 'generation':
                conditional_logits = conditional_img_logits

                uncondition_x = x.clone()
                text_token_mask = datainfo['text_token_mask']
                for bs in range(text_token_mask.shape[0]):
                    nz = datainfo['text_token_mask'][bs].nonzero()
                    if nz.numel() > 0:  # Make sure there's at least one nonzero
                        text_nonzero_idx_begin = nz[0, 0]  # first nonzero along dim=0
                        text_nonzero_idx_end = nz[-1, 0]
                        uncondition_x[bs, text_nonzero_idx_begin:text_nonzero_idx_end+1] = uncond_id
                unconditional_img_logits, _ = self.model(uncondition_x, datainfo)
                unconditional_logits = unconditional_img_logits
                result_img = (1.0 + cfg_scale) * conditional_logits - cfg_scale * unconditional_logits
                # result_img = top_k_logits(result_img, top_k=1) # forcing this reduces diversity
                result_txt = None
            else:
                result_img = conditional_img_logits
                result_txt = conditional_txt_logits

        self.nfe_counter += 1
        if self.g_or_u == 'understanding':
            return torch.softmax(result_txt.to(dtype=torch.float32), dim=-1), result_img, datainfo
        elif self.g_or_u == 'generation':
            return result_txt, torch.softmax(result_img.to(dtype=torch.float32), dim=-1), datainfo
        else:
            return torch.softmax(result_txt.to(dtype=torch.float32), dim=-1), torch.softmax(result_img.to(dtype=torch.float32), dim=-1), datainfo
      
    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter
