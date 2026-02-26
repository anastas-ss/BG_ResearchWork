

import torch
import torch.nn as nn

class DualImageAttnProcessor(nn.Module):
    def __init__(
        self,
        base_processor,
        hidden_size: int,
        cross_attention_dim: int,
        scale_id: float = 1.0,
        scale_hair: float = 1.0,
        attn_fp32: bool = True,
    ):
        super().__init__()
        self.base = base_processor
        self.scale_id = scale_id
        self.scale_hair = scale_hair
        self.attn_fp32 = attn_fp32

        self.to_k_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_k_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        nn.init.zeros_(self.to_k_id.weight)
        nn.init.zeros_(self.to_v_id.weight)
        nn.init.zeros_(self.to_k_hair.weight)
        nn.init.zeros_(self.to_v_hair.weight)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs
    ):
        # fallback: normal mode
        if not isinstance(encoder_hidden_states, dict):
            return self.base(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs
            )

        text_states = encoder_hidden_states["text"]
        id_states = encoder_hidden_states["id"]
        hair_states = encoder_hidden_states["hair"]

        # base attention to text (no recursion)
        base_out = self.base(
            attn,
            hidden_states,
            encoder_hidden_states=text_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs
        )

        if (self.scale_id == 0.0) and (self.scale_hair == 0.0):
            return base_out

        out_dtype = base_out.dtype
        result = base_out

        # shared query
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        query_ = query.float() if self.attn_fp32 else query

        # ID branch
        if self.scale_id != 0.0:
            k_id = attn.head_to_batch_dim(self.to_k_id(id_states))
            v_id = attn.head_to_batch_dim(self.to_v_id(id_states))

            if self.attn_fp32:
                p_id = attn.get_attention_scores(query_, k_id.float(), attention_mask=None)
                out_id = torch.bmm(p_id, v_id.float()).to(dtype=out_dtype)
            else:
                p_id = attn.get_attention_scores(query_, k_id, attention_mask=None)
                out_id = torch.bmm(p_id, v_id)

            out_id = attn.batch_to_head_dim(out_id)
            out_id = attn.to_out[1](attn.to_out[0](out_id))

            result = result + self.scale_id * out_id

        # Hair branch
        if self.scale_hair != 0.0:
            k_h = attn.head_to_batch_dim(self.to_k_hair(hair_states))
            v_h = attn.head_to_batch_dim(self.to_v_hair(hair_states))

            if self.attn_fp32:
                p_h = attn.get_attention_scores(query_, k_h.float(), attention_mask=None)
                out_h = torch.bmm(p_h, v_h.float()).to(dtype=out_dtype)
            else:
                p_h = attn.get_attention_scores(query_, k_h, attention_mask=None)
                out_h = torch.bmm(p_h, v_h)

            out_h = attn.batch_to_head_dim(out_h)
            out_h = attn.to_out[1](attn.to_out[0](out_h))

            result = result + self.scale_hair * out_h

        return result
