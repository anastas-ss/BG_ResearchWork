
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if isinstance(encoder_hidden_states, dict):
            text_states = encoder_hidden_states["text"]
            if torch.all(encoder_hidden_states["id"] == 0) and torch.all(encoder_hidden_states["hair"] == 0):
                return self.base(
                    attn,
                    hidden_states,
                    encoder_hidden_states=text_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    **kwargs
                )
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

        id_states = id_states.to(dtype=self.to_k_id.weight.dtype)
        hair_states = hair_states.to(dtype=self.to_k_hair.weight.dtype)

        # base attention to text (no recursion)
        base_out = self.base(
            attn,
            hidden_states,
            encoder_hidden_states=text_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs
        )

        # --- FIX: make shapes consistent between base_out and id/hair branches ---
        # Some diffusers blocks pass hidden_states as 4D (B,C,H,W), while attention math expects 3D (B,L,C).
        # Base processor handles reshaping internally; we must mirror it so seq_len matches.
        
        hs = hidden_states
        is_4d = (hs.dim() == 4)
        if is_4d:
            b, c, h, w = hs.shape
            hs_3d = hs.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, L, C)
        else:
            hs_3d = hs  # (B, L, C)
        
        # base_out can be 4D or 3D depending on block; convert to 3D for addition
        bo = base_out
        if bo.dim() == 4:
            b2, c2, h2, w2 = bo.shape
            bo_3d = bo.permute(0, 2, 3, 1).reshape(b2, h2 * w2, c2)  # (B, L, C)
            bo_hw = (h2, w2, c2)
        else:
            bo_3d = bo
            bo_hw = None
        
        out_dtype = bo_3d.dtype
        result = bo_3d
        
        if (self.scale_id == 0.0) and (self.scale_hair == 0.0):
            return base_out

        # shared query (use reshaped hs_3d so seq_len matches base_out)
        query = attn.to_q(hs_3d)
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
        # restore original base_out shape if needed
        if bo_hw is not None:
            h2, w2, c2 = bo_hw
            result = result.reshape(-1, h2, w2, c2).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        return result
