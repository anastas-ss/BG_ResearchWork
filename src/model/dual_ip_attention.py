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
        self.scale_id = float(scale_id)
        self.scale_hair = float(scale_hair)
        self.attn_fp32 = bool(attn_fp32)

        self.to_k_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_k_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        nn.init.zeros_(self.to_k_id.weight)
        nn.init.zeros_(self.to_v_id.weight)
        nn.init.zeros_(self.to_k_hair.weight)
        nn.init.zeros_(self.to_v_hair.weight)

    @staticmethod
    def _to_3d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int] | None]:
        """
        Convert (B,C,H,W) -> (B, H*W, C). If already 3D, return as-is.
        Returns: (x_3d, hwc_or_none) where hwc=(H,W,C) used for restoring.
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x3 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            return x3, (h, w, c)
        return x, None

    @staticmethod
    def _restore_from_3d(x3: torch.Tensor, hwc: tuple[int, int, int] | None) -> torch.Tensor:
        if hwc is None:
            return x3
        h, w, c = hwc
        return x3.reshape(-1, h, w, c).permute(0, 3, 1, 2).contiguous()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        # If not our dict-mode: behave like base
        if not isinstance(encoder_hidden_states, dict):
            return self.base(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        text_states = encoder_hidden_states["text"]
        id_states = encoder_hidden_states["id"]
        hair_states = encoder_hidden_states["hair"]

        # Fast-path: both external streams off
        if torch.all(id_states == 0) and torch.all(hair_states == 0):
            return self.base(
                attn,
                hidden_states,
                encoder_hidden_states=text_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        # Base attention to text (diffusers handles masking/reshaping internally)
        base_out = self.base(
            attn,
            hidden_states,
            encoder_hidden_states=text_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs,
        )

        # If both scales are 0, return base_out exactly (keep original shape)
        if (self.scale_id == 0.0) and (self.scale_hair == 0.0):
            return base_out

        # Align hidden_states and base_out to (B,L,C) so additions are safe
        hs_3d, _ = self._to_3d(hidden_states)
        bo_3d, bo_hwc = self._to_3d(base_out)

        result = bo_3d
        out_dtype = bo_3d.dtype

        # Ensure id/hair dtype matches the projection layers
        id_states = id_states.to(dtype=self.to_k_id.weight.dtype)
        hair_states = hair_states.to(dtype=self.to_k_hair.weight.dtype)

        # Shared query for both branches (must use same L as base_out)
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

            out_id = attn.batch_to_head_dim(out_id)                 # (B,L,C)
            out_id = attn.to_out[1](attn.to_out[0](out_id))         # (B,L,C)

            # Safety: match shapes (should already match, but protect against rare cases)
            if out_id.shape[1] != result.shape[1]:
                if out_id.shape[1] > result.shape[1]:
                    out_id = out_id[:, : result.shape[1], :]
                else:
                    out_id = F.pad(out_id, (0, 0, 0, result.shape[1] - out_id.shape[1]), value=0.0)

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

            out_h = attn.batch_to_head_dim(out_h)                   # (B,L,C)
            out_h = attn.to_out[1](attn.to_out[0](out_h))           # (B,L,C)

            if out_h.shape[1] != result.shape[1]:
                if out_h.shape[1] > result.shape[1]:
                    out_h = out_h[:, : result.shape[1], :]
                else:
                    out_h = F.pad(out_h, (0, 0, 0, result.shape[1] - out_h.shape[1]), value=0.0)

            result = result + self.scale_hair * out_h

        # Restore shape if base_out was 4D
        result = self._restore_from_3d(result, bo_hwc)
        return result
