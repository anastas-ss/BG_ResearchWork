import torch
import torch.nn as nn
import torch.nn.functional as F


class DualImageAttnProcessor(nn.Module):
    """
    Hair-only IP-Adapter style attention processor.

    The project used to carry a second ID-token branch. Identity is now provided
    only through the Arc2Face text stream, while this processor learns only the
    hair image stream. In dict mode it expects:
      {"text": text_encoder_states, "hair": hair_tokens}

    Important: like IP-Adapter, the extra image-attention output is added to the
    text-attention output before the shared `attn.to_out` projection.
    """

    def __init__(
        self,
        base_processor,
        hidden_size: int,
        cross_attention_dim: int,
        scale_hair: float = 1.0,
        attn_fp32: bool = True,
    ):
        super().__init__()
        self.base = base_processor
        self.scale_hair = float(scale_hair)
        self.attn_fp32 = bool(attn_fp32)

        self.to_k_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_hair = nn.Linear(cross_attention_dim, hidden_size, bias=False)

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
        hair_states = encoder_hidden_states["hair"]

        # Fast-path: external stream off -> exactly the original text cross-attention.
        if self.scale_hair == 0.0 or torch.all(hair_states == 0):
            return self.base(
                attn,
                hidden_states,
                encoder_hidden_states=text_states,
                attention_mask=attention_mask,
                temb=temb,
                **kwargs,
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None

        sequence_length = text_states.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        out_dtype = query.dtype

        if attn.norm_cross:
            text_states = attn.norm_encoder_hidden_states(text_states)

        k_text = attn.head_to_batch_dim(attn.to_k(text_states))
        v_text = attn.head_to_batch_dim(attn.to_v(text_states))
        query = attn.head_to_batch_dim(query)
        query_for_scores = query.float() if self.attn_fp32 else query

        if self.attn_fp32:
            p_text = attn.get_attention_scores(query_for_scores, k_text.float(), attention_mask)
            result = torch.bmm(p_text, v_text.float()).to(dtype=out_dtype)
        else:
            p_text = attn.get_attention_scores(query_for_scores, k_text, attention_mask)
            result = torch.bmm(p_text, v_text)

        result = attn.batch_to_head_dim(result)

        # Hair branch. It is added before the shared output projection, matching IP-Adapter.
        hair_states = hair_states.to(dtype=self.to_k_hair.weight.dtype)
        k_h = attn.head_to_batch_dim(self.to_k_hair(hair_states))
        v_h = attn.head_to_batch_dim(self.to_v_hair(hair_states))

        if self.attn_fp32:
            p_h = attn.get_attention_scores(query_for_scores, k_h.float(), attention_mask=None)
            out_h = torch.bmm(p_h, v_h.float()).to(dtype=out_dtype)
        else:
            p_h = attn.get_attention_scores(query_for_scores, k_h, attention_mask=None)
            out_h = torch.bmm(p_h, v_h)

        out_h = attn.batch_to_head_dim(out_h)

        if out_h.shape[1] != result.shape[1]:
            if out_h.shape[1] > result.shape[1]:
                out_h = out_h[:, : result.shape[1], :]
            else:
                out_h = F.pad(out_h, (0, 0, 0, result.shape[1] - out_h.shape[1]), value=0.0)

        result = result + self.scale_hair * out_h

        result = result.to(dtype=attn.to_out[0].weight.dtype)
        result = attn.to_out[0](result)
        result = attn.to_out[1](result)

        if input_ndim == 4:
            result = result.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            result = result + residual

        result = result / attn.rescale_output_factor
        return result
