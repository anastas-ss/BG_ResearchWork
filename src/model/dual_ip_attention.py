import math

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
        processor_name: str = "",
        spatial_gate_hair: bool = False,
    ):
        super().__init__()
        self.base = base_processor
        self.scale_hair = float(scale_hair)
        self.attn_fp32 = bool(attn_fp32)
        self.processor_name = str(processor_name)
        self.spatial_gate_hair = bool(spatial_gate_hair)
        self.hair_attention_callback = None
        self.last_hair_localization_loss = None
        self.last_hair_debug_stats = None

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
        hair_token_mask = encoder_hidden_states.get("hair_token_mask")
        hair_spatial_mask = encoder_hidden_states.get("hair_spatial_mask")
        self.last_hair_localization_loss = None
        self.last_hair_debug_stats = None

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

        key_for_scores = k_h.float() if self.attn_fp32 else k_h
        value_for_scores = v_h.float() if self.attn_fp32 else v_h
        hair_scores = torch.bmm(query_for_scores, key_for_scores.transpose(1, 2))
        hair_scores = hair_scores * attn.scale
        if hair_token_mask is not None:
            token_mask = hair_token_mask.to(device=hair_scores.device, dtype=hair_scores.dtype)
            empty_rows = token_mask.sum(dim=1, keepdim=True) <= 1e-8
            score_token_mask = torch.where(
                empty_rows,
                torch.ones_like(token_mask),
                token_mask,
            )
            score_token_mask = score_token_mask.repeat_interleave(attn.heads, dim=0).unsqueeze(1)
            hair_scores = hair_scores + score_token_mask.clamp_min(1e-8).log()
        p_h = hair_scores.softmax(dim=-1)
        out_h = torch.bmm(p_h, value_for_scores).to(dtype=out_dtype)

        out_h = attn.batch_to_head_dim(out_h)
        raw_out_h = out_h

        spatial_height = height
        spatial_width = width
        if spatial_height is None or spatial_width is None:
            query_length = out_h.shape[1]
            side = int(math.isqrt(query_length))
            if side * side == query_length:
                spatial_height = side
                spatial_width = side

        spatial_mask = None
        if (
            hair_spatial_mask is not None
            and spatial_height is not None
            and spatial_width is not None
        ):
            spatial_mask = F.interpolate(
                hair_spatial_mask.to(device=out_h.device, dtype=torch.float32).unsqueeze(1),
                size=(spatial_height, spatial_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0].flatten(1)
            energy = raw_out_h.float().pow(2).mean(dim=-1)
            valid = spatial_mask.sum(dim=1) > 1e-6
            if hair_token_mask is not None:
                valid = valid & (hair_token_mask.to(device=valid.device).sum(dim=1) > 1e-6)
            if valid.any():
                outside_weight = 1.0 - spatial_mask
                outside_energy = (energy * outside_weight).sum(dim=1)
                outside_mean = outside_energy / outside_weight.sum(dim=1).clamp_min(1.0)
                inside_energy = (energy * spatial_mask).sum(dim=1)
                inside_mean = inside_energy / spatial_mask.sum(dim=1).clamp_min(1.0)
                localization_fraction = outside_mean / (
                    outside_mean + inside_mean + 1e-8
                )
                self.last_hair_localization_loss = localization_fraction[valid].mean()
            else:
                self.last_hair_localization_loss = raw_out_h.float().sum() * 0.0

            if self.spatial_gate_hair:
                out_h = out_h * spatial_mask.unsqueeze(-1).to(dtype=out_h.dtype)

        text_output_norm = result.detach().float().norm(dim=-1).mean()
        gated_output_norm = out_h.detach().float().norm(dim=-1).mean()
        finite_scores = hair_scores.detach().float()
        finite_scores = finite_scores[torch.isfinite(finite_scores)]
        self.last_hair_debug_stats = {
            "key_norm": k_h.detach().float().norm(dim=-1).mean(),
            "value_norm": v_h.detach().float().norm(dim=-1).mean(),
            "attention_logit_std": (
                finite_scores.std(unbiased=False)
                if finite_scores.numel() > 0
                else torch.zeros((), device=out_h.device)
            ),
            "attention_max": p_h.detach().float().max(dim=-1).values.mean(),
            "raw_output_norm": raw_out_h.detach().float().norm(dim=-1).mean(),
            "gated_output_norm": gated_output_norm,
            "text_output_norm": text_output_norm,
            "hair_to_text_ratio": gated_output_norm / text_output_norm.clamp_min(1e-8),
            "attention_entropy": (
                -(p_h.detach().float().clamp_min(1e-8) * p_h.detach().float().clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
            ),
            "token_mask_mean": (
                hair_token_mask.detach().float().mean()
                if hair_token_mask is not None
                else torch.ones((), device=out_h.device)
            ),
            "spatial_mask_mean": (
                spatial_mask.detach().float().mean()
                if spatial_mask is not None
                else torch.ones((), device=out_h.device)
            ),
        }

        if self.hair_attention_callback is not None:
            probs = p_h.detach().float().reshape(
                batch_size,
                -1,
                p_h.shape[1],
                p_h.shape[2],
            ).mean(dim=1)
            entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
            self.hair_attention_callback(
                {
                    "processor_name": self.processor_name,
                    "spatial_height": spatial_height,
                    "spatial_width": spatial_width,
                    "contribution_norm": raw_out_h.detach().float().norm(dim=-1),
                    "token_attention_max": probs.max(dim=-1).values,
                    "token_attention_entropy": entropy,
                }
            )

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
