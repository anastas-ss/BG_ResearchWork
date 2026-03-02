# src/utils/project_face_embs.py

import torch
import torch.nn.functional as F


@torch.no_grad()
def project_face_embs(
    pipeline,
    face_embs: torch.Tensor,
    *,
    prompt: str = "photo of a id person",
    id_token: str = "id",
) -> torch.Tensor:
    """
    Arc2Face-style prompt embedding:
      - takes ArcFace embeddings (B,512)
      - injects them into the CLIP text prompt at token `id_token`
      - returns CLIP last_hidden_state (B,T,H)

    face_embs: (B,512) torch tensor (preferably L2-normalized)
    """

    device = pipeline.device
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    text_model = text_encoder.text_model

    if face_embs.ndim != 2 or face_embs.shape[1] != 512:
        raise ValueError(f"face_embs must be (B,512), got {tuple(face_embs.shape)}")

    # id_token must be exactly one token
    ids = tokenizer.encode(id_token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f'"{id_token}" tokenizes into {len(ids)} tokens: {ids}. Use a single-token marker.')
    arcface_token_id = ids[0]

    tok = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)            # (1,T)
    attention_mask = tok.attention_mask.to(device)  # (1,T)

    B = face_embs.shape[0]
    input_ids_b = input_ids.repeat(B, 1)            # (B,T)
    attention_mask_b = attention_mask.repeat(B, 1)  # (B,T)

    # match dtype to text encoder embeddings dtype (usually fp16 under SD1.5 fp16)
    emb_layer = text_encoder.get_input_embeddings()
    emb_dtype = emb_layer.weight.dtype
    face_embs = face_embs.to(device=device, dtype=emb_dtype)

    # get token embeddings (B,T,H)
    token_embs = emb_layer(input_ids_b)

    H = token_embs.shape[-1]
    if H < 512:
        raise ValueError(f"text hidden size {H} < 512")

    face_embs_padded = F.pad(face_embs, (0, H - 512), value=0)  # (B,H)

    # replace embeddings at id_token positions
    mask = (input_ids_b == arcface_token_id)  # (B,T)
    if not torch.any(mask):
        raise ValueError(f'Prompt "{prompt}" does not contain token "{id_token}" (id={arcface_token_id})')

    # handle possibly multiple 'id' tokens per prompt (rare): repeat per occurrence
    k = int(mask.sum(dim=1)[0].item())
    token_embs[mask] = face_embs_padded.repeat_interleave(k, dim=0)

    # ---- manual CLIPTextTransformer forward (robust across transformers versions) ----
    # build additive attention bias for SDPA: (B,1,1,T)
    attn = (1.0 - attention_mask_b.float()) * -10000.0
    attn = attn[:, None, None, :].to(dtype=token_embs.dtype)

    enc_out = text_model.encoder(
        inputs_embeds=token_embs,
        attention_mask=attn,
        return_dict=True,
    )
    prompt_embeds = text_model.final_layer_norm(enc_out.last_hidden_state)  # (B,T,H)
    return prompt_embeds
