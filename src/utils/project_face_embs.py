import torch
import torch.nn.functional as F

@torch.no_grad()
def project_face_embs(pipeline, face_embs):
    """
    face_embs: (N, 512) normalized ArcFace embeddings
    Returns: [N, T, H] text embeddings ready for UNet conditioning
    """

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    device = pipeline.device
    N = face_embs.shape[0]

    # Tokenize prompt
    prompt = "photo of a id person"
    tok = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)

    input_ids = tok.input_ids      # [1, T]
    attention_mask = tok.attention_mask  # [1, T]
    input_ids_b = input_ids.repeat(N, 1)  # [N, T]
    attention_mask_b = attention_mask.repeat(N, 1)  # [N, T]

    # Find index of "id" token
    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
    id_mask = (input_ids == arcface_token_id).repeat(N, 1)

    # Arc2Face path: requires CLIPTextModelWrapper (return_token_embs + input_token_embs)
    if hasattr(text_encoder, "forward"):
        try:
            token_embs = text_encoder(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                return_token_embs=True,
            )

            hidden_size = token_embs.shape[-1]
            if hidden_size >= 512:
                face_embs_padded = F.pad(face_embs.to(device), (0, hidden_size - 512), "constant", 0)
            else:
                face_embs_padded = face_embs.to(device)[:, :hidden_size]

            token_embs[id_mask] = face_embs_padded
            prompt_embeds = text_encoder(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                input_token_embs=token_embs,
            )[0]
            return prompt_embeds
        except TypeError:
            # Fallback to vanilla CLIPTextModel-compatible path below.
            pass

    # Vanilla CLIP fallback (less faithful to Arc2Face, but robust).
    outputs = text_encoder(input_ids=input_ids_b, attention_mask=attention_mask_b, return_dict=True)
    token_embs = outputs.last_hidden_state  # [N, T, H]
    hidden_size = token_embs.shape[-1]
    if hidden_size >= 512:
        face_embs_padded = F.pad(face_embs.to(device), (0, hidden_size - 512), "constant", 0)
    else:
        face_embs_padded = face_embs.to(device)[:, :hidden_size]

    token_embs[id_mask] = face_embs_padded
    return token_embs  # [N, T, H]
