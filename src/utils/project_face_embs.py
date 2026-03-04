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

    # Forward text encoder in the most version-compatible way.
    outputs = text_encoder(input_ids=input_ids_b, attention_mask=attention_mask_b, return_dict=True)
    token_embs = outputs.last_hidden_state  # [N, T, H]

    # Find index of "id" token
    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
    id_pos = (input_ids[0] == arcface_token_id).nonzero(as_tuple=True)[0].item()

    # Match ArcFace size (512) to CLIP hidden size.
    hidden_size = token_embs.shape[-1]
    if hidden_size >= 512:
        face_embs_padded = F.pad(face_embs.to(device), (0, hidden_size - 512), "constant", 0)
    else:
        face_embs_padded = face_embs.to(device)[:, :hidden_size]

    # Replace "id" token state with ArcFace embedding proxy.
    token_embs[:, id_pos, :] = face_embs_padded
    return token_embs  # [N, T, H]
