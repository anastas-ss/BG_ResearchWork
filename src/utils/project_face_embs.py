import torch
import torch.nn.functional as F

@torch.no_grad()
def project_face_embs(pipeline, face_embs):
    """
    face_embs: (N, 512) normalized ArcFace embeddings
    Returns:
        prompt_embeds: (N, T, D) готовые текстовые эмбеддинги с заменой токена 'id'
    """

    device = pipeline.device
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]

    # 1. Токенизация базового текста
    tok = tokenizer(
        "photo of a id person",
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)

    N = face_embs.shape[0]

    # 2. Расширение batch
    input_ids_b = input_ids.repeat(N, 1)
    attention_mask_b = attention_mask.repeat(N, 1)

    # 3. Получаем стандартные токен-эмбеддинги
    token_embs = text_encoder(input_ids=input_ids_b, return_token_embs=True)  # (N, T, D)

    # 4. Подготавливаем ArcFace эмбеддинги и pad до hidden_size
    hidden = text_encoder.config.hidden_size
    face_embs_padded = F.pad(face_embs.to(device), (0, hidden - 512), "constant", 0).to(dtype=token_embs.dtype)

    # 5. Заменяем токен 'id' на эмбеддинг лица
    for i in range(N):
        id_pos = (input_ids_b[i] == arcface_token_id).nonzero(as_tuple=True)[0]
        if len(id_pos) > 0:
            token_embs[i, id_pos[0]] = face_embs_padded[i]

    # 6. Forward через CLIPTextModel с input_token_embs
    prompt_embeds = text_encoder(input_ids=input_ids_b, input_token_embs=token_embs, attention_mask=attention_mask_b)[0]

    return prompt_embeds
