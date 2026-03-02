import torch
import torch.nn.functional as F

@torch.no_grad()
def project_face_embs(pipeline, face_embs):
    """
    face_embs: (N, 512) normalized ArcFace embeddings
    Возвращает: (N, T, H) эмбеддинги для текстового энкодера
    """

    device = pipeline.device
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder  # Diffusers wrapper
    text_model = text_encoder.text_model  # Сам CLIPTextModel

    N = face_embs.shape[0]

    # ---- 1. токенизируем prompt
    text = "photo of a id person"
    tok = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)

    # ---- 2. повторяем батч
    input_ids_b = input_ids.repeat(N, 1)
    attention_mask_b = attention_mask.repeat(N, 1)

    # ---- 3. получаем стандартные token embeddings
    token_embs = text_model.get_input_embeddings()(input_ids_b)

    # ---- 4. готовим face embedding
    hidden = text_model.config.hidden_size
    face_embs = face_embs.to(device)
    face_embs_padded = F.pad(face_embs, (0, hidden - 512), "constant", 0)
    face_embs_padded = face_embs_padded.to(dtype=token_embs.dtype)

    # ---- 5. находим позицию токена "id"
    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]

    for i in range(N):
        id_pos = (input_ids_b[i] == arcface_token_id).nonzero(as_tuple=True)[0]
        if len(id_pos) > 0:
            token_embs[i, id_pos[0]] = face_embs_padded[i]

    # ---- 6. forward через CLIPTextModel напрямую
    outputs = text_model(
        inputs_embeds=token_embs,
        attention_mask=attention_mask_b,
        return_dict=True,
    )

    prompt_embeds = outputs.last_hidden_state
    return prompt_embeds
