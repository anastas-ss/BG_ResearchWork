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

    # токенизируем промпт
    text = "photo of a id person"
    tok = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )

    input_ids = tok.input_ids.to(device)             # [1, T]
    attention_mask = tok.attention_mask.to(device)  # [1, T]

    # повторяем для батча
    input_ids_b = input_ids.repeat(N, 1)            # [N, T]
    attention_mask_b = attention_mask.repeat(N, 1)  # [N, T]

    # получаем токен эмбеддинги напрямую через input_ids
    token_embs_out = text_encoder(input_ids=input_ids_b, return_token_embs=True)
    token_embs = token_embs_out.last_hidden_state  # теперь это [N, T, H] с dtype

    # находим позицию токена "id"
    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
    id_pos = (input_ids_b[0] == arcface_token_id).nonzero(as_tuple=True)[0].item()

    # расширяем face_embs до hidden_size и подменяем
    hidden_size = text_encoder.config.hidden_size
    face_embs_padded = F.pad(face_embs.to(device), (0, hidden_size - 512), "constant", 0)
    token_embs[:, id_pos, :] = face_embs_padded

    # forward через text_encoder только с inputs_embeds
    outputs = text_encoder(
        inputs_embeds=token_embs,
        attention_mask=attention_mask_b,
        return_dict=True
    )

    return outputs.last_hidden_state
