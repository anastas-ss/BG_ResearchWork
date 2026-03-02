import scipy
import PIL
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def project_face_embs(pipeline, face_embs):

    '''
    face_embs: (N, 512) normalized ArcFace embeddings
    '''

    arcface_token_id = pipeline.tokenizer.encode("id", add_special_tokens=False)[0]

    tok = pipeline.tokenizer(
        "photo of a id person",
        truncation=True,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(pipeline.device)              
    attention_mask = tok.attention_mask.to(pipeline.device)   
    
    N = len(face_embs)
    input_ids_b = input_ids.repeat(N, 1)
    attention_mask_b = attention_mask.repeat(N, 1)

    hidden = pipeline.text_encoder.config.hidden_size
    face_embs_padded = F.pad(face_embs, (0, hidden - 512), "constant", 0)

    # ✅ FIX #1: вместо return_token_embs=True
    token_embs = pipeline.text_encoder.get_input_embeddings()(input_ids_b)  # (N,T,H)

    # replace embeddings at "id"
    mask = (input_ids_b == arcface_token_id)
    token_embs[mask] = face_embs_padded.repeat_interleave(int(mask.sum(dim=1)[0].item()), dim=0)

    # ✅ FIX #2: вместо input_token_embs=...
    prompt_embeds = pipeline.text_encoder(
        input_ids=None,
        inputs_embeds=token_embs,
        attention_mask=attention_mask_b,
        return_dict=True,
    ).last_hidden_state

    return prompt_embeds
