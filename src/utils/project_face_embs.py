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
    
    # ✅ FIX: привести dtype источника к dtype назначения (обычно fp16)
    face_embs_padded = face_embs_padded.to(dtype=token_embs.dtype)
    # replace embeddings at "id"
    # replace embeddings at "id"
    mask = (input_ids_b == arcface_token_id)
    token_embs[mask] = face_embs_padded.repeat_interleave(int(mask.sum(dim=1)[0].item()), dim=0)
    
    # --- manual CLIPTextTransformer forward (robust) ---
    text_model = pipeline.text_encoder.text_model
    
    attn = attention_mask_b
    attn = (1.0 - attn.float()) * -10000.0
    attn = attn[:, None, None, :]  # (N,1,1,T)
    
    enc_out = text_model.encoder(
        inputs_embeds=token_embs,
        attention_mask=attn,
        return_dict=True,
    )
    
    prompt_embeds = text_model.final_layer_norm(enc_out.last_hidden_state)
    return prompt_embeds
