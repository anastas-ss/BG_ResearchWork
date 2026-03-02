import torch
import torchvision
from pathlib import Path

@torch.no_grad()
def hair_leakage_check_one(
    *,
    run_dir: Path,
    step: int,
    pipe,
    scheduler,
    pil_images,
    id_cond,
    hair_cond,
    dtype_unet: torch.dtype,
    num_steps: int = 30,
    seed: int = 123,
    cfg_scale: float = 7.0,
):
    """
    Проверка "протечки" лица из hair-ветки:
    Делаем 3 варианта с одним и тем же шумом:
      1) text only
      2) text + hair(original mask)
      3) text + hair(face-masked)  (hair маска занулена на лице, остаются только волосы)
    Если (2) и (3) сильно отличаются лицом => в hair-ветку протекает лицо.
    """
    device = pipe.device
    B = len(pil_images)
    assert B > 0, "pil_images is empty"

    out_dir = Path(run_dir) / "hair_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- text embeddings
    prompt = "a portrait photo of a person"
    tok = pipe.tokenizer([prompt] * B, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt").to(device)
    tok_uc = pipe.tokenizer([""] * B, padding="max_length",
                            max_length=pipe.tokenizer.model_max_length,
                            return_tensors="pt").to(device)
    text_c = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)
    text_u = pipe.text_encoder(**tok_uc).last_hidden_state.to(dtype_unet)

    # ---- get hair tokens TWO ways (original and "face-masked")
    # 1) original
    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

    # 2) face-masked: используем face bbox из insightface и зануляем маску на лице
    #    Для этого надо, чтобы hair_cond умел принять "override_mask" или "override_pil".
    #    Быстрый путь: просто выкинем лицо из hair-картинки до CLIP:
    #    (тут делаем это через существующий hair_cond.enc_h + apply_mask_to_pil)
    masks = hair_cond.enc_h(pil_images)  # (B,512,512) hair mask
    # примитив: "убить центр лица" кругом, если нет bbox (работает как quick smoke test)
    # лучше заменить на bbox (ниже дам как)
    fm = masks.clone()
    H, W = fm.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy = yy.to(fm.device)
    xx = xx.to(fm.device)
    cy, cx = H // 2, W // 2
    r = int(0.22 * H)
    face_disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < (r ** 2)
    fm[:, face_disk] = 0.0  # вырезали центральную область (лицо)

    # соберём hair_pil и прогоним через CLIP напрямую (копируем логику HairConditioner)
    from src.model.hair_conditioner_parsing import apply_mask_to_pil
    hair_pil_face_off = [apply_mask_to_pil(im, fm[i], bg=hair_cond.bg_value) for i, im in enumerate(pil_images)]
    inputs = hair_cond.proc(images=hair_pil_face_off, return_tensors="pt").to(device)
    pooled = hair_cond.clip(**inputs).pooler_output
    hair_tokens_face_off = hair_cond.proj(pooled.float()).view(B, hair_cond.n_tokens, hair_cond.cross_dim).to(dtype=dtype_unet)

    # ---- latents
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn((B, 4, 64, 64), device=device, dtype=dtype_unet, generator=gen)

    scheduler.set_timesteps(int(num_steps), device=device)
    latents = latents * scheduler.init_noise_sigma

    def run(enc_c, enc_u):
        x = latents.clone()
        for t in scheduler.timesteps:
            t_int = int(t.item())
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
            x_in = scheduler.scale_model_input(x, t) if hasattr(scheduler, "scale_model_input") else x
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps_u = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_u).sample
                eps_c = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_c).sample
            eps = eps_u + float(cfg_scale) * (eps_c - eps_u)
            x = scheduler.step(eps, t, x).prev_sample
        imgs = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
        imgs = (imgs.float() * 0.5 + 0.5).clamp(0, 1)
        return imgs

    # id = zeros for leakage test
    zeros_id = torch.zeros((B, 1, pipe.unet.config.cross_attention_dim), device=device, dtype=dtype_unet)

    enc_text_only_c = {"text": text_c, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}
    enc_text_only_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}

    enc_hair_on_c = {"text": text_c, "id": zeros_id, "hair": hair_tokens}
    enc_hair_on_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}

    enc_hair_faceoff_c = {"text": text_c, "id": zeros_id, "hair": hair_tokens_face_off}
    enc_hair_faceoff_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens_face_off)}

    img_text = run(enc_text_only_c, enc_text_only_u)[0]
    img_hair = run(enc_hair_on_c, enc_hair_on_u)[0]
    img_faceoff = run(enc_hair_faceoff_c, enc_hair_faceoff_u)[0]

    grid = torchvision.utils.make_grid(torch.stack([img_text.cpu(), img_hair.cpu(), img_faceoff.cpu()], dim=0), nrow=3)
    out_path = out_dir / f"step_{step:07d}.png"
    torchvision.utils.save_image(grid, str(out_path))
    print(f"[hair_leakage] saved {out_path} (text_only | hair_on | hair_face_off)")
# import torch
# import numpy as np
# from PIL import Image

# @torch.no_grad()
# def hair_leakage_check_one(
#     *,
#     pil: Image.Image,
#     hair_cond,         # твой HairConditioner (с debug_save=True)
#     face_app,          # insightface.app.FaceAnalysis
#     arcface_embedder,  # твой InsightFaceArcFaceEmbedder (или просто FaceAnalysis + faces[0].embedding)
#     out_prefix="leak",
# ):
#     # 1) получаем hair_masked картинку так же, как в hair_cond
#     hair_cond.debug_save = True
#     _ = hair_cond([pil], out_dtype=torch.float16)   # триггерит debug_hair.png
#     hair_masked = Image.open("debug_hair.png").convert("RGB")

#     pil.save(f"{out_prefix}_orig.png")
#     hair_masked.save(f"{out_prefix}_hair_masked.png")

#     # 2) ищем лицо на hair_masked (если находится — лицо явно “протекает” в визуальном смысле)
#     img_bgr = np.array(hair_masked)[:, :, ::-1]
#     faces = face_app.get(img_bgr)
#     if len(faces) == 0:
#         print("[leak] no face detected on hair_masked (good sign)")
#         return

#     f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
#     x1,y1,x2,y2 = [int(v) for v in f.bbox]
#     x1,y1 = max(0,x1), max(0,y1)
#     crop = hair_masked.crop((x1,y1,x2,y2))
#     crop.save(f"{out_prefix}_face_cut_from_hair.png")
#     print("[leak] face detected on hair_masked -> визуальная утечка вероятна")

#     # 3) ArcFace similarity: orig vs hair_masked
#     emb_orig = arcface_embedder([pil])[0].float().cpu().numpy()
#     emb_hair = arcface_embedder([hair_masked])[0].float().cpu().numpy()

#     def cos(a,b):
#         a = a / (np.linalg.norm(a) + 1e-8)
#         b = b / (np.linalg.norm(b) + 1e-8)
#         return float((a*b).sum())

#     sim = cos(emb_orig, emb_hair)
#     print(f"[leak] arcface cos(orig, hair_masked) = {sim:.4f}")
