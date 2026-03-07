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
    Сравнивает три выхода при одном и том же стартовом шуме:
      1) только text,
      2) text + исходные hair-токены,
      3) text + hair-токены с вырезанной областью лица в hair-маске.
    """
    device = pipe.device
    bsz = len(pil_images)
    assert bsz > 0, "pil_images is empty"

    out_dir = Path(run_dir) / "hair_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = "a portrait photo of a person"
    tok = pipe.tokenizer(
        [prompt] * bsz,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    tok_uc = pipe.tokenizer(
        [""] * bsz,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    text_c = pipe.text_encoder(**tok).last_hidden_state.to(dtype_unet)
    text_u = pipe.text_encoder(**tok_uc).last_hidden_state.to(dtype_unet)

    hair_tokens = hair_cond(pil_images, out_dtype=dtype_unet)

    masks = hair_cond.enc_h(pil_images)
    fm = masks.clone()
    h, w = fm.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    yy = yy.to(fm.device)
    xx = xx.to(fm.device)
    cy, cx = h // 2, w // 2
    r = int(0.22 * h)
    face_disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < (r ** 2)
    fm[:, face_disk] = 0.0

    from src.model.hair_conditioner_parsing import apply_mask_to_pil

    hair_pil_face_off = [apply_mask_to_pil(im, fm[i], bg=hair_cond.bg_value) for i, im in enumerate(pil_images)]
    inputs = hair_cond.proc(images=hair_pil_face_off, return_tensors="pt").to(device)
    pooled = hair_cond.clip(**inputs).pooler_output
    hair_tokens_face_off = hair_cond.proj(pooled.float()).view(bsz, hair_cond.n_tokens, hair_cond.cross_dim).to(dtype=dtype_unet)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn((bsz, 4, 64, 64), device=device, dtype=dtype_unet, generator=gen)

    scheduler.set_timesteps(int(num_steps), device=device)
    latents = latents * scheduler.init_noise_sigma

    def run(enc_c, enc_u):
        x = latents.clone()
        for t in scheduler.timesteps:
            t_int = int(t.item())
            t_batch = torch.full((bsz,), t_int, device=device, dtype=torch.long)
            x_in = scheduler.scale_model_input(x, t) if hasattr(scheduler, "scale_model_input") else x
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps_u = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_u).sample
                eps_c = pipe.unet(x_in, t_batch, encoder_hidden_states=enc_c).sample
            eps = eps_u + float(cfg_scale) * (eps_c - eps_u)
            x = scheduler.step(eps, t, x).prev_sample
        imgs = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
        imgs = (imgs.float() * 0.5 + 0.5).clamp(0, 1)
        return imgs

    zeros_id = torch.zeros((bsz, 1, pipe.unet.config.cross_attention_dim), device=device, dtype=dtype_unet)

    enc_text_only_c = {"text": text_c, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}
    enc_text_only_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}

    enc_hair_on_c = {"text": text_c, "id": zeros_id, "hair": hair_tokens}
    enc_hair_on_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens)}

    enc_hair_faceoff_c = {"text": text_c, "id": zeros_id, "hair": hair_tokens_face_off}
    enc_hair_faceoff_u = {"text": text_u, "id": zeros_id, "hair": torch.zeros_like(hair_tokens_face_off)}

    img_text = run(enc_text_only_c, enc_text_only_u)[0]
    img_hair = run(enc_hair_on_c, enc_hair_on_u)[0]
    img_faceoff = run(enc_hair_faceoff_c, enc_hair_faceoff_u)[0]

    grid = torchvision.utils.make_grid(
        torch.stack([img_text.cpu(), img_hair.cpu(), img_faceoff.cpu()], dim=0),
        nrow=3,
    )
    out_path = out_dir / f"step_{step:07d}.png"
    torchvision.utils.save_image(grid, str(out_path))
    print(f"[hair_leakage] saved {out_path} (text_only | hair_on | hair_face_off)")
