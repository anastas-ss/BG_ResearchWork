import torch
import numpy as np
from PIL import Image

@torch.no_grad()
def hair_leakage_check_one(
    *,
    pil: Image.Image,
    hair_cond,
    face_app,
    arcface_embedder,
    out_prefix="leak",
):
    # Сохраняем hair-masked изображение, сформированное HairConditioner.
    hair_cond.debug_save = True
    _ = hair_cond([pil], out_dtype=torch.float16)
    hair_masked = Image.open("debug_hair.png").convert("RGB")

    pil.save(f"{out_prefix}_orig.png")
    hair_masked.save(f"{out_prefix}_hair_masked.png")

    # Проверяем, детектируется ли лицо на hair-masked изображении.
    img_bgr = np.array(hair_masked)[:, :, ::-1]
    faces = face_app.get(img_bgr)
    if len(faces) == 0:
        print("[leak] no face detected on hair_masked (good sign)")
        return

    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    x1,y1,x2,y2 = [int(v) for v in f.bbox]
    x1,y1 = max(0,x1), max(0,y1)
    crop = hair_masked.crop((x1,y1,x2,y2))
    crop.save(f"{out_prefix}_face_cut_from_hair.png")
    print("[leak] face detected on hair_masked -> визуальная утечка вероятна")

    # ArcFace similarity между исходным и hair-masked изображением.
    emb_orig = arcface_embedder([pil])[0].float().cpu().numpy()
    emb_hair = arcface_embedder([hair_masked])[0].float().cpu().numpy()

    def cos(a,b):
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float((a*b).sum())

    sim = cos(emb_orig, emb_hair)
    print(f"[leak] arcface cos(orig, hair_masked) = {sim:.4f}")
