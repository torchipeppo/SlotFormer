# earlier I ran: python slotformer/base_slots/extract_phyre_slots.py --params slotformer/base_slots/configs/savi_phyre_params-fold0.py --weight pretrained/savi_phyre_params-fold0/model_30.pth --save_path data/PHYRE --vid_len 11 --split -1


import numpy as np
import scipy
import imageio
import einops
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

REPO_ROOT = Path(__file__).parent.parent

# x = np.load(REPO_ROOT / "data/PHYRE/slots/savi_phyre_params-fold0/within-fold_0-train-data_0.1-pos_0.2/002214.npy")
# print(x)
# print(x.shape)

def make_masked_videos(video, attns):
    # unsqueeze an extra axis to align slot and channel as different dimensions,
    # allowing to broadcast both tensors in a single operation later.
    # Credits to the STEVE code [Singh2022] for the idea
    video = einops.rearrange(video, "t c h w -> t 1 c h w")
    attns = einops.rearrange(attns, "t s h w -> t s 1 h w")
    # Let's-a go
    masked = video*attns + 0.5*(1.0-attns)
    return masked

def float2byte(vec):
    return np.array(vec*255, dtype=np.uint8)

def save_gif(video, masked, fpath):
    video = einops.rearrange(video, "time channel height width -> time height width channel")
    masked = einops.rearrange(masked, "time slot channel height width -> time slot height width channel")
    TIME, SLOTS, HEIGHT, WIDTH, CHANNELS = masked.shape

    frames = []
    for t in range(TIME):
        frame = np.empty((HEIGHT, WIDTH*(SLOTS+1), CHANNELS))
        frame[:, :WIDTH, :] = video[t]
        for s in range(SLOTS):
            frame[:, (s+1)*WIDTH:(s+2)*WIDTH, :] = masked[t,s]
        frames.append(float2byte(frame))

    imageio.mimwrite(fpath, frames, format="gif")


paths = list((REPO_ROOT / "data/PHYRE_trained_by_me/slots/savi_phyre_params-fold0").rglob("*-attn.npy"))
for attn_path in tqdm.tqdm(paths):
    idx = attn_path.stem.split("-")[0]
    img_path = attn_path.with_name(f"{idx}-img.npy")
    img = np.load(img_path)

    attn = np.load(attn_path)
    attn_hw = int(np.sqrt(attn.shape[1]))
    attn = einops.rearrange(attn, "time (height width) slot -> time slot height width", height=attn_hw, width=attn_hw)
    # only zoom in the images, not time or slots
    attn = scipy.ndimage.zoom(attn, zoom=[1,1,2,2], mode="nearest")

    masked = make_masked_videos(img, attn)

    attn_dir = attn_path.parent
    gif_dir = attn_dir.with_name(attn_dir.name + "__GIF")
    gif_dir.mkdir(exist_ok=True)
    save_gif(img, masked, gif_dir/f"{idx}.gif")
