"""
PyTorch dataset classes for loading the datasets.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable
from tqdm.auto import tqdm
from pathlib import Path
import einops
import zlib

BASE_RESOLUTION = 32


class InterventionalPongDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'background': 'categ_2',
        'ball-vel-dir': 'angle',
        'ball-vel-magn': 'continuous_1',
        'ball-x': 'continuous_1',
        'ball-y': 'continuous_1',
        'paddle-left-y': 'continuous_1',
        'paddle-right-y': 'continuous_1',
        'score-left': 'categ_5',
        'score-right': 'categ_5'
    })

    def extract_sample_list(self, fname):
        fname = Path(fname)
        if not fname.is_absolute():
            fname = self.data_folder / fname
        with open(fname, 'r') as f:
            sample_list = sorted(f.readlines())
        sample_list = [self.data_folder / s.strip() for s in sample_list if s]
        return sample_list

    def __init__(
            self,
            data_folder,
            single_image=False,
            return_latents=False,
            return_masks=False,
            return_causal_graphs=False,
            occlusion_level=0,
            occlusion_type='black',
            triplet=False,
            seq_len=2,
            resolution=BASE_RESOLUTION,
            norm_range="-1-1",
            # contains a split definition as one or more text files,
            # each containing a list of data files relative to data_folder
            sample_list_path=None,
            **kwargs
        ):
        assert not triplet
        assert not single_image
        assert resolution % BASE_RESOLUTION == 0
        assert occlusion_type in {'black', 'random'}
        super().__init__()

        self.data_folder = Path(data_folder)
        if sample_list_path is not None:
            if isinstance(sample_list_path, str) or isinstance(sample_list_path, Path):
                data_files = self.extract_sample_list(sample_list_path)
            else:
                # support for lists of fnames
                data_files = []
                for path in sample_list_path:
                    data_files.extend(self.extract_sample_list(path))
            assert_msg = f'Could not find ComplexInterventionalPong dataset with the list(s) at {sample_list_path}'
        else:
            data_files = sorted(self.data_folder.rglob("*.npz"))
            assert_msg = f'Could not find ComplexInterventionalPong dataset at {data_folder}'
        assert len(data_files) > 0, assert_msg

        assert norm_range in {"0-1", "-1-1"}

        # first, get a sense of what interventions/environments we have
        interv_id_reference = dict()
        interv_pics_fnames = []
        for dfile in data_files:
            interv = dfile.parent.name
            if interv not in interv_id_reference:
                interv_id_reference[interv] = len(interv_id_reference)
                interv_pics_fnames.append(dfile.parent.with_suffix(".png"))
        if "intervention0" in interv_id_reference:
            assert interv_id_reference["intervention0"] == 0, "If we sort the glob, this should be guaranteed..."

        # first, find out all the dimensions and allocate everything
        images_list = []
        masks_list = []
        latents_list = []
        interv_ids_list = []
        causal_graphs_list = []
        images_raw_shape = None
        masks_raw_shape = None
        latents_raw_shape = None
        for dfile in data_files:
            try:
                arr = np.load(dfile)
                images = arr['images']
                masks = arr['masks']
                latents = arr['latents']
                causal_graphs = arr['causal_graphs']
            except (FileNotFoundError, KeyError, zlib.error):
                continue  # skip work-in-progress files
            if images_raw_shape is None:
                images_raw_shape = images.shape
            if masks_raw_shape is None:
                masks_raw_shape = masks.shape
                assert images_raw_shape[0] == masks_raw_shape[0]
            if latents_raw_shape is None:
                latents_raw_shape = latents.shape
                assert images_raw_shape[0] == latents_raw_shape[0]
            assert images.shape == images_raw_shape
            assert masks.shape == masks_raw_shape
            assert latents.shape == latents_raw_shape
            images = einops.rearrange(images, "(samples seqlen) ... -> samples seqlen ...", seqlen=seq_len)
            masks = einops.rearrange(masks, "(samples seqlen) ... -> samples seqlen ...", seqlen=seq_len)
            latents = einops.rearrange(latents, "(samples seqlen) ... -> samples seqlen ...", seqlen=seq_len)
            causal_graphs = einops.rearrange(causal_graphs, "(samples seqlen) ... -> samples seqlen ...", seqlen=seq_len)
            interv_ids = np.full((images.shape[0],), interv_id_reference[dfile.parent.name])
            images_list.append(images)
            masks_list.append(masks)
            latents_list.append(latents)
            interv_ids_list.append(interv_ids)
            causal_graphs_list.append(causal_graphs)
        the_images = einops.rearrange(images_list, "listaxis samples ... -> (listaxis samples) ...")
        the_images = the_images[..., :3]  # cut 4th channel, i.e. "ball velocity", for now
        the_masks = einops.rearrange(masks_list, "listaxis samples ... -> (listaxis samples) ...")
        the_latents = einops.rearrange(latents_list, "listaxis samples ... -> (listaxis samples) ...")
        the_interv_ids = einops.rearrange(interv_ids_list, "listaxis samples ... -> (listaxis samples) ...")
        the_causal_graphs = einops.rearrange(causal_graphs_list, "listaxis samples ... -> (listaxis samples) ...")

        self.imgs = torch.from_numpy(the_images)
        self.masks = torch.from_numpy(the_masks)
        self.latents = torch.from_numpy(the_latents)
        self.interv_ids = torch.from_numpy(the_interv_ids).to(int)
        self.causal_graphs = torch.from_numpy(the_causal_graphs)
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self.causal_vars = arr['causal_vars'].tolist()
        self._clean_up_data()
        print(f'Using the causal variables {self.causal_vars}')

        # save reverse lookup for causal_vars list, might need this soon
        self.causal_vars_idx = {var:i for (i, var) in enumerate(self.causal_vars)}

        # bring masks from [0,255] to [0,1] immediately (imgs are brought to [-1,1] in _prepare_images)
        self.masks = self.masks.float() / 255

        # sanity
        assert self.imgs.shape[-2] == BASE_RESOLUTION
        assert self.imgs.shape[-1] == BASE_RESOLUTION

        if resolution > BASE_RESOLUTION:
            upscale = resolution // BASE_RESOLUTION
            self.imgs = einops.repeat(self.imgs, "... w h -> ... (w w_up) (h h_up)", w_up=upscale, h_up=upscale)

        self.single_image = single_image
        self.return_latents = return_latents
        self.return_masks = return_masks
        self.occlusion_level = occlusion_level
        self.occlusion_type = occlusion_type
        self.return_causal_graphs = return_causal_graphs
        self.triplet = triplet
        self.norm_range = norm_range
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1
        self.interv_id_reference = interv_id_reference
        self.interv_pics_fnames = np.array(interv_pics_fnames)
        self.num_of_interventions = len(interv_id_reference)

    def _clean_up_data(self):
        # Push channels to PyTorch dimension
        self.imgs = einops.rearrange(self.imgs, "... width height channel -> ... channel width height")
        self.masks = einops.rearrange(self.masks, "... width height object -> ... object width height")

        all_latents = []
        keys_var_info = list(InterventionalPongDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                InterventionalPongDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            if key.endswith('-proj'):
                continue
            latent = self.latents[...,i]
            if key == 'ball_vel_magn' and latent.unique().shape[0] == 1:
                if key in InterventionalPongDataset.VAR_INFO:
                    InterventionalPongDataset.VAR_INFO.pop(key)
                continue
            if "vel" in key and "viz" in key or "intervention-on" in key:
                continue  # TODO aggiungere le nostre chiavi a VAR_INFO?
            if InterventionalPongDataset.VAR_INFO[key].startswith('continuous'):
                if key.endswith('-x') or key.endswith('-y'):
                    latent = latent / 16.0 - 1.0
                else:
                    latent = latent - 2.0
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        raise NotImplementedError('Unused for now, since this means "Apply a FROZEN, PRETRAINED encoder"')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:  # se usata, trasformare in un generico len > 4
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            if self.norm_range == "-1-1":
                imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_vars(self):
        return len(self.causal_vars)

    def target_names(self):
        return self.causal_vars

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]
    
    def get_interv_pic_fname(self, ids):
        npstr = self.interv_pics_fnames[ids]
        if isinstance(ids, Iterable):
            return [str(n) for n in npstr]
        else:
            return str(npstr)
    
    def has_occlusion(self):
        return self.occlusion_level > 0

    # nothing but a utility for gifify
    def first_for_intervention(self, interv_name):
        interv_id = self.interv_id_reference[interv_name]
        # find idx
        found_idx = None
        for idx in range(len(self)):
            if self.interv_ids[idx] == interv_id:
                found_idx = idx
                break
        # return data sample
        return self[found_idx]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        returns = {}

        img_seq = self.imgs[idx]
        msk_seq = self.masks[idx]
        pos = self.latents[idx]
        interv_id = self.interv_ids[idx]
        cg_seq = self.causal_graphs[idx]

        if self.single_image:
            img_seq = img_seq[0]
            msk_seq = msk_seq[0]
            pos = pos[0]
            cg_seq = cg_seq[0]
        img_seq = self._prepare_imgs(img_seq)
        returns["pixel_values"] = img_seq

        if self.return_latents:
            returns["latents"] = pos
        
        if self.return_masks:
            returns["masks"] = msk_seq

        if self.occlusion_level > 0:
            if self.occlusion_level == 1:
                occlusion_mask = _generate_single_occlusion_mask(img_seq.shape, occlusion_size=8)
            elif self.occlusion_level == 2:
                occlusion_mask = _generate_single_occlusion_mask(img_seq.shape, occlusion_size=12)
            elif self.occlusion_level == 3:
                occlusion_mask = _generate_single_occlusion_mask(img_seq.shape, occlusion_size=16)
            elif self.occlusion_level == 4:
                occlusion_mask = _generate_single_occlusion_mask(img_seq.shape, occlusion_size=24)

            returns["occlusion_mask"] = occlusion_mask
            returns["occluded_pixels"] = img_seq * occlusion_mask
            returns["occluded_masks"] = msk_seq * occlusion_mask
            if self.occlusion_type == 'random':
                # let's get FUNKY!
                rand = torch.rand_like(img_seq)
                if self.norm_range == "-1-1":
                    rand = rand * 2.0 - 1.0
                returns["occluded_pixels"] += rand * (1-occlusion_mask)
                for i in range(returns["occluded_masks"].shape[0]):
                    for j in range(returns["occluded_masks"].shape[1]):
                        if torch.count_nonzero(returns["occluded_masks"][i,j]) == 0:
                            # so that, once the object is fully hidden, the model actually sees the FUNK
                            # but doesn't have the clue of the mask's shape (as it would if it was given mask_seq)
                            returns["occluded_masks"][i,j] = occlusion_mask

        returns["interv_id"] = interv_id
        
        if self.return_causal_graphs:
            returns["causal_graphs"] = cg_seq

        return returns

def _generate_single_occlusion_mask(image_shape, occlusion_size):
    # taking advantage of the assumption of square images (see assert in __init__)
    occlusion_x, occlusion_y = torch.randint(0, image_shape[-1]-occlusion_size+1, (2,))
    occlusion_mask = torch.ones(image_shape[-2:])
    occlusion_mask[occlusion_y:(occlusion_y+occlusion_size), occlusion_x:(occlusion_x+occlusion_size)] = 0
    return occlusion_mask






def gifify(dataset_dir, interv_name, gifname):
    SEQ_LEN = 960
    UPSCALE_FACTOR = 4
    dataset = InterventionalPongDataset(dataset_dir, seq_len=SEQ_LEN, return_masks=True, return_causal_graphs=True)
    # dunno what this is for (yet), but I do know that if I turn it on
    # it won't normalize the images to the range [-1,1],
    # and that's good enough for this mini test
    dataset.encodings_active = True
    sample = dataset.first_for_intervention(interv_name)
    rgb = einops.rearrange(sample["pixel_values"][:,:3,:,:].numpy(), "time channel width height -> time width height channel")
    mask = einops.repeat(sample["masks"].numpy(), "time mask height width -> time height (mask width) c", c=3)
    mask = mask*255  # for the gif
    full_stuff = np.concat([rgb, mask], axis=2)
    full_stuff = full_stuff.repeat(UPSCALE_FACTOR, axis=1).repeat(UPSCALE_FACTOR, axis=2)  # blow it up
    full_stuff = full_stuff.astype(np.uint8)
    frame_list = [full_stuff[i] for i in range(full_stuff.shape[0])]
    import imageio
    imageio.mimwrite(Path(dataset_dir) / gifname, frame_list, format="gif")

if __name__ == "__main__":
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention0", 'a.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention1", 'b.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention2", 'c.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention16", 'd.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention32", 'e.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention192", 'f.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention256", 'g.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention448", 'w.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention96", 'x.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention3", 'y.gif')
    gifify("dataset_BIGBALLv5a_11envs_142reps_960frames", "intervention260", 'z.gif')
