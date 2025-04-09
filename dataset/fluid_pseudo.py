import os
from PIL import Image
import torch
import numpy as np
from dataset.base import BaseDataset
from dataset.utils import pil_loader
from utils.utils import recursive_glob


class FluidPseudo(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set manually for now – no encoder
        self.full_res_shape = (512, 512)
        self.fx = 1.0
        self.fy = 1.0
        self.u0 = 256
        self.v0 = 256

    def prepare_filenames(self):
        # Assumes structure: root/train_img/, root/train_layer_mask/, root/train_fluid_mask/
        img_dir = os.path.join(self.root, f"{self.split}_img")
        files = sorted(recursive_glob(rootdir=img_dir))
        return files

    def get_image_path(self, index, offset=0):
        return self.files[index]["name"]

    def get_layer_path(self, index):
        base = os.path.basename(self.get_image_path(index)).rsplit('.', 1)[0]
        return os.path.join(self.root, f"{self.split}_layer_mask", f"{base}.png")

    def get_fluid_path(self, index):
        base = os.path.basename(self.get_image_path(index)).rsplit('.', 1)[0]
        return os.path.join(self.root, f"{self.split}_fluid_mask", f"{base}.png")

    def get_segmentation(self, index, do_flip):
        path = self.get_layer_path(index)
        lbl = pil_loader(path, self.width, self.height, is_segmentation=True)

        if do_flip:
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        return lbl

    def get_fluid_mask(self, index, do_flip):
        path = self.get_fluid_path(index)
        mask = pil_loader(path, self.width, self.height, is_segmentation=True)

        if do_flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        mask = np.array(mask, dtype=np.float32)
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)

        return torch.tensor(mask).unsqueeze(0)

    def __getitem__(self, index):
        inputs = super().__getitem__(index)
        inputs["layer_seg"] = inputs.pop("lbl")  # rename key
        inputs["fluid_seg"] = self.get_fluid_mask(index, do_flip=self.is_train)
        return inputs

