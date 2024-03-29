import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    random_rotate=True,
    anomaly=False,
    infinte_loop=True,
    name='MVTecAD',
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    anom_gt = None
    if anomaly:
        if name == 'MVTecAD':
            anom_gt = [int('good' not in path) for path in all_files]
            # load gt mask
            gt_mask_path = []
            for path in all_files:
                gt_path = None
                if 'good' not in path:
                    gt_path = path.replace('test', 'ground_truth')
                    gt_path = gt_path.replace('.png', '_mask.png')
                gt_mask_path.append(gt_path)
        elif name == 'btad':
            anom_gt = [int('ok' not in path) for path in all_files]
            # load gt mask
            gt_mask_path = []
            for path in all_files:
                gt_path = None
                if 'ok' not in path:
                    gt_path = path.replace('test', 'ground_truth')
                    # gt_path = gt_path.replace('.bmp', '.png')
                gt_mask_path.append(gt_path)
            
    dataset = AnomalyImageDataset(
        image_size,
        all_files,
        anom_gt,
        gt_mask_path,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        random_rotate=random_rotate,
        name=name,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    # if not infinte_loop:
    return loader
    # else:
        # while True:
            # yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class AnomalyImageDataset(ImageDataset):

    def __init__(
        self,
        resolution,
        image_paths,
        anom_gt,
        gt_mask_path,
        random_rotate,
        name='mvtec',
        **kwargs,
    ):
        super().__init__(resolution, image_paths, **kwargs)
        self.name = name
        self.anom_gt = anom_gt
        self.gt_mask_path = gt_mask_path
        self.random_rotate = random_rotate
        if self.name == 'MVTecAD':
            self.classnames = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut","pill",  "screw", "tile", "toothbrush", "transistor", "wood", "zipper"] 
        elif self.name == 'btad':
            self.classnames = ["01", "02", "03"]
        self.local_classes = [self.get_label(x) for x in self.local_images]

    def get_label(self, img_name):
        for y, name in enumerate(self.classnames):
            if name in img_name:
                return y
        raise RuntimeError("{} unknown class".format(img_name))


    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        gt_path = self.gt_mask_path[idx]
        if gt_path is not None:
            with bf.BlobFile(gt_path, "rb") as f:
                pil_mask = Image.open(f)
                pil_mask.load()
            pil_mask = pil_mask.convert("L")
        else:
            pil_mask = Image.fromarray(np.zeros(pil_image.size).astype(np.uint8)).convert("L")

        arr = center_crop_arr(pil_image, self.resolution)
        mask = center_crop_arr(pil_mask, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            mask = mask[:, ::-1]
        if self.random_rotate:
            ang = (random.random() - 0.5) * 10.0 # / 180.0 * np.pi
            pil_image = pil_image.rotate(ang, Image.BILINEAR, expand = False)
            pil_mask = pil_mask.rotate(ang, Image.BILINEAR, expand = False)

        arr = arr.astype(np.float32) / 127.5 - 1
        # arr = (arr.astype(np.float32)/255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        out_dict["anom_gt"] = np.array(self.anom_gt[idx], dtype=np.int64)
        out_dict["img_path"] = '_'.join(path.split('/')[-2:])

        return np.transpose(arr, [2, 0, 1]), mask, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
