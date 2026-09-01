"""
Prepare the BIOS esophagus OCT dataset (nnU-Net Dataset515_EsoOCT2D) for MambaLiteUNet.

Reads the nnU-Net raw images/labels and the nnU-Net fold split, so the train/val
division matches the one used to train U-Mamba Enc (fair comparison). Writes the
six .npy files that loader.py expects into ./data/EsoOCT/.

The test split is a copy of the val split: the real held-out evaluation happens
later on the 11-patient val set, outside this pipeline.

Run on the machine that holds the nnU-Net data:
    python dataprepare/prepare_eso_oct.py
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

height = 256
width = 256
channels = 3


def find_file(folder, stem):
    matches = sorted(glob.glob(os.path.join(folder, stem + '.*')))
    if not matches:
        raise FileNotFoundError(f'no file for case {stem} in {folder}')
    return matches[0]


def load_image(path):
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:  # already multi-channel, keep first 3
        arr = arr[:, :, :3].mean(axis=2)
    arr = arr.astype(np.float64)
    if arr.max() > 255:  # 16-bit input, rescale to 0-255
        arr = arr / arr.max() * 255.0
    resized = Image.fromarray(arr.astype(np.uint8)).resize((width, height), Image.BILINEAR)
    gray = np.array(resized, dtype=np.float32)
    return np.stack([gray] * channels, axis=-1)  # grayscale -> 3 identical channels


def load_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    binary = (arr > 0).astype(np.uint8) * 255
    resized = Image.fromarray(binary).resize((width, height), Image.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def build_split(cases, images_dir, labels_dir):
    data = np.zeros([len(cases), height, width, channels], dtype=np.float32)
    mask = np.zeros([len(cases), height, width], dtype=np.uint8)
    for idx, case in enumerate(cases):
        data[idx] = load_image(find_file(images_dir, case + '_0000'))
        mask[idx] = load_mask(find_file(labels_dir, case))
        if (idx + 1) % 100 == 0 or idx + 1 == len(cases):
            print(f'  {idx + 1}/{len(cases)}')
    return data, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=os.path.expanduser('~/U-Mamba/data/nnUNet_raw/Dataset515_EsoOCT2D'))
    parser.add_argument('--splits', default=os.path.expanduser('~/U-Mamba/data/nnUNet_preprocessed/Dataset515_EsoOCT2D/splits_final.json'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--out', default='./data/EsoOCT/')
    args = parser.parse_args()

    with open(args.splits) as f:
        split = json.load(f)[args.fold]
    train_cases, val_cases = sorted(split['train']), sorted(split['val'])
    print(f'fold {args.fold}: {len(train_cases)} train, {len(val_cases)} val cases')

    images_dir = os.path.join(args.raw, 'imagesTr')
    labels_dir = os.path.join(args.raw, 'labelsTr')

    print('Reading train split')
    train_img, train_mask = build_split(train_cases, images_dir, labels_dir)
    print('Reading val split')
    val_img, val_mask = build_split(val_cases, images_dir, labels_dir)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'data_train'), train_img)
    np.save(os.path.join(args.out, 'mask_train'), train_mask)
    np.save(os.path.join(args.out, 'data_val'), val_img)
    np.save(os.path.join(args.out, 'mask_val'), val_mask)
    np.save(os.path.join(args.out, 'data_test'), val_img)   # placeholder, see docstring
    np.save(os.path.join(args.out, 'mask_test'), val_mask)

    with open(os.path.join(args.out, 'split_cases.json'), 'w') as f:
        json.dump({'fold': args.fold, 'train': train_cases, 'val': val_cases}, f, indent=1)

    print(f'train {train_img.shape} mask coverage {train_mask.mean():.3f}')
    print(f'val   {val_img.shape} mask coverage {val_mask.mean():.3f}')
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
