"""
Predict epithelium masks with a trained MambaLiteUNet on a folder of 2D OCT frames.

Preprocessing is identical to training (dataprepare/prepare_eso_oct.py: grayscale ->
3 channels, 256x256 bilinear, /255). The predicted probability map is resized back to
the frame's original resolution, thresholded, and written as a uint8 {0,1} .tif named
like the input minus the nnU-Net `_0000` channel suffix -- the same layout that
nnUNetv2_predict produces in Output515_val, so the folder drops straight into the
val_eval pipeline.

Example (Spark, conda env mambalite, from the repo root):
    python predict_eso.py \
        --ckpt results/<run>/checkpoints/best-epoch255-loss0.1674.pth \
        --input ~/U-Mamba/Inference/Input515_val \
        --output results/<run>/val_pred
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

from dataprepare.prepare_eso_oct import load_image

FRAME_EXTS = ('.tif', '.tiff', '.png')


def list_frames(input_dir):
    files = []
    for ext in FRAME_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, '*' + ext)))
    return sorted(files)


def output_name(path):
    """`pat08_sq2_0000_0000.tif` -> `pat08_sq2_0000` (strip the nnU-Net channel suffix)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith('_0000'):
        stem = stem[:-5]
    return stem


def original_size(path):
    with Image.open(path) as img:
        return img.size[1], img.size[0]  # (H, W)


def preprocess(path):
    """(3, 256, 256) float32 in [0, 1], exactly what the training loader fed the model."""
    arr = load_image(path) / 255.0  # (256, 256, 3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)


def postprocess(prob, hw, threshold):
    """(256, 256) probability map -> (H, W) uint8 {0,1} mask at the original frame size."""
    h, w = hw
    up = Image.fromarray(prob.astype(np.float32), mode='F').resize((w, h), Image.BILINEAR)
    return (np.array(up) >= threshold).astype(np.uint8)


def load_state(ckpt_path):
    """Model state dict from either checkpoint format train_eso.py writes: a plain state
    dict (best-*.pth) or a training dict (latest.pth). The training dict holds numpy
    scalars, which torch>=2.6 refuses under the default weights_only=True; these are our
    own files, so they are loaded as trusted."""
    import torch
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    return state


def load_model(ckpt_path):
    from models.MambaLiteUNet import MambaLiteUNet
    from configs.config_setting_eso import setting_config as cfg

    mc = cfg.model_config
    model = MambaLiteUNet(num_classes=mc['num_classes'],
                          input_channels=mc['input_channels'],
                          c_list=mc['c_list'])
    model.load_state_dict(load_state(ckpt_path), strict=True)
    return model.cuda().eval(), cfg.threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='state dict, e.g. results/<run>/checkpoints/best-*.pth')
    ap.add_argument('--input', required=True, help='folder of 2D frames (.tif/.png)')
    ap.add_argument('--output', required=True, help='folder for the predicted masks')
    ap.add_argument('--threshold', type=float, default=None, help='default: config threshold (0.5)')
    args = ap.parse_args()

    frames = list_frames(args.input)
    if not frames:
        raise SystemExit(f'no frames found in {args.input}')

    import torch
    model, cfg_threshold = load_model(args.ckpt)
    threshold = cfg_threshold if args.threshold is None else args.threshold
    os.makedirs(args.output, exist_ok=True)
    print(f'{len(frames)} frames, checkpoint {args.ckpt}, threshold {threshold}')

    coverage = []
    with torch.no_grad():
        for i, path in enumerate(frames):
            x = torch.from_numpy(preprocess(path))[None].cuda()
            prob = model(x)[0, 0].cpu().numpy()
            mask = postprocess(prob, original_size(path), threshold)
            Image.fromarray(mask).save(os.path.join(args.output, output_name(path) + '.tif'))
            coverage.append(mask.mean())
            if (i + 1) % 25 == 0 or i + 1 == len(frames):
                print(f'  {i + 1}/{len(frames)}')

    print(f'done: {len(frames)} masks in {args.output}, mean foreground fraction {np.mean(coverage):.4f}')


if __name__ == '__main__':
    main()
