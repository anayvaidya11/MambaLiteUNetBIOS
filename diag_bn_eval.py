"""Diagnose the train/eval gap of a MambaLiteUNet checkpoint on the EsoOCT val split.

Reports val loss and pooled Dice for one checkpoint, computed four ways:
  A1  eval mode, batch 1              -- exactly what train_eso.py logged for that epoch
  A8  eval mode, batch 8              -- same weights, same running stats, batched
  B   BatchNorm using batch statistics (train-mode BN, no gradients, batch 8)
                                      -- what the network "saw" while training
  C   eval mode after recalibrating the BatchNorm running statistics on training batches

If A is bad and B/C are good, the gap is entirely the BatchNorm running statistics
(they go stale in the attention gates while the learning rate is high). The script also
prints each attention gate's mask BatchNorm2d(1) running mean/var before and after
recalibration -- that layer feeds the sigmoid that gates the skip connection.

Spark, env mambalite, repo root (data prepared in ./data/EsoOCT/):
    python diag_bn_eval.py --ckpt results/<run>/checkpoints/latest.pth
"""
import argparse
import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def pooled_dice(pred, gt, threshold):
    """Dice over all pixels of all images (same definition as engine.py's f1_or_dsc)."""
    p = pred >= threshold
    g = gt >= 0.5
    tp = (p & g).sum().item()
    fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item()
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def evaluate(model, loader, criterion, threshold, bn_batch_stats=False):
    """Mean loss and pooled Dice over `loader`. With bn_batch_stats, BatchNorm layers use
    the statistics of each batch instead of their running averages."""
    model.eval()
    if bn_batch_stats:
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                m.train()
    losses, preds, gts = [], [], []
    with torch.no_grad():
        for img, msk in loader:
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            losses.append(criterion(out, msk).item())
            preds.append(out.cpu())
            gts.append(msk.cpu())
    model.eval()
    return sum(losses) / len(losses), pooled_dice(torch.cat(preds), torch.cat(gts), threshold)


def mask_bn_stats(model):
    """{layer name: (running_mean, running_var)} for the single-channel BatchNorm2d layers
    that feed each attention gate's sigmoid mask."""
    return {name: (m.running_mean.item(), m.running_var.item())
            for name, m in model.named_modules()
            if isinstance(m, _BatchNorm) and m.num_features == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='latest.pth (training dict) or best-*.pth (state dict)')
    ap.add_argument('--recal-batches', type=int, default=40, help='training batches used to re-estimate BN stats')
    ap.add_argument('--batch', type=int, default=8, help='val batch size for A8/B/C')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from configs.config_setting_eso import setting_config as cfg
    from loader import isic_loader
    from utils import BceDiceLoss, set_seed
    from predict_eso import load_model
    from bn_recalib import recalibrate_bn

    set_seed(args.seed)
    model, threshold = load_model(args.ckpt)
    criterion = BceDiceLoss()
    val_set = isic_loader(path_Data=cfg.data_path, train=False)
    val1 = DataLoader(val_set, batch_size=1, shuffle=False)
    valb = DataLoader(val_set, batch_size=args.batch, shuffle=False)
    train_loader = DataLoader(isic_loader(path_Data=cfg.data_path, train=True),
                              batch_size=cfg.batch_size, shuffle=True)

    print(f'\ncheckpoint: {args.ckpt}')
    print('mask BatchNorm2d(1) running stats as stored (mean, var):')
    for name, (mu, var) in mask_bn_stats(model).items():
        print(f'  {name:32s} {mu:+.4f}  {var:.4f}')

    rows = []
    rows.append(('A1  eval mode, batch 1 (as logged)', *evaluate(model, val1, criterion, threshold)))
    rows.append((f'A{args.batch}  eval mode, batch {args.batch}', *evaluate(model, valb, criterion, threshold)))
    rows.append(('B   BN batch statistics (train-mode BN)',
                 *evaluate(copy.deepcopy(model), valb, criterion, threshold, bn_batch_stats=True)))
    recal = copy.deepcopy(model)
    recalibrate_bn(recal, train_loader, args.recal_batches)
    rows.append((f'C   eval mode after BN recalibration ({args.recal_batches} train batches)',
                 *evaluate(recal, valb, criterion, threshold)))

    print(f'\n{"mode":62s} {"val loss":>9s} {"val Dice":>9s}')
    for name, loss, dice in rows:
        print(f'{name:62s} {loss:9.4f} {dice:9.4f}')

    print('\nmask BatchNorm2d(1) running stats after recalibration (mean, var):')
    for name, (mu, var) in mask_bn_stats(recal).items():
        print(f'  {name:32s} {mu:+.4f}  {var:.4f}')


if __name__ == '__main__':
    main()
