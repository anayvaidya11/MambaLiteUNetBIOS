"""Plot train/val loss, val Dice and learning rate per epoch from a train.info.log.

Runs anywhere with matplotlib (e.g. on the Mac after scp-ing the log from the Spark):
    python plot_loss.py results/<run>/log/train.info.log --out loss_curve.png

Train loss is the epoch mean when the log has the `epoch_mean_loss` summary line
(engine.py after 2026-09-03); older logs fall back to the running mean at the last
logged iteration of the epoch.
"""
import argparse
import os
import re

TRAIN_ITER = re.compile(r'train: epoch (\d+), iter:(\d+), loss: ([\d.]+), lr: ([\d.eE+-]+)')
TRAIN_EPOCH = re.compile(r'train: epoch (\d+), epoch_mean_loss: ([\d.]+), lr: ([\d.eE+-]+)')
VAL = re.compile(r'val epoch: (\d+), loss: ([\d.]+)(?:.*?f1_or_dsc: ([\d.]+))?')


def parse_log(lines):
    """Return per-epoch lists: epoch, train_loss, val_loss, val_dice, lr (None if missing)."""
    epochs = {}
    for line in lines:
        m = TRAIN_EPOCH.search(line)
        if m:
            d = epochs.setdefault(int(m.group(1)), {})
            d['train_loss'], d['lr'], d['mean'] = float(m.group(2)), float(m.group(3)), True
            continue
        m = TRAIN_ITER.search(line)
        if m:
            d = epochs.setdefault(int(m.group(1)), {})
            if not d.get('mean'):
                d['train_loss'], d['lr'] = float(m.group(3)), float(m.group(4))
            continue
        m = VAL.search(line)
        if m:
            d = epochs.setdefault(int(m.group(1)), {})
            d['val_loss'] = float(m.group(2))
            d['val_dice'] = float(m.group(3)) if m.group(3) else None
    keys = sorted(epochs)
    return {
        'epoch': keys,
        'train_loss': [epochs[e].get('train_loss') for e in keys],
        'val_loss': [epochs[e].get('val_loss') for e in keys],
        'val_dice': [epochs[e].get('val_dice') for e in keys],
        'lr': [epochs[e].get('lr') for e in keys],
    }


# Colours: validated categorical slots 1-3 of the dataviz palette + secondary ink.
BLUE, ORANGE, AQUA, INK2, MUTED, GRID, AXIS = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'


def _style(ax):
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def _series(epochs, values):
    xs = [e for e, v in zip(epochs, values) if v is not None]
    ys = [v for v in values if v is not None]
    return xs, ys


def plot(data, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_dice, ax_lr) = plt.subplots(
        3, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.6, 1.1]})
    fig.patch.set_facecolor('#fcfcfb')

    ep = data['epoch']
    xs, ys = _series(ep, data['train_loss'])
    ax_loss.plot(xs, ys, color=BLUE, linewidth=1.6, label='Train loss (epoch mean)')
    xs, ys = _series(ep, data['val_loss'])
    ax_loss.plot(xs, ys, color=ORANGE, linewidth=1.6, label='Validation loss')
    if ys:
        i_best = min(range(len(ys)), key=ys.__getitem__)
        ax_loss.plot(xs[i_best], ys[i_best], 'o', color=ORANGE, markersize=7,
                     markeredgecolor='#fcfcfb', markeredgewidth=1.5)
        ax_loss.annotate(f'best val {ys[i_best]:.4f} @ epoch {xs[i_best]}',
                         (xs[i_best], ys[i_best]), xytext=(0, 26), textcoords='offset points',
                         ha='center', fontsize=9, color=INK2,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fcfcfb', edgecolor=AXIS, linewidth=0.8),
                         arrowprops=dict(arrowstyle='-', color=AXIS, linewidth=0.8))
    ax_loss.set_ylabel('BCE + Dice loss')
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax_loss.set_title(title, fontsize=12, color='#0b0b0b', loc='left')

    xs, ys = _series(ep, data['val_dice'])
    if ys:
        ax_dice.plot(xs, ys, color=AQUA, linewidth=1.6)
        ax_dice.set_ylim(0, 1)
    else:
        ax_dice.text(0.5, 0.5, 'no per-epoch Dice in this log (val_interval > 1)',
                     transform=ax_dice.transAxes, ha='center', va='center', fontsize=9, color=MUTED)
    ax_dice.set_ylabel('Validation Dice')

    xs, ys = _series(ep, data['lr'])
    ax_lr.plot(xs, ys, color=INK2, linewidth=1.6)
    ax_lr.set_yscale('log')
    ax_lr.set_ylabel('Learning rate')
    ax_lr.set_xlabel('Epoch')

    for ax in (ax_loss, ax_dice, ax_lr):
        _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', help='path to results/<run>/log/train.info.log')
    ap.add_argument('--out', default=None, help='output PNG (default: <run>/loss_curve.png)')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    with open(args.log) as f:
        data = parse_log(f)
    if not data['epoch']:
        raise SystemExit(f'no train/val lines found in {args.log}')
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.log)))
    out = args.out or os.path.join(run_dir, 'loss_curve.png')
    title = args.title or f'{os.path.basename(run_dir)}: loss vs epoch'
    plot(data, out, title)
    vl = [v for v in data['val_loss'] if v is not None]
    print(f'{len(data["epoch"])} epochs parsed, best val loss {min(vl):.4f} -> {out}' if vl else f'{len(data["epoch"])} epochs -> {out}')


if __name__ == '__main__':
    main()
