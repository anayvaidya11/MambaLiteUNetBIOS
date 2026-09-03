"""Tests for plot_loss.parse_log (log format of engine.py / utils.get_logger)."""
from plot_loss import parse_log

HEADER = """2026-09-04 09:59:58 - network: MambaLiteUNet
2026-09-04 09:59:58 - epochs: 300
2026-09-04 09:59:58 - T_max: 300
"""

NEW_LOG = HEADER + """2026-09-04 10:00:01 - train: epoch 1, iter:0, loss: 0.9000, lr: 0.001
2026-09-04 10:00:05 - train: epoch 1, iter:20, loss: 0.7000, lr: 0.001
2026-09-04 10:00:12 - train: epoch 1, epoch_mean_loss: 0.6500, lr: 0.001
2026-09-04 10:00:15 - val epoch: 1, loss: 0.5654, miou: 0.4, f1_or_dsc: 0.55, accuracy: 0.9,                 specificity: 0.99, sensitivity: 0.5, confusion_matrix: [[1 2]
 [3 4]]
2026-09-04 10:00:20 - train: epoch 2, iter:0, loss: 0.5000, lr: 0.00099
2026-09-04 10:00:30 - train: epoch 2, epoch_mean_loss: 0.4800, lr: 0.00099
2026-09-04 10:00:33 - val epoch: 2, loss: 0.4000
"""

OLD_LOG = HEADER + """2026-09-02 10:57:30 - train: epoch 1, iter:0, loss: 10.3600, lr: 0.001
2026-09-02 10:57:40 - train: epoch 1, iter:60, loss: 1.0300, lr: 0.001
2026-09-02 10:57:44 - val epoch: 1, loss: 0.5654
2026-09-02 10:57:55 - train: epoch 2, iter:0, loss: 0.8000, lr: 0.000999
2026-09-02 10:58:05 - train: epoch 2, iter:60, loss: 0.6100, lr: 0.000999
2026-09-02 10:58:09 - val epoch: 2, loss: 0.5000
"""


def test_new_format_uses_epoch_mean_loss_and_reads_dice():
    r = parse_log(NEW_LOG.splitlines())
    assert r['epoch'] == [1, 2]
    assert r['train_loss'] == [0.65, 0.48]
    assert r['val_loss'] == [0.5654, 0.4]
    assert r['val_dice'] == [0.55, None]
    assert r['lr'] == [0.001, 0.00099]


def test_old_format_falls_back_to_last_logged_iter_mean():
    r = parse_log(OLD_LOG.splitlines())
    assert r['epoch'] == [1, 2]
    assert r['train_loss'] == [1.03, 0.61]
    assert r['val_loss'] == [0.5654, 0.5]
    assert r['val_dice'] == [None, None]
    assert r['lr'] == [0.001, 0.000999]


def test_epoch_without_val_line_is_still_reported():
    lines = HEADER.splitlines() + [
        '2026-09-04 10:00:12 - train: epoch 1, epoch_mean_loss: 0.6500, lr: 0.001',
    ]
    r = parse_log(lines)
    assert r['epoch'] == [1]
    assert r['train_loss'] == [0.65]
    assert r['val_loss'] == [None]
