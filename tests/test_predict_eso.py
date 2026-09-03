"""load_state must accept both checkpoint formats train_eso.py writes."""
import numpy as np
import torch

from predict_eso import load_state


def test_load_state_accepts_training_dict_with_numpy_scalars(tmp_path):
    weights = {'w': torch.ones(2)}
    path = tmp_path / 'latest.pth'
    torch.save({'epoch': 300, 'min_loss': np.float64(0.1674), 'loss': np.mean([0.5]),
                'model_state_dict': weights}, path)

    state = load_state(str(path))

    assert set(state) == {'w'}
    assert torch.equal(state['w'], weights['w'])


def test_load_state_accepts_plain_state_dict(tmp_path):
    weights = {'w': torch.ones(2)}
    path = tmp_path / 'best.pth'
    torch.save(weights, path)

    assert torch.equal(load_state(str(path))['w'], weights['w'])
