"""Explicit learning-rate changes must preserve real Adam continuation state."""
from copy import deepcopy

import pytest
import torch

from qwen_exo_booster.native_mid_lora_experiment import (
    _load_training_state, _restore_training_state, _save_training_state,
)


def make_optimizer(parameter, lr):
    return torch.optim.AdamW([parameter], lr=lr, betas=(.9, .999), eps=1e-8,
                             weight_decay=0., foreach=False)


@pytest.fixture
def checkpoint(tmp_path):
    p = torch.nn.Parameter(torch.tensor([.7, -.2]))
    optimizer = make_optimizer(p, 1e-5)
    for gradient in ([.3, -.4], [-.2, .6]):
        p.grad = torch.tensor(gradient)
        optimizer.step()
    adapter = tmp_path / 'adapter.bin'
    adapter.write_bytes(b'synthetic-adapter-binding')
    companion = tmp_path / 'training.pt'
    training = {'learning_rate': 1e-5, 'prepared_sha256': 'original-data', 'seed': 193}
    progress = {'arm': 'self_think_sft', 'optimizer_step': 49,
                'optimizer_session_step': 2, 'next_sample_index': 2}
    _save_training_state(companion, optimizer=optimizer, adapter_path=adapter,
                         parameter_names=['p'], progress=progress, training=training)
    return p, optimizer, adapter, companion, training


def load(checkpoint, lr, **kwargs):
    _, _, adapter, companion, training = checkpoint
    return _load_training_state(companion, adapter_path=adapter,
        expected_training={**training, 'learning_rate': lr},
        optimizer_step=49, next_sample_index=2, **kwargs)


def test_explicit_lr_change_matches_uninterrupted_adam_and_normal_resume(checkpoint, tmp_path):
    original, control, adapter, _, _ = checkpoint
    state = load(checkpoint, 1.5e-5, override_learning_rate=True)
    resumed = torch.nn.Parameter(original.detach().clone())
    optimizer = make_optimizer(resumed, 1.5e-5)
    rng = state['rng']['torch_cpu'].clone()
    assert _restore_training_state(state, optimizer, ['p'], override_learning_rate=True) == 2
    assert torch.equal(torch.get_rng_state(), rng)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(optimizer.state[resumed][key], control.state[original][key])
    assert optimizer.param_groups[0]['lr'] == 1.5e-5
    assert state['optimizer']['param_groups'][0]['lr'] == 1e-5
    control.param_groups[0]['lr'] = 1.5e-5
    for p in (original, resumed):
        p.grad = torch.tensor([.8, -.1])
    control.step()
    optimizer.step()
    torch.testing.assert_close(original, resumed, rtol=0, atol=0)
    companion = tmp_path / 'new.training.pt'
    _save_training_state(companion, optimizer=optimizer, adapter_path=adapter,
        parameter_names=['p'], progress={'arm': 'self_think_sft', 'optimizer_step': 50,
        'optimizer_session_step': 3, 'next_sample_index': 3},
        training={'learning_rate': 1.5e-5, 'prepared_sha256': 'original-data', 'seed': 193})
    saved = _load_training_state(companion, adapter_path=adapter,
        expected_training={'learning_rate': 1.5e-5, 'prepared_sha256': 'original-data', 'seed': 193},
        optimizer_step=50, next_sample_index=3)
    restored = make_optimizer(torch.nn.Parameter(resumed.detach().clone()), 1.5e-5)
    assert _restore_training_state(saved, restored, ['p']) == 3


def test_implicit_lr_change_rejected(checkpoint):
    with pytest.raises(ValueError, match='learning_rate'):
        load(checkpoint, 1.5e-5)
    state = load(checkpoint, 1e-5)
    optimizer = make_optimizer(torch.nn.Parameter(checkpoint[0].detach().clone()), 1.5e-5)
    with pytest.raises(ValueError, match='lr'):
        _restore_training_state(state, optimizer, ['p'])
    assert not optimizer.state


@pytest.mark.parametrize('lr', [0., -1e-5, float('nan'), float('inf'), .01, 1e-5])
def test_invalid_or_unchanged_override_rejected(checkpoint, lr):
    with pytest.raises(ValueError, match='Learning-rate override'):
        load(checkpoint, lr, override_learning_rate=True)


def test_lr_override_does_not_admit_other_hyperparameter_changes(checkpoint):
    state = load(checkpoint, 1.5e-5, override_learning_rate=True)
    optimizer = make_optimizer(torch.nn.Parameter(checkpoint[0].detach().clone()), 1.5e-5)
    optimizer.param_groups[0]['betas'] = (.8, .999)
    with pytest.raises(ValueError, match='betas'):
        _restore_training_state(state, optimizer, ['p'], override_learning_rate=True)
    assert not optimizer.state


def test_saved_rate_binding_tampering_rejected(checkpoint):
    state = deepcopy(load(checkpoint, 1.5e-5, override_learning_rate=True))
    state['optimizer']['param_groups'][0]['lr'] = 2e-5
    optimizer = make_optimizer(torch.nn.Parameter(checkpoint[0].detach().clone()), 1.5e-5)
    with pytest.raises(ValueError, match='disagrees'):
        _restore_training_state(state, optimizer, ['p'], override_learning_rate=True)
    assert not optimizer.state
