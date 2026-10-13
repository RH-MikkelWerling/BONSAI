import torch

from opera.run.inspect_router_source_optimizer import adam_adaptive_update


def test_saved_adam_moments_reconstruct_update_and_reveal_epsilon_attenuation():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4, eps=1e-6, weight_decay=0)
    parameter.grad = torch.tensor([1e-8, 1e-4])
    optimizer.step()
    update, factor = adam_adaptive_update(optimizer.state[parameter], optimizer.param_groups[0])
    assert torch.allclose(parameter.detach(), -update, atol=1e-9, rtol=1e-5)
    assert factor[0] < .02
    assert factor[1] > .98
