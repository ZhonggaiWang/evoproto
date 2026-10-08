"""torchrun 8-GPU exact equivalence with unequal rank batches and rank 7 empty evidence.

Run: torchrun --standalone --nproc-per-node=8 test_online_confusion_ddp.py
Synthetic CAM evidence uses binary fractions so all pooling orders are exact.
No training/model/GT data or random numbers are involved.
"""
import copy
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from online_directed_confusion import OnlineDirectedConfusion


def batch(rank, tick, device):
    n = rank % 3 + 1
    labels = torch.tensor([[1, 2, 3], [0, 255, 1]], dtype=torch.long, device=device)
    labels = labels[None].repeat(n, 1, 1)
    # Last rank still processes images but has no accepted/broad pseudo anchors.
    if rank == 7:
        labels.fill_(255)
    predictions = (torch.arange(n * 6, device=device).reshape(n, 2, 3) + rank + tick) % 4
    logits = F.one_hot(predictions, 4).permute(0, 3, 1, 2).float() * 8
    cams = torch.full((n, 3, 2, 3), .125, device=device)
    for class_id in range(1, 4):
        cams[:, class_id - 1] = torch.where(labels == class_id, .875, .125)
    cams = torch.where((labels == 0)[:, None], cams.new_tensor(.125), cams)
    # Class 3 is absent on the middle update, exercising stale-row persistence.
    if tick == 1:
        labels[labels == 3] = 255
    # Reject one otherwise valid foreground pixel using the actual image box.
    boxes = [[0, 2, 0, 3] for _ in range(n)]
    if rank % 2 == 1:
        boxes[-1] = [0, 1, 0, 3]
    return logits, labels, cams, boxes


def combined(tick, device):
    parts = [batch(rank, tick, device) for rank in range(8)]
    return tuple(torch.cat([part[i] for part in parts]) for i in range(3)) + (
        [box for part in parts for box in part[3]],)


def assert_exact(actual, expected):
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key in actual_state:
        a, b = actual_state[key], expected_state[key]
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda m: key + ': ' + m)
        else:
            assert a == b, key
    assert actual.export() == expected.export()


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, 'This test is explicitly an 8-rank contract.'
    device = torch.device('cuda', local_rank)
    estimator = OnlineDirectedConfusion(4, momentum=.5, stage=2).to(device)
    reference = OnlineDirectedConfusion(4, momentum=.5, stage=2).to(device)
    for tick in range(3):
        estimator.update(*batch(rank, tick, device), synchronize=True)
        reference.update(*combined(tick, device), synchronize=False)
        assert_exact(estimator, reference)
        exported = estimator.export()
        gathered = [None] * world
        dist.all_gather_object(gathered, exported)
        assert all(value == exported for value in gathered), 'Rank states differ'
        if tick == 1:
            # Resume both global and reference observers, then continue streaming.
            restored = OnlineDirectedConfusion(4, momentum=.5, stage=2).to(device)
            restored.load_state_dict(copy.deepcopy(estimator.state_dict()))
            restored_reference = OnlineDirectedConfusion(4, momentum=.5, stage=2).to(device)
            restored_reference.load_state_dict(copy.deepcopy(reference.state_dict()))
            estimator, reference = restored, restored_reference
            assert_exact(estimator, reference)
    if rank == 0:
        print(json.dumps({'test': 'online_confusion_8rank_global_and_resume_equivalence',
                          'ranks': world, 'passed': True, 'updates': int(estimator.updates),
                          'seen_images': int(estimator.seen_images),
                          'rank_batch_sizes': [r % 3 + 1 for r in range(8)],
                          'zero_evidence_rank': 7, 'all_tensor_states_exact': True}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
