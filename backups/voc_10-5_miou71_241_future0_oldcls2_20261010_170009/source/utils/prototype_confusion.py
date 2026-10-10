"""Paper Eqs. (3), (4), (6), (7), and the iteration-based refresh schedule."""
import math

import torch


@torch.no_grad()
def symmetric_confusion(counts):
    """Normalize GT-row/prediction-column counts, then add both directions.

    Empty reference rows have zero directional disagreement. Background is a
    class in the prototype matrix, so its row and column are retained.
    """
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("Confusion counts must be a square matrix.")
    counts = counts.detach().float()
    directional = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    directional.fill_diagonal_(0)
    return directional + directional.t()


@torch.no_grad()
def confusion_counterparts(confusion, old_count, temperature=0.5):
    """Select one opposite-group counterpart for every class and tanh weights.

    A symmetric score can yield nonreciprocal selections. Nonselected pairs
    have zero extra weight; zero-confusion ties also contribute zero weight.
    """
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion scores must be a square matrix.")
    classes = confusion.shape[0]
    if not 0 < old_count < classes:
        raise ValueError("Counterpart selection requires both old and new classes.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Confusion temperature K must be finite and positive.")
    confusion = confusion.detach()
    old_to_new = confusion[:old_count, old_count:].argmax(dim=1) + old_count
    new_to_old = confusion[old_count:, :old_count].argmax(dim=1)
    counterparts = torch.cat((old_to_new, new_to_old))
    rows = torch.arange(classes, device=confusion.device)
    weights = torch.zeros_like(confusion)
    weights[rows, counterparts] = torch.tanh(confusion[rows, counterparts] / temperature)
    return counterparts, weights


def confusion_update_due(completed_iterations, start_iterations=4000, interval=2000):
    """Refresh after the configured start, independently of loss warmup."""
    if start_iterations < 0 or interval <= 0:
        raise ValueError("Confusion start must be nonnegative and interval positive.")
    return (completed_iterations >= start_iterations
            and (completed_iterations - start_iterations) % interval == 0)
