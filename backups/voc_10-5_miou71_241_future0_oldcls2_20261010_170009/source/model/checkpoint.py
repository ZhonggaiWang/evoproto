"""Load an actual previous stage of the patch-prototype architecture."""
import torch


def load_incremental_checkpoint(model, model_old, path, step):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith(("decoder.conv6.", "decoder.conv7.", "decoder.conv8.")) for key in state):
        raise RuntimeError(
            "This checkpoint contains a LargeFOV decoder. Patch-token prototypes "
            "use the ViT feature space; retrain step0 with the prototype-only model "
            "and use its previous-stage checkpoint."
        )
    # Legacy fixed-temperature checkpoints retain their original predictions.
    # Fresh runs use the new configured initial scale (default 100).
    legacy_temperature = state.pop('decoder._temperature', None)
    if legacy_temperature is not None:
        if not torch.isfinite(legacy_temperature).all() or not (legacy_temperature > 0).all():
            raise RuntimeError('Legacy prototype temperature must be finite and positive.')
        state.setdefault('decoder.logit_scale', -legacy_temperature.log())
    allowed_new = (f"decoder.class_prototypes.{step}.",
                   f"classifier.{step}.", f"aux_classifier.{step}.")
    # Check both models before copying any tensors. Old prototypes must be
    # present and have the correct dimension; only the new stage may be absent.
    for name, target, allowed_missing in (
        ("teacher", model_old, ()), ("student", model, allowed_new),
    ):
        expected = target.state_dict()
        missing = [key for key in expected if key not in state and not key.startswith(allowed_missing)]
        unexpected = [key for key in state if key not in expected]
        mismatched = [key for key in state if key in expected and state[key].shape != expected[key].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"Invalid previous prototype checkpoint for {name}: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
    model_old.load_state_dict(state, strict=True)
    model.load_state_dict(state, strict=False)
