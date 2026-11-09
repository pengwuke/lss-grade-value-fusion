"""Architecture reconstruction for inference; never imports a training controller."""

import torch
from torch import nn

from .artifacts import checkpoint_path, find_entry
from .cnn import RiskAwareOrdinalCNN, cumulative_and_class_probabilities


CNN_FAMILIES = {
    "grade_value_coral_selected", "A0_NATURAL", "A1_SCALED",
    "A2_BINARY_P", "A3_GRADE_FIXED",
}
REFERENCE_FAMILIES = {"p4c_resnet18_corn_ema", "sr1_resnet18_corn_ema"}


def _attach_training_only_scalar_heads(model, state):
    """Retain train-only Linear(4,1) tensors without using them for predictions."""
    base = set(model.state_dict())
    extra = set(state) - base
    leaves = {name.rsplit(".", 1)[0] for name in extra}
    added = []
    for prefix in sorted(leaves):
        weight, bias = prefix + ".weight", prefix + ".bias"
        if weight not in extra or bias not in extra:
            raise ValueError("Unexpected non-linear training-only state: " + prefix)
        if tuple(state[weight].shape) != (1, 4) or tuple(state[bias].shape) != (1,):
            raise ValueError("Only the reported probability-interface Linear(4,1) head is accepted")
        parts = prefix.split(".")
        if any(not part.isidentifier() for part in parts) or parts[0] in model._modules:
            raise ValueError("Unexpected training-only head path: " + prefix)
        node = model
        for name in parts[:-1]:
            if name not in node._modules:
                node.add_module(name, nn.Module())
            node = node._modules[name]
        node.add_module(parts[-1], nn.Linear(4, 1))
        added.append(prefix)
    return added


def construct_model(entry, state):
    family = entry["family"]
    if family in CNN_FAMILIES:
        model = RiskAwareOrdinalCNN("coral", dropout=0.5)
        model.training_only_compatibility_heads = _attach_training_only_scalar_heads(model, state)
    elif family == "deit_tiny":
        import timm
        model = timm.create_model(
            "deit_tiny_patch16_224.fb_in1k", pretrained=False, num_classes=4, drop_rate=0.25
        )
    elif family in REFERENCE_FAMILIES:
        from .reference import Net
        model = Net(pretrained=False)
    else:
        raise ValueError("Unrecognized model family: " + family)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model(models_root, group, family, fold, device="cpu"):
    entry = find_entry(models_root, group, family, fold)
    path = checkpoint_path(models_root, entry, verify=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise ValueError("Expected the tensor-only Paper2 export format")
    state = payload["state_dict"]
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("State dictionary contains non-tensor metadata")
    return construct_model(entry, state).to(device), entry


@torch.no_grad()
def probabilities(model, entry, inputs):
    """Use the corresponding deterministic input chain. Reference outputs retain
    their source 1e-6 floor without row renormalization; these are class scores.
    """
    family = entry["family"]
    if inputs.ndim != 4 or inputs.shape[0] < 1:
        raise ValueError("Expected a nonempty NCHW input tensor")
    shape = (1, 64, 64) if family in CNN_FAMILIES else ((3, 224, 224) if family == "deit_tiny" else (1, 224, 224))
    if tuple(inputs.shape[1:]) != shape or not torch.isfinite(inputs).all():
        raise ValueError("Input shape or values do not match the declared model family")
    if model.training:
        raise ValueError("Inference requires eval mode")
    if family in CNN_FAMILIES:
        logits, _ = model(inputs)
        return cumulative_and_class_probabilities(logits, "coral")[1]
    if family == "deit_tiny":
        return model(inputs).softmax(dim=1)
    if family in REFERENCE_FAMILIES:
        from .reference import corn_probs
        return corn_probs(model(inputs))
    raise ValueError("Unrecognized model family")
