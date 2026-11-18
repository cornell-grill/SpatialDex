from safetensors.torch import load_file

from spatialdex.model.spatialdex import SpatialDex

TIED_KEY = "backbone.language_model.lm_head.weight"


def load_weights(model, checkpoint, strict=True):
    state = load_file(str(checkpoint))
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    if not strict:
        own = model.state_dict()
        state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if strict and (unexpected or set(missing) - {TIED_KEY}):
        raise RuntimeError(f"Checkpoint does not match the model: missing={missing}, unexpected={unexpected}")
    language_model = model.backbone.language_model
    language_model.lm_head.weight = language_model.model.embed_tokens.weight
    return missing


def load_model(config, checkpoint):
    model = SpatialDex(config)
    load_weights(model, checkpoint)
    model.use_bf16 = not config.get("fp32_inference", False)
    return model


__all__ = ["SpatialDex", "load_model", "load_weights"]
