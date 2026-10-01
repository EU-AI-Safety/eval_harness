"""Compat shim for Gemma-4 + transformers>=5.14 heterogeneous configs.

transformers 5.14+ remaps Gemma-4 dual head dims into per_layer_config and
raises AmbiguousGlobalPerLayerAttributeError on global ``head_dim`` reads.
vLLM 0.26 still expects legacy ``head_dim`` / ``global_head_dim`` fields.

Apply before launching the OpenAI API server::

    python3 -m eval_harness.vllm_gemma4_hetero_compat -- \\
      -m vllm.entrypoints.openai.api_server ...
"""

from __future__ import annotations

import runpy
import sys
from typing import Any


def _enable_global_access(config: Any) -> None:
    if config is None:
        return
    try:
        config.allow_global_per_layer_attribute_access = True
    except Exception:
        pass


def _reinject_legacy_dual_head_attrs(config: Any) -> None:
    """Restore global_head_dim / num_global_key_value_heads for native gemma4."""
    if config is None:
        return
    _enable_global_access(config)

    text = config
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        try:
            text = get_text() or config
        except Exception:
            text = config
    _enable_global_access(text)

    if not bool(getattr(text, "is_heterogeneous", False)):
        return

    per_layer = getattr(text, "per_layer_config", None)
    layer_types = getattr(text, "layer_types", None) or []
    if per_layer is None or not layer_types:
        return

    head_dims: list[int] = []
    full_kv: list[int] = []
    for i, layer_type in enumerate(layer_types):
        try:
            layer = per_layer[i]
        except Exception:
            continue
        hd = getattr(layer, "head_dim", None)
        nkv = getattr(layer, "num_key_value_heads", None)
        if hd is not None:
            head_dims.append(int(hd))
        if nkv is not None and layer_type == "full_attention":
            full_kv.append(int(nkv))

    if head_dims and getattr(text, "global_head_dim", None) in (None, 0):
        try:
            text.global_head_dim = max(head_dims)
        except Exception:
            pass
    if full_kv and getattr(text, "num_global_key_value_heads", None) in (None, 0):
        try:
            text.num_global_key_value_heads = full_kv[0]
        except Exception:
            pass

    sub_configs = getattr(config, "sub_configs", None)
    if isinstance(sub_configs, dict):
        for name in sub_configs:
            _enable_global_access(getattr(config, name, None))


def apply_patch() -> None:
    # Patch get_config only — reinjecting legacy attrs is enough for vLLM 0.26's
    # Gemma4ModelArchConfigConvertor.get_head_size and native gemma4.py.
    # Avoid importing model_arch_config_convertor here (circular import with config).
    import vllm.transformers_utils.config as vllm_config_mod

    if getattr(vllm_config_mod.get_config, "_eu_guard_gemma4_patched", False):
        return

    original_get_config = vllm_config_mod.get_config

    def get_config_patched(*args: Any, **kwargs: Any):
        config = original_get_config(*args, **kwargs)
        _reinject_legacy_dual_head_attrs(config)
        return config

    get_config_patched._eu_guard_gemma4_patched = True  # type: ignore[attr-defined]
    vllm_config_mod.get_config = get_config_patched  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print(
            "Usage: python3 -m eval_harness.vllm_gemma4_hetero_compat -- "
            "-m vllm.entrypoints.openai.api_server ...",
            file=sys.stderr,
        )
        raise SystemExit(2)

    apply_patch()

    if args[0] == "-m":
        if len(args) < 2:
            raise SystemExit("Missing module name after -m")
        module = args[1]
        sys.argv = [module, *args[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
