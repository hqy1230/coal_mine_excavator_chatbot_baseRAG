from contextvars import ContextVar
from typing import Any, TypedDict


class InferenceOptions(TypedDict, total=False):
    enable_sft: bool
    enable_rl: bool


_default_opts: InferenceOptions = {"enable_sft": True, "enable_rl": True}
_opts_var: ContextVar[InferenceOptions | None] = ContextVar("robot_cs_inference_opts", default=None)


def set_inference_options(opts: InferenceOptions | None) -> Any:
    base = dict(_default_opts)
    if opts:
        base.update(opts)
    return _opts_var.set(base)


def reset_inference_options(token: Any) -> None:
    try:
        _opts_var.reset(token)
    except ValueError:
        pass


def get_inference_options() -> InferenceOptions:
    v = _opts_var.get()
    if v is None:
        return dict(_default_opts)
    return dict(v)
