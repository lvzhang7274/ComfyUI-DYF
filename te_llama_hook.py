"""Hook TE_MAN's llama.cpp GGUF weights into ComfyUI's global unload path.

Why this exists
---------------
TE_MAN's `TE_Qwen_Image_2_1_Prompt_Enhancer` has three backends.  The
"本地" (local) one uses `llama-cpp-python` to load a GGUF straight onto the
GPU via `cudaMalloc`.  That allocation is invisible to ComfyUI twice over:

  * the `Llama` instance is never registered in
    `comfy.model_management.current_loaded_models`, so `unload_all_models()`
    walks right past it;
  * it does not live in PyTorch's caching allocator, so `soft_empty_cache()`
    has nothing to release.

Net effect: hitting /free (Manager's "Free model and node cache") returns 200
and frees nothing while a 5 GB GGUF stays resident.  The node also reports
`vram 0B` afterwards, because ComfyUI measures with
`torch.cuda.memory_allocated()` and llama.cpp never touches torch.

TE_MAN already ships the release path — `TEQwenImage21LocalStorage.unload()`,
a zero-argument classmethod.  Nothing ever calls it except the node's own
"生成后自动卸载模型" switch.  This module wires it to the global unload.

Only the *local* storage is touched.  `TEQwenImage21OfficialStorage` loads
through ComfyUI's own `CLIPLoader`, so it is already in
`current_loaded_models` and /free already handles it; unloading it a second
time here would risk a double free.

Trigger surface is narrow by design — `unload_all_models()` is called from
exactly two places in core: the /free endpoint (via the `unload_models` queue
flag, main.py:414) and the OOM handler (execution.py:647).  Ordinary prompt
execution never reaches it, so this does not cause GGUF thrashing.
"""

import gc
import logging
import sys
from functools import wraps

import comfy.model_management as mm

HOOK_FLAG = "_dyf_te_llama_unload_hook"
STORAGE_CLASS_NAME = "TEQwenImage21LocalStorage"

_log = logging.getLogger(__name__)


def _find_storage_class():
    """Locate TE_MAN's storage class among *already imported* modules.

    It has to be the live class object.  Re-importing the extension module
    builds a fresh class whose `model` attribute is None, and `unload()` on
    that one is a silent no-op — the classic way to write this hook and have
    it appear to work while freeing nothing.
    """
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            cls = getattr(mod, STORAGE_CLASS_NAME, None)
        except Exception:
            continue
        if isinstance(cls, type):
            return cls
    return None


def unload_te_llama() -> bool:
    """Release TE_MAN's resident GGUF.  Returns True if something was loaded."""
    cls = _find_storage_class()
    if cls is None:
        return False
    try:
        was_loaded = getattr(cls, "model", None) is not None
        cls.unload()
        if was_loaded:
            gc.collect()
            _log.info("[DYF] 已卸载 TE_MAN 本地 GGUF 模型，释放 llama.cpp 显存")
        return was_loaded
    except Exception as exc:
        _log.warning("[DYF] 卸载 TE_MAN 本地 GGUF 模型失败: %s", exc)
        return False


def install_hook() -> None:
    """Wrap `unload_all_models` so /free also drops the TE_MAN GGUF.

    Safe to call at import time: the storage class is resolved lazily on each
    invocation, so load order relative to TE_MAN does not matter.
    """
    if getattr(mm, HOOK_FLAG, False):
        return

    original = getattr(mm, "unload_all_models", None)
    if original is None or not callable(original):
        _log.warning("[DYF] 未找到 unload_all_models，跳过 TE_MAN 显存挂接")
        return

    @wraps(original)
    def wrapped_unload_all_models(*args, **kwargs):
        unload_te_llama()
        return original(*args, **kwargs)

    mm.unload_all_models = wrapped_unload_all_models
    setattr(mm, HOOK_FLAG, True)
    _log.info("[DYF] 已挂接 TE_MAN 本地 GGUF 显存清理到全局卸载")
