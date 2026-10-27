from .nodes import comfy_entrypoint
from .te_llama_hook import install_hook as _install_te_llama_hook

# 让 Manager 的「清理节点缓存和模型」顺带卸掉 TE_MAN 的 llama.cpp GGUF 权重。
# 详见 te_llama_hook.py —— 那部分显存不归 ComfyUI 管，原生 /free 够不到。
_install_te_llama_hook()

__all__ = ["comfy_entrypoint"]
