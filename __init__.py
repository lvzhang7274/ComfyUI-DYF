from .nodes import comfy_entrypoint
from .te_llama_hook import install_hook as _install_te_llama_hook

# 前端补丁目录（web/js/paste_link_fix.js）：修「复制一堆节点粘贴后部分连线消失」。
# 纯前端，不影响节点注册；删掉这个变量即还原。
WEB_DIRECTORY = "./web"

# 让 Manager 的「清理节点缓存和模型」顺带卸掉 TE_MAN 的 llama.cpp GGUF 权重。
# 详见 te_llama_hook.py —— 那部分显存不归 ComfyUI 管，原生 /free 够不到。
_install_te_llama_hook()

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
