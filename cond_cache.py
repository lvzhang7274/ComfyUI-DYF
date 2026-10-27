# -*- coding: utf-8 -*-
"""Lossless MiniMax H3 conditioning cache.

ComfyUI's built-in SaveConditioning only accepts tensor lists. H3 carries its
reference blocks in ``minimax_refs`` as a list of dicts mixing ints, strings and
tensors, so the core node rejects it with
``ValueError: Conditioning option 'minimax_refs' (list) can't be saved``.

These nodes pickle the conditioning object as a whole, so every side channel
survives the round trip unchanged.
"""

from __future__ import annotations

import hashlib
import os
import time

import torch
from comfy_api.latest import io, ui

try:
    import folder_paths
except ImportError:  # pragma: no cover - standalone import during smoke tests
    folder_paths = None

CACHE_DIRNAME = "h3_conditioning"
EXT = ".h3c"
FORMAT_TAG = "dyf_h3_conditioning"
FORMAT_VERSION = "1"
FILE_ROUTE = "/ComfyUI-DYF/conditioning-files"

_ROUTE_READY = False


def _output_root() -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")


def _cache_dir(create: bool = False) -> str:
    path = os.path.join(_output_root(), CACHE_DIRNAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _cache_files() -> list[str]:
    """Newest first, bare names without extension."""
    directory = _cache_dir()
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        if not name.endswith(EXT):
            continue
        full = os.path.join(directory, name)
        try:
            entries.append((os.path.getmtime(full), name[: -len(EXT)]))
        except OSError:
            continue
    entries.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in entries]


def _resolve(name: str) -> str | None:
    if not name:
        return None
    path = os.path.join(_cache_dir(), name + EXT if not name.endswith(EXT) else name)
    return path if os.path.isfile(path) else None


def _register_file_route(verbose: bool = True) -> bool:
    """Let the front end pull the file list so it can refresh without a restart."""
    global _ROUTE_READY
    if _ROUTE_READY:
        return True
    try:
        from aiohttp import web
        from server import PromptServer

        routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
        if routes is None:
            return False

        async def list_conditioning_files(request):
            return web.json_response(_cache_files())

        routes.get(FILE_ROUTE)(list_conditioning_files)
        _ROUTE_READY = True
        return True
    except Exception as exc:  # pragma: no cover
        if verbose:
            print("[ComfyUI-DYF] could not register {!r} ({}); the load node falls back to a "
                  "static list, restart ComfyUI to pick up new saves".format(FILE_ROUTE, exc))
        return False


def _scan(obj, path: str, sink: list) -> None:
    """Record fields that are neither tensors, scalars, strings nor containers."""
    if torch.is_tensor(obj):
        return
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _scan(value, f"{path}[{i}]", sink)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _scan(value, f"{path}.{key}", sink)
        return
    sink.append((path, type(obj).__name__))


def _measure(obj) -> tuple[int, int]:
    """Count tensors and their raw byte size anywhere in the object."""
    if torch.is_tensor(obj):
        return 1, int(obj.numel() * obj.element_size())
    total_tensors = total_bytes = 0
    if isinstance(obj, (list, tuple)):
        children = obj
    elif isinstance(obj, dict):
        children = list(obj.values())
    else:
        return 0, 0
    for child in children:
        tensors, nbytes = _measure(child)
        total_tensors += tensors
        total_bytes += nbytes
    return total_tensors, total_bytes


class DYFSaveConditioning(io.ComfyNode):
    """Write the full conditioning object to output/h3_conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DYF_SaveConditioning",
            display_name="保存 H3 条件 (完整)",
            category="model/conditioning/minimax",
            search_aliases=["save conditioning", "cache text encoder output", "保存条件"],
            description="保存完整的 H3 conditioning（含 minimax_refs / keyframes 等侧通道）到磁盘，"
                        "之后可用「加载 H3 条件 (完整)」跳过 CLIP 与原参考图编码。",
            is_output_node=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("filename", default="h3_cond",
                                tooltip="不含扩展名。写好后出现在 加载 H3 条件 的下拉里。"),
            ],
            outputs=[io.Conditioning.Output(display_name="CONDITIONING")],
        )

    @classmethod
    def execute(cls, conditioning, filename) -> io.NodeOutput:
        safe = "".join(c for c in (filename or "") if c.isalnum() or c in ("_", "-", "."))
        safe = safe or "h3_cond"
        path = os.path.join(_cache_dir(create=True), safe + EXT)

        offenders: list[tuple[str, str]] = []
        _scan(conditioning, "conditioning", offenders)
        if offenders:
            head = ", ".join(f"{p}:{t}" for p, t in offenders[:6])
            suffix = " ..." if len(offenders) > 6 else ""
            print(f"[ComfyUI-DYF] condition contains non-trivial fields -> {head}{suffix}")

        tensors, nbytes = _measure(conditioning)
        torch.save(
            {
                "format": FORMAT_TAG,
                "version": FORMAT_VERSION,
                "created": time.time(),
                "conditioning": conditioning,
            },
            path,
        )

        size_mb = os.path.getsize(path) / (1024 * 1024)
        message = f"已保存 {size_mb:.1f} MB / {tensors} 个张量 / {nbytes/1024**2:.1f} MB 数据 -> {path}"
        print(f"[ComfyUI-DYF] {message}")
        return io.NodeOutput(conditioning, ui=ui.PreviewText(message))


class DYFLoadConditioning(io.ComfyNode):
    """Restore a conditioning written by 保存 H3 条件 (完整)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files = _cache_files()
        _register_file_route()
        options = files or ["(尚无缓存，先跑一次 保存 H3 条件)"]
        return io.Schema(
            node_id="DYF_LoadConditioning",
            display_name="加载 H3 条件 (完整)",
            category="model/conditioning/minimax",
            search_aliases=["load conditioning", "skip text encoder", "加载条件"],
            description="读回「保存 H3 条件 (完整)」写出的 conditioning。"
                        "用它可以在采样工作流里彻底去掉 CLIPLoader 与参考图分支，从而不再加载它们的显存/内存。",
            inputs=[
                io.Combo.Input("filename", options=options, default=options[0],
                               remote=io.RemoteOptions(route=FILE_ROUTE, refresh_button=True,
                                                       control_after_refresh="first")
                               if _ROUTE_READY else None,
                               tooltip="最新在前。点输入框旁的刷新按钮可拿到最新保存的文件，无需重启 ComfyUI。"),
            ],
            outputs=[io.Conditioning.Output(display_name="CONDITIONING")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        name = kwargs.get("filename")
        if _resolve(name) is None:
            return f"找不到条件缓存：{name}。先跑一次「保存 H3 条件 (完整)」。"
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        path = _resolve(kwargs.get("filename"))
        if path is None:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        digest = hashlib.sha256(f"{path}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()
        return digest

    @classmethod
    def execute(cls, filename) -> io.NodeOutput:
        path = _resolve(filename)
        if path is None:
            raise ValueError(f"conditioning cache not found: {filename}")

        # pickle of arbitrary python objects: torch >=2.6 defaults weights_only=True and would refuse it
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "conditioning" in payload:
            if payload.get("format") != FORMAT_TAG:
                print(f"[ComfyUI-DYF] {filename} looks foreign (format={payload.get('format')}), loading anyway")
            conditioning = payload["conditioning"]
        else:
            conditioning = payload

        entries = len(conditioning) if isinstance(conditioning, list) else -1
        keys = sorted(conditioning[0][1].keys()) if entries > 0 else []
        print(f"[ComfyUI-DYF] loaded {filename}: {entries} entry/entries, options={keys}")
        return io.NodeOutput(conditioning)


# Attempt at import time: main.py runs init_extra_nodes() before prompt_server.add_routes(),
# so the route table is still open then. define_schema() retries later if this fails.
_register_file_route(verbose=False)
