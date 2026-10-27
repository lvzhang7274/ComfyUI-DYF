"""Latent-space guides / continuation for MiniMax H3.

Why this exists
---------------
Core's `MiniMaxH3AddGuide` takes an IMAGE and immediately does
`vae.encode(_resize(image, ...))`.  For long-video continuation the previous
chunk's tail is already a latent: decoding it to pixels just so the guide node
can re-encode it is a pure round trip through a lossy VAE (plus a lanczos
resize), and that loss shows up as a visible seam.

Nothing downstream requires the guide latent to have come from a VAE.  Core
only does `patchify_video(z.float32, patch_size)` on `keyframe["latent"]` and
pins those rows at the condition timestep.  So a latent sliced straight out of
the previous chunk's own latent is a perfectly valid guide.

H3 temporal grid
----------------
`FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`, cyclic: token k covers
`FRAME_PER_TOKEN[k % 5]` pixel frames, so 5 tokens = 17 frames and a legal
video is `17k + 5` frames = `5k + 2` tokens (5->2, 22->7, 39->12, ...).
A keyframe chunk restarts that cycle at phase 0 regardless of where it came
from, so a tail slice must start at a token index congruent to 0 (mod 5).
Conveniently every legal video has `latent_t = 5k + 2` and every legal guide
has `5j + 2` tokens, so `latent_t - guide_t` is always 0 (mod 5): slicing the
tail is automatically phase-correct.
"""

import hashlib
import json
import os

import safetensors.torch
import torch
from safetensors import safe_open

import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
from comfy.cli_args import args
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .cond_cache import DYFLoadConditioning, DYFSaveConditioning

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32

# Core's SaveLatent does `samples["samples"].contiguous()`; an H3 AV latent's
# samples is a NestedTensor (video + audio), which is not a tensor at all. So
# H3 latents get their own container: one tensor per stream plus the shapes
# needed to rebuild the NestedTensor. The extension is deliberately not
# ".latent" so core's Load Latent never offers a file it cannot read.
LATENT_EXT = ".h3latent"
FORMAT_TAG = "h3_av_latent"
FORMAT_VERSION = "1"
PRECISION_OPTIONS = ["fp32", "fp16"]

# A combo's options are baked into /object_info when the frontend fetches it, so a
# file saved after page load never shows up until the user refreshes node
# definitions. This route backs the widget's refresh button instead.
FILE_ROUTE = "/h3latent/files"
_ROUTE_READY = False

# H3 video lengths are 17k + 5 frames.  Audio runs at 40 Hz against 24 fps video,
# so an exact video+audio boundary additionally needs frames % 3 == 0.
FRAME_GRID = [17 * k + 5 for k in range(12)]                     # 5, 22, 39 ... 192
AUDIO_EXACT = [f for f in FRAME_GRID if f % 3 == 0]              # 39, 90, 141, 192
FRAME_OPTIONS = [str(f) for f in FRAME_GRID]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _video_stream(samples):
    """Video tensor out of an H3 AV latent, or out of a bare [B,24,T,H,W] latent."""
    if getattr(samples, "is_nested", False):
        return samples.tensors[0]
    return samples


def frames_for_tokens(t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(t))


def tokens_for_frames(frames):
    """Inverse of frames_for_tokens on the 17k+5 grid."""
    if frames not in FRAME_GRID:
        raise ValueError("{} is not an H3 frame length (17k+5): {}".format(frames, FRAME_GRID))
    return ((frames - 5) // 17) * 5 + 2


def _parse_frames(value):
    """Combo values are untrusted: an API prompt can submit anything. Allowlist only."""
    try:
        frames = int(value)
    except (TypeError, ValueError):
        return None
    return frames if frames in FRAME_GRID else None


def _check_av(samples, what):
    if not (getattr(samples, "is_nested", False) and len(samples.tensors) == 2):
        raise ValueError("{} must be a MiniMax H3 AV latent (video + audio streams)".format(what))
    video = samples.tensors[0]
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise ValueError("{} is not an H3 AV latent, got video shape {}".format(what, tuple(video.shape)))
    return video, samples.tensors[1]


def _grid_error(value):
    return "must be an H3 17k+5 frame length ({}), got {!r}".format(", ".join(FRAME_OPTIONS), value)


def _streams(samples):
    """A latent's streams as a list: [video, audio] for H3 AV, [video] for a bare latent."""
    if getattr(samples, "is_nested", False):
        return list(samples.tensors)
    return [samples]


def _pack(samples):
    if len(samples) > 1:
        return comfy.nested_tensor.NestedTensor(samples)
    return samples[0]


def _saved_latent_files():
    """`*.h3latent` in output/ and input/, newest first, tagged `[output]` / `[input]`.

    The tag is what `folder_paths.get_annotated_filepath` resolves against, so the
    same node can load back what it just saved without the user moving files around.
    """
    found = []
    for folder, tag in (("output", folder_paths.get_output_directory()),
                        ("input", folder_paths.get_input_directory())):
        for root, _, files in os.walk(tag):
            for name in files:
                if not name.endswith(LATENT_EXT):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, tag).replace(os.sep, "/")
                found.append((os.path.getmtime(path), "{0} [{1}]".format(rel, folder)))
    found.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in found]


def _resolve_latent_path(name):
    """Annotated (`x.h3latent [output]`) or bare (`latents/x.h3latent`) name -> absolute path, or None."""
    if not name:
        return None
    try:
        annotated = folder_paths.get_annotated_filepath(name)
        if os.path.isfile(annotated):
            return annotated
    except ValueError:
        pass
    # a bare name defaults to input/ in core; output/ is where this node writes, try it first
    for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
        candidate = os.path.abspath(os.path.join(base, name))
        if os.path.isfile(candidate) and os.path.commonpath([candidate, os.path.abspath(base)]) == os.path.abspath(base):
            return candidate
    return None


def _read_metadata(path):
    """Safetensors metadata without touching the tensors."""
    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def _register_file_route():
    """Append the file-listing route to PromptServer's table.

    Safe at import time: main.py runs `init_extra_nodes()` before
    `prompt_server.add_routes()`, so the table is still open.
    """
    global _ROUTE_READY
    if _ROUTE_READY:
        return True
    try:
        from aiohttp import web
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        routes = getattr(instance, "routes", None)
        if routes is None:
            return False

        async def list_h3_latents(request):
            return web.json_response(_saved_latent_files())

        routes.get(FILE_ROUTE)(list_h3_latents)
        _ROUTE_READY = True
        return True
    except Exception as exc:                                    # pragma: no cover
        print("[H3LatentGuide] could not register {!r} ({}); the load node falls back to a "
              "static file list, refresh ComfyUI to pick up new saves".format(FILE_ROUTE, exc))
        return False


# --------------------------------------------------------------------------- #
# 1. slice a guide out of the previous chunk's latent
# --------------------------------------------------------------------------- #
class H3LatentTailSlice(io.ComfyNode):
    """Take the last N frames of a previous chunk straight out of its latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentGuide_TailSlice",
            display_name="H3 潜空间取尾 (Latent Tail Slice)",
            category="latent/minimax",
            description="Slice the tail of a finished MiniMax H3 chunk out of its own latent. No VAE decode, no re-encode, no loss.",
            inputs=[
                io.Latent.Input("latent", tooltip="The previous chunk's H3 AV latent, straight from the sampler."),
                io.Combo.Input("tail_frames", options=FRAME_OPTIONS, default="39",
                               tooltip="Frames to keep. 39 = 12 latent tokens, and the smallest exact video+audio boundary."),
            ],
            outputs=[io.Latent.Output(display_name="guide_latent",
                                      tooltip="A bare 24-channel H3 video latent, ready to anchor as a guide.")],
        )

    @classmethod
    def validate_inputs(cls, tail_frames, latent=None):
        if _parse_frames(tail_frames) is None:
            return "tail_frames " + _grid_error(tail_frames)
        return True

    @classmethod
    def execute(cls, latent, tail_frames) -> io.NodeOutput:
        frames = _parse_frames(tail_frames)
        if frames is None:
            raise ValueError("tail_frames " + _grid_error(tail_frames))

        prev = _video_stream(latent["samples"])
        if prev.ndim != 5 or prev.shape[1] != VIDEO_CHANNELS:
            raise ValueError("previous latent is not an H3 video latent, got shape {}".format(tuple(prev.shape)))

        n = tokens_for_frames(frames)
        t_prev = prev.shape[2]
        if n > t_prev:
            raise ValueError("a {}-frame tail ({} tokens) is longer than the previous chunk ({} tokens)".format(
                frames, n, t_prev))

        start = t_prev - n
        # a legal source (5k+2 tokens) minus a legal guide (5j+2 tokens) is always 0 mod 5,
        # so this only fires for hand-built or latently-trimmed sources
        if start % 5 != 0:
            print("[H3LatentGuide_TailSlice] tail starts at token {} ({} mod 5): a guide's frame layout "
                  "restarts at phase 0, so its timing is shifted".format(start, start % 5))

        return io.NodeOutput({"samples": prev[:, :, start:, :, :].clone()})


# --------------------------------------------------------------------------- #
# 2. latent-input version of Core's MiniMaxH3AddGuide
# --------------------------------------------------------------------------- #
class MiniMaxH3AddLatentGuide(io.ComfyNode):
    """Anchor an H3 video latent at any frame of the target, with no VAE round trip."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentGuide_AddLatentGuide",
            display_name="为 MiniMax H3 添加引导 (Latent 输入)",
            category="model/conditioning/minimax",
            description="Anchor an H3 video latent at any frame of the target. Same contract as MiniMaxH3AddGuide, but the guide is already a latent, so nothing is decoded or re-encoded.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="The target H3 AV latent, e.g. from Empty MiniMax H3 AV Latent."),
                io.Latent.Input("guide_latent", tooltip="24-channel H3 video latent to anchor. Spatial size must match the target exactly."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999, step=1,
                             tooltip="Target frame to anchor the guide at. Negative values count from the end."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, guide_latent, frame_idx) -> io.NodeOutput:
        video, _ = _check_av(latent["samples"], "latent")
        frame_count = frames_for_tokens(video.shape[2])

        guide = _video_stream(guide_latent["samples"])
        if guide.ndim != 5 or guide.shape[1] != VIDEO_CHANNELS:
            raise ValueError("guide_latent must be an H3 video latent [B,24,T,H,W], got {}".format(tuple(guide.shape)))
        if guide.shape[3] != video.shape[3] or guide.shape[4] != video.shape[4]:
            raise ValueError("guide latent is {}x{} but the target is {}x{}; there is no lossless latent-space "
                             "resize, so decode+encode or match the resolution".format(
                                 guide.shape[4] * 16, guide.shape[3] * 16,
                                 video.shape[4] * 16, video.shape[3] * 16))

        guide_frames = frames_for_tokens(guide.shape[2])
        resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved < 0 or resolved + guide_frames > frame_count:
            raise ValueError("a {}-frame guide at frame_idx {} does not fit in the target's {} frames".format(
                guide_frames, frame_idx, frame_count))

        keyframe = {"resolved_frame_index": resolved, "latent": guide.clone()}
        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        return io.NodeOutput(node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes}))


# --------------------------------------------------------------------------- #
# 3. lossless continuation: copy the tail in and mask it as preserved
# --------------------------------------------------------------------------- #
class MiniMaxH3LatentContinuation(io.ComfyNode):
    """Hard continuation. The previous chunk's latent tail becomes the new chunk's prefix.

    Unlike a guide this is exact: the masked prefix is never denoised, so the
    sampler returns it bit-for-bit. Decode the whole chunk and drop the first
    `overlap_frames` frames when assembling.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentGuide_LatentContinuation",
            display_name="MiniMax H3 潜空间硬续接 (无损失)",
            category="latent/minimax",
            description="Copy the previous chunk's latent tail into the head of a fresh H3 AV latent and mark it preserved with a native per-token denoise mask. Zero loss, because nothing is decoded.",
            inputs=[
                io.Latent.Input("latent", tooltip="Fresh target AV latent, e.g. from Empty MiniMax H3 AV Latent."),
                io.Latent.Input("previous_latent", tooltip="The accepted previous chunk's AV latent, straight from the sampler."),
                io.Combo.Input("overlap_frames", options=FRAME_OPTIONS, default="39",
                               tooltip="Frames of the previous chunk repeated at the head of the new one. Trim them after decoding."),
                io.Boolean.Input("protect_audio", default=True,
                                 tooltip="Also carry the aligned audio tail. Needs an exact boundary: 39, 90, 141 or 192."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT",
                                      tooltip="The target latent with the preserved prefix and a native H3 noise_mask attached.")],
        )

    @classmethod
    def validate_inputs(cls, overlap_frames, protect_audio=True, latent=None, previous_latent=None):
        frames = _parse_frames(overlap_frames)
        if frames is None:
            return "overlap_frames " + _grid_error(overlap_frames)
        if protect_audio and frames not in AUDIO_EXACT:
            return "{} frames is {} audio steps, not an integer; audio-protected continuation needs one of {} (or turn protect_audio off)".format(
                frames, frames * FRAME_RESCALE, AUDIO_EXACT)
        return True

    @classmethod
    def execute(cls, latent, previous_latent, overlap_frames, protect_audio) -> io.NodeOutput:
        frames = _parse_frames(overlap_frames)
        if frames is None:
            raise ValueError("overlap_frames " + _grid_error(overlap_frames))
        if protect_audio and frames not in AUDIO_EXACT:
            raise ValueError("{} frames is not an exact audio boundary; use one of {} or disable protect_audio".format(
                frames, AUDIO_EXACT))

        tv, ta = _check_av(latent["samples"], "latent")
        pv, pa = _check_av(previous_latent["samples"], "previous_latent")

        n = tokens_for_frames(frames)
        if pv.shape[3] != tv.shape[3] or pv.shape[4] != tv.shape[4]:
            raise ValueError("previous chunk is {}x{} but the target is {}x{}; continuation needs identical geometry".format(
                pv.shape[4] * 16, pv.shape[3] * 16, tv.shape[4] * 16, tv.shape[3] * 16))
        if n > tv.shape[2]:
            raise ValueError("a {}-frame overlap ({} tokens) does not fit the target ({} tokens)".format(
                frames, n, tv.shape[2]))
        if n > pv.shape[2]:
            raise ValueError("a {}-frame overlap ({} tokens) is longer than the previous chunk ({} tokens)".format(
                frames, n, pv.shape[2]))

        na = round(frames * FRAME_RESCALE) if protect_audio else 0
        if na and (na > ta.shape[-1] or na > pa.shape[-1]):
            raise ValueError("audio overlap of {} steps does not fit (target {}, previous {})".format(
                na, ta.shape[-1], pa.shape[-1]))

        # copy so unrelated keys (batch_index, ...) survive
        out = latent.copy()
        video = tv.clone()
        video[:, :, :n] = pv[:, :, pv.shape[2] - n:]
        audio = ta.clone()
        if na:
            audio[..., :na] = pa[..., pa.shape[-1] - na:]
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))

        # 0 = preserve (the sampler keeps latent_image there), 1 = generate
        vmask = torch.ones((1, 1, tv.shape[2], tv.shape[3], tv.shape[4]), dtype=torch.float32)
        vmask[:, :, :n] = 0.0
        amask = torch.ones((1, 1, ta.shape[2], ta.shape[3]), dtype=torch.float32)   # [1,1,2,audio_t]
        if na:
            amask[..., :na] = 0.0
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))

        return io.NodeOutput(out)


# --------------------------------------------------------------------------- #
# 4. save / load an H3 AV latent
# --------------------------------------------------------------------------- #
class H3LatentSave(io.ComfyNode):
    """Save an H3 latent, including the audio stream that core's SaveLatent cannot see."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentGuide_SaveLatent",
            display_name="保存 H3 Latent",
            category="latent/minimax",
            search_aliases=["export latent", "保存潜空间", "save latent"],
            description="Save a MiniMax H3 latent to disk, video and audio streams intact. "
                        "Core's Save Latent crashes on H3 output because its samples are a NestedTensor, not a tensor.",
            inputs=[
                io.Latent.Input("latent", tooltip="An H3 AV latent (video + audio), or a bare 24-channel H3 video latent."),
                io.String.Input("filename_prefix", default="latents/H3"),
                io.Combo.Input("precision", options=PRECISION_OPTIONS, default="fp32",
                               tooltip="Storage precision. fp32 is bit-exact; fp16 halves the file size at the cost of "
                                       "rounding (fine for guides, not for lossless continuation)."),
                io.Boolean.Input("save_noise_mask", default=True,
                                 tooltip="Also store a noise_mask if the latent carries one."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT", tooltip="The input latent, unchanged, so it can be chained.")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        precision = kwargs.get("precision", "fp32")
        if precision not in PRECISION_OPTIONS:
            return "precision must be one of {}, got {!r}".format(PRECISION_OPTIONS, precision)
        return True

    @classmethod
    def execute(cls, latent, filename_prefix="latents/H3", precision="fp32",
                save_noise_mask=True) -> io.NodeOutput:
        if precision not in PRECISION_OPTIONS:
            raise ValueError("precision must be one of {}, got {!r}".format(PRECISION_OPTIONS, precision))

        streams = _streams(latent["samples"])
        for i, s in enumerate(streams):
            if not torch.is_tensor(s):
                raise ValueError("stream {} is a {}, not a tensor; this node only saves latents".format(
                    i, type(s).__name__))

        dtype = torch.float32 if precision == "fp32" else torch.float16
        sd = {}
        for i, s in enumerate(streams):
            sd["latent_tensor_{}".format(i)] = s.detach().to(device="cpu", dtype=dtype).contiguous()

        mask_streams = None
        if save_noise_mask:
            mask = latent.get("noise_mask")
            if mask is not None:
                mask_streams = _streams(mask)
                if len(mask_streams) == len(streams):
                    for i, m in enumerate(mask_streams):
                        sd["noise_mask_{}".format(i)] = m.detach().to(device="cpu", dtype=torch.float32).contiguous()
                else:
                    mask_streams = None

        # keeps any generic reader from applying the SD 0.18215 scale convention
        sd["latent_format_version_0"] = torch.tensor([])

        metadata = {
            FORMAT_TAG: FORMAT_VERSION,
            "stream_count": str(len(streams)),
            "stream_shapes": json.dumps([list(s.shape) for s in streams]),
            "has_noise_mask": "1" if mask_streams else "0",
            "precision": precision,
        }
        # cls.hidden is only populated during a real execution
        hidden = getattr(cls, "hidden", None)
        if not args.disable_metadata and hidden is not None:
            if hidden.prompt is not None:
                metadata["prompt"] = json.dumps(hidden.prompt)
            for key, value in (hidden.extra_pnginfo or {}).items():
                metadata[key] = json.dumps(value)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_output_folder, exist_ok=True)
        file = "{}_{:05}_".format(filename, counter) + LATENT_EXT
        path = os.path.join(full_output_folder, file)
        comfy.utils.save_torch_file(sd, path, metadata=metadata)
        print("[H3LatentGuide_SaveLatent] wrote {} ({} MB)".format(
            path, round(os.path.getsize(path) / 1e6, 2)))

        return io.NodeOutput(latent, ui={"latents": [{
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        }]})


class H3LatentLoad(io.ComfyNode):
    """Load back what `H3LatentGuide_SaveLatent` wrote."""

    @classmethod
    def define_schema(cls):
        files = _saved_latent_files()
        _register_file_route()
        return io.Schema(
            node_id="H3LatentGuide_LoadLatent",
            display_name="加载 H3 Latent",
            category="latent/minimax",
            search_aliases=["import latent", "open latent", "加载潜空间", "load latent"],
            description="Load a MiniMax H3 latent saved by 保存 H3 Latent. Rebuilds the video + audio NestedTensor exactly.",
            inputs=[
                io.Combo.Input("latent_file", options=files or ["(no saved H3 latent found)"],
                               default=files[0] if files else "(no saved H3 latent found)",
                               remote=io.RemoteOptions(route=FILE_ROUTE, refresh_button=True,
                                                       control_after_refresh="first") if _ROUTE_READY else None,
                               tooltip="Newest first; use the refresh button after saving. Files in output/ need no "
                                       "moving, input/ works too."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        latent_file = kwargs.get("latent_file")
        if _resolve_latent_path(latent_file) is None:
            return "Invalid H3 latent file: {}. Save one first, or click the widget's refresh button.".format(latent_file)
        # reading metadata is cheap (no tensor is materialised), so reject a foreign
        # file here rather than blowing up mid-run
        try:
            if _read_metadata(_resolve_latent_path(latent_file)).get(FORMAT_TAG) != FORMAT_VERSION:
                return "{} was not saved by 保存 H3 Latent, use Load Latent instead".format(latent_file)
        except Exception:
            return "{} is not a readable safetensors file".format(latent_file)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        path = _resolve_latent_path(kwargs.get("latent_file"))
        if path is None:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            digest.update(f.read())
        return digest.digest().hex()

    @classmethod
    def execute(cls, latent_file) -> io.NodeOutput:
        path = _resolve_latent_path(latent_file)
        if path is None:
            raise ValueError("H3 latent file not found: {}".format(latent_file))
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            tensors = {key: f.get_tensor(key) for key in f.keys()}

        if metadata.get(FORMAT_TAG) != FORMAT_VERSION:
            raise ValueError("{} is not an H3 latent saved by 保存 H3 Latent ({}), use Load Latent instead".format(
                latent_file, FORMAT_TAG))

        count = int(metadata.get("stream_count", "1"))
        streams = []
        for i in range(count):
            key = "latent_tensor_{}".format(i)
            if key not in tensors:
                raise ValueError("{} is missing {}: the file looks truncated".format(latent_file, key))
            streams.append(tensors[key].to(torch.float32))

        out = {"samples": _pack(streams)}
        if metadata.get("has_noise_mask") == "1":
            masks = [tensors["noise_mask_{}".format(i)].to(torch.float32)
                     for i in range(count) if "noise_mask_{}".format(i) in tensors]
            if len(masks) == count:
                out["noise_mask"] = _pack(masks)

        return io.NodeOutput(out)


class H3LatentGuideExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        _register_file_route()

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3LatentTailSlice,
            MiniMaxH3AddLatentGuide,
            MiniMaxH3LatentContinuation,
            H3LatentSave,
            H3LatentLoad,
            DYFSaveConditioning,
            DYFLoadConditioning,
        ]


async def comfy_entrypoint() -> H3LatentGuideExtension:
    return H3LatentGuideExtension()


# must happen at import time: main.py calls prompt_server.add_routes() after
# custom nodes are loaded, so the route table is closed by the time a node runs
_register_file_route()
