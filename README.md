# ComfyUI-DYF

Latent-space helpers for MiniMax H3: lossless continuation, latent-based guides, and full conditioning cache.

MiniMax H3 的潜空间工具集：无损长视频续接、直接用 latent 做引导、完整条件缓存。

---

## 为什么需要这个包

ComfyUI 核心的 `MiniMaxH3AddGuide` 只接受 **IMAGE**，拿到后立刻
`vae.encode(_resize(image, ...))`。做长视频续接时，上一段的尾帧本来就已经是
latent 了 —— 先解码成像素、再重新编码回去，纯属一次有损往返（还要额外过一次
lanczos 缩放），这个损失在最终画面里表现为接缝。

但下游其实并不要求 guide 必须来自 VAE：核心只是对 `keyframe["latent"]` 做
`patchify_video` 然后把那些行钉在 condition 时间步上。所以**从上一段自己的
latent 里切出来的尾部，是完全合法的 guide**，不需要过 VAE。

本包就是围绕这个思路做的。

## 节点

| 节点 | node_id | 作用 |
|---|---|---|
| H3 潜空间取尾 (Latent Tail Slice) | `H3LatentTailSlice` | 从上一段 latent 切出尾部作为下一段的 guide，避免 VAE 往返 |
| 为 MiniMax H3 添加引导 (Latent 输入) | `MiniMaxH3AddLatentGuide` | 核心节点只吃 IMAGE，这个直接吃 LATENT |
| MiniMax H3 潜空间硬续接 (无损失) | `MiniMaxH3LatentContinuation` | 拼接两段 latent，接缝处 mask 标 0 硬保留 |
| 保存 H3 Latent | `H3LatentSave` | 核心 `SaveLatent` 对 H3 不可用，见下 |
| 加载 H3 Latent | `H3LatentLoad` | 与上一个配对 |
| 保存 H3 条件 (完整) | `DYFSaveConditioning` | 完整存取 conditioning，含私有键 |
| 加载 H3 条件 (完整) | `DYFLoadConditioning` | 与上一个配对 |

### 为什么不能直接用核心的 SaveLatent

H3 的 `latent["samples"]` 是 `NestedTensor`（video `[B,24,T,H/16,W/16]` + audio
`[B,32,2,A]` 双流），它没有 `.contiguous` 方法，而核心 `SaveLatent` 会调用。所以必须走
`H3LatentSave`。

## H3 时间网格速查

```python
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)   # 周期 5 token = 17 帧
FRAME_RESCALE   = 5.0 / 3.0
```

- token k 覆盖 `FRAME_PER_TOKEN[k % 5]` 个像素帧，所以 **5 token = 17 帧**
- 合法视频长度：`17k + 5` 帧 = 5 / 22 / 39 / 56 / 73 / 90 …
- **39 帧是又短又音视频双精确的最小长度**（音频步长 65，无舍入）
- 关键帧块的时间相位会从自己的锚点重新起算，所以尾部切片必须从 token 索引
  对 5 取模为 0 的位置开始 —— 好在所有合法长度都满足 `latent_t = 5k + 2`，
  任意两段相减恒为 5 的倍数，切尾自动相位对齐

## 关于「无损」

采样器最终合成是：

```python
out = out * denoise_mask + latent_image * (1 - denoise_mask)
```

mask=0 的区域**逐元素等于输入 latent**，这是「零损耗」的唯一来源。

需要留意的一点：mask=0 的区域在**去噪过程中**并不是原样参与计算的（会被以
conditioning 强度注入），只在最终结果里被替换回来。所以「硬保留」≠「模型看到了原样」。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-name>/ComfyUI-DYF.git
```

重启 ComfyUI 即可，无第三方依赖。

要求：ComfyUI **0.42+**（使用 V3 扩展 API `comfy_api.latest`，`__init__.py` 只导出
`comfy_entrypoint`，不提供 `NODE_CLASS_MAPPINGS`）。

## 附带的运行时补丁

`te_llama_hook.py` **与本包节点功能无关**，是给另一个插件
[TE_MAN](../TE_MAN) 打的运行时补丁，装不装都不影响上面 7 个节点。

它解决的问题：TE_MAN 的 Qwen Image 2.1 提示词增强节点在「本地」模式下用
`llama-cpp-python` 直接加载 GGUF，权重由 llama.cpp 用 `cudaMalloc` 分配。这块显存
既不在 `comfy.model_management.current_loaded_models` 里（`unload_all_models()` 看不见），
也不归 PyTorch 的 caching allocator 管（`soft_empty_cache()` 够不着）。结果是点 ComfyUI
的「清理节点缓存和模型」毫无效果，5GB 权重原地不动。

补丁的做法是在启动时包一层 `unload_all_models`，顺带调用 TE_MAN 自己提供但无人调用的
`TEQwenImage21LocalStorage.unload()`。

- 只在 `/free`（Manager 那个按钮）和 OOM 时触发，正常执行不会误卸
- 没装 TE_MAN 时是空操作，不会报错
- 不想要的话，删掉 `te_llama_hook.py` 并把 `__init__.py` 里那三行去掉即可

## 许可

MIT
