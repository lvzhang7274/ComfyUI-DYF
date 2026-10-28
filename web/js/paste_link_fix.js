// DYF.PasteLinkFix —— 修 ComfyUI「复制一堆节点再粘贴，部分连线消失」
//
// 病因（frontend 1.55.x，LGraphCanvas）：
//   _serializeItems() 用 item.clone()?.serialize() 生成剪贴板数据，
//   而 clone() 里有一句 input.link = null —— 所有「哪个输入连了哪条线」被抹掉。
//   粘贴时 detachSerialisedLinks() 正是靠 input.link 建「输入名 -> link」映射，
//   拿到空表后 createClipboardLink() 只能退化用旧的 target_slot（槽位序号）。
//   而 ComfyNode.configure() 会把 data.inputs 重排成「定义布局 + 额外输入追加到末尾」，
//   所以只要源节点的槽位数量和新建时的定义布局不一致（Autogrow / 动态 combo / JS 动态加输入的节点都算），
//   旧的序号就落不到同名槽位上：类型不符 -> connectSlots 里 isValidConnection 返回 null -> 静默丢线。
//
// 修法（两层，都只在前端，不动 Python）：
//   1. 序列化后把 input.link 按「输入名」补回去 —— 让官方那套按名对齐的机制复活。
//   2. 粘贴后按名巡检：该连而没连上的，用 origin.connect() 重连一次。
//      第二层兜住第一层够不到的槽位（典型是 Autogrow 还没长出来的 image_2..N）。

import { app } from "../../../scripts/app.js";

const TAG = "[DYF 粘贴修复]";
const INSTALLED = "__DYF_PASTE_LINK_FIX__";

const stats = { injected: 0, repaired: 0, pastes: 0 };

// NodeId 在前后端之间可能是数字、字符串或带前缀的串，统一压成「最后一段数字」再比。
function keyOf(id) {
  if (id == null) return null;
  const text = String(id);
  const runs = text.match(/\d+/g);
  return runs && runs.length ? runs[runs.length - 1] : text;
}

function materialise(items) {
  if (!items) return [];
  return Array.isArray(items) ? items.slice() : [...items];
}

// ---- 第一层：把 input.link 按输入名补进剪贴板数据 ----
function injectInputLinks(sources, serialised) {
  if (!serialised?.nodes?.length) return 0;

  const sourceById = new Map();
  for (const item of sources) {
    if (item?.inputs && item.id != null) sourceById.set(keyOf(item.id), item);
  }

  let injected = 0;
  for (const nodeData of serialised.nodes) {
    const source = sourceById.get(keyOf(nodeData.id));
    if (!source) continue;

    const linkByName = new Map();
    for (const input of source.inputs) {
      if (input.link != null) linkByName.set(input.name, input.link);
    }
    if (!linkByName.size) continue;

    for (const input of nodeData.inputs ?? []) {
      const linkId = linkByName.get(input.name);
      if (linkId == null) continue;
      input.link = linkId;
      injected++;
    }
  }
  return injected;
}

// ---- 第二层：粘贴后按名补连 ----
function repairDroppedLinks(items, result) {
  const newByOldId = new Map();
  for (const [key, node] of result.nodes ?? []) {
    if (node) newByOldId.set(keyOf(key), node);
  }
  if (!newByOldId.size) return 0;

  const linksById = new Map();
  for (const link of items.links ?? []) linksById.set(keyOf(link.id), link);

  let repaired = 0;
  for (const nodeData of items.nodes ?? []) {
    const node = newByOldId.get(keyOf(nodeData.id));
    if (!node) continue;

    for (const serialisedInput of nodeData.inputs ?? []) {
      if (serialisedInput.link == null) continue;

      const linkInfo = linksById.get(keyOf(serialisedInput.link));
      if (!linkInfo) continue;

      // 对端不在这次粘贴的节点里，交给官方 connectInputs 逻辑，不越界干预
      const originNode = newByOldId.get(keyOf(linkInfo.origin_id));
      if (!originNode) continue;

      const slot = node.findInputSlot?.(serialisedInput.name);
      if (slot == null || slot === -1) continue;

      const target = node.inputs?.[slot];
      if (!target || target.link != null) continue;

      // Autogrow 的下一格要等上一格连上才长出来，所以按序列化顺序逐根连
      if (originNode.connect(linkInfo.origin_slot, node, slot)) repaired++;
    }
  }
  return repaired;
}

function install() {
  const canvas = app.canvas;
  const proto = canvas && Object.getPrototypeOf(canvas);
  if (!proto || typeof proto._serializeItems !== "function" || typeof proto._deserializeItems !== "function") {
    console.warn(TAG, "没找到 LGraphCanvas 的复制/粘贴方法，跳过（前端版本可能已变）");
    return false;
  }

  const originalSerialize = proto._serializeItems;
  const originalDeserialize = proto._deserializeItems;

  proto._serializeItems = function (items) {
    const sources = materialise(items ?? this.selectedItems);
    let serialised;
    try {
      serialised = originalSerialize.call(this, sources);
    } catch (error) {
      console.error(TAG, "序列化失败，按原样返回", error);
      return originalSerialize.call(this, sources);
    }
    try {
      const injected = injectInputLinks(sources, serialised);
      if (injected) stats.injected = injected;
    } catch (error) {
      console.error(TAG, "补写 input.link 失败", error);
    }
    return serialised;
  };

  proto._deserializeItems = function (items, options) {
    // 官方实现会就地改写 items（清空 input.link、重排 inputs、把 id 置 -1），
    // 所以巡检要用一份快照
    let snapshot = null;
    try {
      snapshot = JSON.parse(JSON.stringify(items));
    } catch (error) {
      console.error(TAG, "剪贴板数据无法快照，跳过巡检", error);
    }

    const result = originalDeserialize.call(this, items, options);

    stats.pastes++;
    if (snapshot && result) {
      try {
        const repaired = repairDroppedLinks(snapshot, result);
        if (repaired) {
          stats.repaired = repaired;
          console.info(`${TAG} 补回 ${repaired} 根粘贴时丢掉的连线`);
        }
      } catch (error) {
        console.error(TAG, "补连失败", error);
      }
    }
    return result;
  };

  globalThis[INSTALLED] = { installed: true, stats, tag: TAG };
  console.info(`${TAG} 已接管复制/粘贴：按输入名对齐，不再依赖槽位序号`);
  return true;
}

app.registerExtension({
  name: "DYF.PasteLinkFix",
  async setup() {
    if (globalThis[INSTALLED]) return; // 同一份脚本被加载两次时不重复包
    install();
  },
});
