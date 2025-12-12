# -*- coding: utf-8 -*-
from typing import Any, Dict, List
from openai import OpenAI
import os, json, math, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import traceback
"""
Prerequisites:
  export DASHSCOPE_API_KEY=sk-xxx
  (If you use the Singapore region, change base_url to: https://dashscope-intl.aliyuncs.com/compatible-mode/v1)
"""
#新加入了对不在允许答案列表中的类型进行过滤的逻辑，但是count里算上了这些物品，并且在数parked物品的时候不算上这些物品
# ===================== Scene backend (demo data; replace with your loader) =====================
SCENE_STORE = {
    0: {
        "objects": [
            {
                "type": "pedestrian",
                "position": [1182.112, 463.559, 3.078, 0.481, 0.5, 1.496, -1.899249838733],
                "attributes": "standing"},
            {
                "type": "car",
                "position": [1196.322, 467.415, 3.351, 1.697, 4.124, 1.647, 2.888258112657],
                "attributes": "parked"},
            {
                "type": "car",
                "position": [1197.172, 472.958, 3.257, 1.687, 4.493, 1.642, 2.783800156925],
                "attributes": "parked"},
        ],
        "ego_v_dir2d": [0.5965512726756697, 0.802574967880907],
        "ego_pose_translation": [1167.958187209624, 415.98221123790427, 0.0]}
}

# ===================== Session context (process-local; add real session isolation in prod) =====================
CURRENT_SCENE = {"uid": None, "objects": None, "ego_v_dir2d": None, "ego_pose": None}

EGO_TYPE = "ego"
ME_ALIASES = {"me", "ego", "ego-car"}
ALLOWED_ANSWERS = [
    "0","1","2","3","4","5","6","7","8","9","10",
    "no","yes",
    "barrier","bicycle","bus","car","construction vehicle","motorcycle",
    "moving","not standing","parked","pedestrian","standing","stopped",
    "traffic cone","trailer","truck","with rider","without rider","traffic_cone","construction_vehicle",
]

def _get_ego_index():
    objs = CURRENT_SCENE["objects"] or []
    for i, o in enumerate(objs):
        if o.get("type") == EGO_TYPE:
            return i
    return None

def bind_scene_from_sample(sample: Dict[str, Any]):
    """Bind a sample into CURRENT_SCENE and inject a synthetic ego if missing."""
    CURRENT_SCENE["uid"] = sample["uid"]
    CURRENT_SCENE["objects"] = list(sample["objects"])  # copy
    
    # 修复 ego_v_dir2d 的 None 值处理
    ego_v_dir2d = sample.get("ego_v_dir2d", [0.0, 1.0])
    if ego_v_dir2d is None:
        ego_v_dir2d = [0.0, 1.0]
    CURRENT_SCENE["ego_v_dir2d"] = list(ego_v_dir2d)
    
    # 修复 ego_pose_translation 的 None 值处理  
    ego_pose = sample.get("ego_pose_translation", [0.0, 0.0, 0.0])
    if ego_pose is None:
        ego_pose = [0.0, 0.0, 0.0]
    CURRENT_SCENE["ego_pose"] = list(ego_pose)

    # Inject unique ego at index 0, if absent
    if _get_ego_index() is None:
        ego_xy = (CURRENT_SCENE["ego_pose"] or [0.0, 0.0, 0.0])[:2]
        ego_obj = {
            "type": EGO_TYPE,
            # minimal BEV position: [x, y, z, width, length, height, yaw]
            "position": [ego_xy[0], ego_xy[1], 0.0, 0.0, 0.0, 0.0, 0.0],
            "attributes": ""
        }
        CURRENT_SCENE["objects"] = [ego_obj] + CURRENT_SCENE["objects"]
# ===================== Geometry & relation helpers =====================
def calculate_signed_angle_degrees(B1, B2, V_ego):
    """
    Signed angle in degrees in [-180, 180], using atan2(cross, dot).
    B1: candidate object position; B2: reference object position; V_ego: ego direction (x,y).
    """
    B1 = np.array(B1)
    B2 = np.array(B2)
    V_ego = np.array(V_ego)
    
    # 提取前两个分量（x, y）
    B1_2d = B1[:2]
    B2_2d = B2[:2]
    V_ego_2d = V_ego[:2]
    
    # 计算B1和B2之间的差向量
    diff_vector = B1_2d - B2_2d
    
    # 检查零向量情况
    if np.allclose(diff_vector, 0) or np.allclose(V_ego_2d, 0):
        raise ValueError("向量不能为零向量")
    
    # 使用atan2计算两个向量之间的有符号角度
    # atan2(cross_product, dot_product) 可以给出带符号的角度
    cross_product = diff_vector[0] * V_ego_2d[1] - diff_vector[1] * V_ego_2d[0]  # 2D叉积
    dot_product = np.dot(diff_vector, V_ego_2d)  # 点积
    
    # 计算有符号角度（弧度）
    theta_radians = np.arctan2(cross_product, dot_product)
    
    # 转换为度数
    theta_degrees = np.degrees(theta_radians)
    return float(theta_degrees)

def classify_relation(theta_deg: float) -> str:
    """
    Right-closed intervals matching your spec:
      (-30, 30] -> front
      (30, 90]  -> front_left
      (-90,-30] -> front_right
      (90,150]  -> back_left
      (-150,-90]-> back_right
      otherwise -> back
    """
    if -30 < theta_deg <= 30:   return "front"
    if  30 < theta_deg <= 90:   return "front_left"
    if -90 < theta_deg <= -30:  return "front_right"
    if  90 < theta_deg <= 150:  return "back_left"
    if -150 < theta_deg <= -90: return "back_right"
    return "back"

def euclid_xy(a, b):  # x,y only
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ===================== Internal selectors =====================
def _normalize_ego_aliases_in_selector(selector: dict) -> dict:
    """Map 'me'/'ego-car' aliases to the canonical EGO_TYPE."""
    if not selector:
        return selector
    t = selector.get("type")
    if isinstance(t, str) and t.strip().lower() in ME_ALIASES:
        return {**selector, "type": EGO_TYPE}
    return selector


def _select_by_type_status(selector):
    selector = dict(selector or {})
    t_norm, s_norm = normalize_type_and_status(selector.get("type"), selector.get("status"))
    objs = CURRENT_SCENE["objects"]
    cands = []
    for i, o in enumerate(objs):
        o_type = normalize_label(o.get("type",""))
        o_stat = normalize_label(o.get("attributes",""))
        if t_norm != "ego" and o_type == "ego":
            continue
        if t_norm and o_type != t_norm:
            continue
        if s_norm and o_stat != s_norm:
            continue
        if o_type not in ALLOWED_ANSWERS_NORM and s_norm == "parked":
            continue
        cands.append(i)
    return cands

def _select_by_index_or_filter(selector: dict):
    selector = selector or {}
    if selector.get("index") is not None:
        idx = int(selector["index"])
        objs = CURRENT_SCENE["objects"]
        return [idx] if 0 <= idx < len(objs) else []
    return _select_by_type_status(selector)

# ===================== Six tool implementations (read CURRENT_SCENE only) =====================
def tool_intersect_ctx(args: dict) -> str:
    """
    Intersect two index lists, preserving the order of 'base'.
    Safe-guards:
      - validate ints and scene bounds
      - dedup base while keeping order
      - return {"indices": [...]} ; idempotent and side-effect free
    """
    objs = CURRENT_SCENE.get("objects", [])
    n = len(objs)

    def _sanitize(arr):
        out = []
        for v in (arr or []):
            try:
                i = int(v)
            except Exception:
                continue
            if 0 <= i < n:
                out.append(i)
        return out

    base  = _sanitize(args.get("base"))
    other = set(_sanitize(args.get("other")))

    # 若有一边为空，交集必空 —— 这是“失败安全”，避免误用产生虚假结果
    if not base or not other:
        return json.dumps({"indices": []})

    # base 去重但保序（避免 LLM 传入重复）
    seen = set()
    base_unique = []
    for i in base:
        if i not in seen:
            seen.add(i)
            base_unique.append(i)

    out = [i for i in base_unique if i in other]
    return json.dumps({"indices": out})

def tool_find_by_relation_ctx(args: Dict[str, Any]) -> str:
    """Return ALL objects that are in the given relation from the (first) reference hit.
    Filters type/status (with global normalization) and sorts by distance asc. Optional max_k truncation.
    """
    objs = CURRENT_SCENE["objects"]
    ego  = CURRENT_SCENE["ego_v_dir2d"]

    # resolve reference (use first match if multiple)
    ref_idxs = _select_by_index_or_filter(args.get("ref_selector", {}))
    if not ref_idxs:
        return json.dumps({"objects_idx": [], "objects": [], "ref_index": None})
    ref_idx = ref_idxs[0]
    desired = args["relation"]

    # normalize filters
    t_norm, s_norm = normalize_type_and_status(args.get("type_filter"), args.get("status_filter"))

    ref_xy = objs[ref_idx]["position"][:2]

    hits = []
    for i, o in enumerate(objs):
        if i == ref_idx:
            continue
        if normalize_label(o.get("type","")) == "ego":
            continue

        vec = (o["position"][0]-ref_xy[0], o["position"][1]-ref_xy[1])
        if vec == (0,0):
            continue
        # use ego orientation for angle-binning
        theta_deg = calculate_signed_angle_degrees(o["position"], objs[ref_idx]["position"], ego)
        res = classify_relation(theta_deg)
        if res != desired:
            continue

        ot = normalize_label(o.get("type",""))
        os = normalize_label(o.get("attributes",""))
        if t_norm and ot != t_norm:
            continue
        if s_norm and os != s_norm:
            continue

        hits.append({"index": i, "type": o.get("type",""), "status": o.get("attributes",""), "dist": euclid_xy(o["position"], ref_xy)})

    if not hits:
        return json.dumps({"objects_idx": [], "objects": [], "ref_index": ref_idx})

    hits_sorted = sorted(hits, key=lambda h: h["dist"])
    max_k = args.get("max_k")
    if isinstance(max_k, int):
        hits_sorted = hits_sorted[:max_k]
    return json.dumps({
        "objects_idx": [h["index"] for h in hits_sorted],
        "objects": [{"index": h["index"], "type": h["type"], "status": h["status"]} for h in hits_sorted],
        "ref_index": ref_idx
    })

def tool_count_ctx(args: Dict[str, Any]) -> str:
    """
    Count with either:
      (A) indices: list[int]  -> directly count those (ignore filters)
      (B) type_filter/status_filter -> search & count matches

    Returns:
      {"count": int, "objects_idx": [int, ...]}
    """
    # --- Path A: direct counting over provided indices
    if args.get("indices") is not None:
        raw = args.get("indices") or []
        # validate & clamp to scene bounds
        n = len(CURRENT_SCENE["objects"])
        idxs = []
        for v in raw:
            try:
                i = int(v)
                if 0 <= i < n:
                    idxs.append(i)
            except Exception:
                continue  # skip bad entries silently
        return json.dumps({"count": len(idxs), "objects_idx": idxs})

    # --- Path B: fall back to type/status matching
    t_norm, s_norm = normalize_type_and_status(args.get("type_filter"), args.get("status_filter"))
    objs = CURRENT_SCENE["objects"]
    idxs = []
    for i, o in enumerate(objs):
        ot = normalize_label(o.get("type", ""))
        os = normalize_label(o.get("attributes", ""))
        if t_norm and ot != t_norm:
            continue
        if s_norm and os != s_norm:
            continue
        if s_norm == "parked" and (ot in {"barrier", "trafficcone"} or ot not in ALLOWED_ANSWERS_NORM):
            continue
        # if ot == "ego" and t_norm != "ego":
        #     # skip ego unless explicitly counting ego
        #     continue
        idxs.append(i)
    return json.dumps({"count": len(idxs), "objects_idx": idxs})

def tool_exists_ctx(args: dict) -> str:
    """
    Existence check (0/1-hop via ref_selector + relation).
    Params: {type_filter?, status_filter?, ref_selector?, relation?}
    Returns JSON string:
      { "answer":"yes|no", "count":int, "objects_idx":[...] }
    """
    payload = json.loads(tool_count_ctx(args))
    return json.dumps({"answer": "yes" if payload["count"] > 0 else "no",
                       "count": payload["count"],
                       "objects_idx": payload["objects_idx"]})


def tool_get_type_ctx(args: Dict[str, Any]) -> str:
    """
    Return ALL matching objects' indices and types.

    Two modes:
      A) indices-mode: args.indices = [int,...] (order preserved)
         Optional filters: args.type_filter, args.status_filter
      B) selector-mode: args.selector = {index? | type?, status?}

    Returns:
      {"objects":[{"index","type"}, ...], "indices":[...]}
    """
    objs = CURRENT_SCENE["objects"]

    # ---- A) indices-mode ----
    if args.get("indices") is not None:
        raw = args.get("indices") or []
        n = len(objs)
        idxs = []
        for v in raw:
            try:
                i = int(v)
                if 0 <= i < n:
                    idxs.append(i)
            except Exception:
                continue

        # optional target-side filter on provided indices
        t_norm, s_norm = normalize_type_and_status(args.get("type_filter"), args.get("status_filter"))
        if t_norm or s_norm:
            filtered = []
            for i in idxs:
                ot = normalize_label(objs[i].get("type",""))
                os = normalize_label(objs[i].get("attributes",""))
                if t_norm and ot != t_norm: 
                    continue
                if s_norm and os != s_norm: 
                    continue
                if ot not in ALLOWED_ANSWERS_NORM:
                    continue
                filtered.append(i)
            idxs = filtered

        items = [{"index": i, "type": objs[i].get("type","") or ""} for i in idxs]
        return json.dumps({"objects": items, "indices": idxs})

    # ---- B) selector-mode (legacy) ----
    sel = dict(args.get("selector", {}) or {})
    if not sel:
        if args.get("type_filter") or args.get("status_filter"):
            sel = {
                "type": args.get("type_filter",''),
                "status": args.get("status_filter",'')
            }
    t_norm, s_norm = normalize_type_and_status(sel.get("type"), sel.get("status"))
    sel["type"] = t_norm
    sel["status"] = s_norm
    indices = _select_by_index_or_filter(sel)
    items = [{"index": i, "type": objs[i].get("type","") or ""} for i in indices]
    return json.dumps({"objects": items, "indices": indices})


def tool_get_status_ctx(args: Dict[str, Any]) -> str:
    """
    Return ALL matching objects' indices and statuses.

    Modes identical to get_type_ctx (indices-mode preferred if provided).
    Returns:
      {"objects":[{"index","status"}, ...], "indices":[...]}
    """
    objs = CURRENT_SCENE["objects"]

    # ---- A) indices-mode ----
    if args.get("indices") is not None:
        raw = args.get("indices") or []
        n = len(objs)
        idxs = []
        for v in raw:
            try:
                i = int(v)
                if 0 <= i < n:
                    idxs.append(i)
            except Exception:
                continue

        # optional target-side filter on provided indices
        t_norm, s_norm = normalize_type_and_status(args.get("type_filter"), args.get("status_filter"))
        if t_norm or s_norm:
            filtered = []
            for i in idxs:
                ot = normalize_label(objs[i].get("type",""))
                os = normalize_label(objs[i].get("attributes",""))
                if t_norm and ot != t_norm: 
                    continue
                if s_norm and os != s_norm: 
                    continue
                filtered.append(i)
            idxs = filtered

        items = [{"index": i, "status": objs[i].get("attributes","") or ""} for i in idxs]
        return json.dumps({"objects": items, "indices": idxs})

    # ---- B) selector-mode (legacy) ----
    sel = dict(args.get("selector", {}) or {})
    if not sel:
        if args.get("type_filter") or args.get("status_filter"):
            sel = {
                "type": args.get("type_filter",''),
                "status": args.get("status_filter",'')
            }
    t_norm, s_norm = normalize_type_and_status(sel.get("type"), sel.get("status"))
    sel["type"] = t_norm
    sel["status"] = s_norm
    indices = _select_by_index_or_filter(sel)
    items = [{"index": i, "status": objs[i].get("attributes","") or ""} for i in indices]

    
    return json.dumps({"objects": items, "indices": indices})

def tool_compare_status_ctx(args: dict) -> str:
    """
    Compare whether two sets of objects have the same *concrete* status.
    Priority:
      1) if a_indices / b_indices provided (non-empty) -> use them (no selector expansion)
      2) else resolve a_selector / b_selector via _select_by_index_or_filter
      3) if either side empty -> no match (fail-safe, don't expand to whole scene)

    Matching rule:
      - normalize status via normalize_label
      - treat "" / "any" (and None) as empty -> do NOT match on empty statuses
      - build pairs where normalized statuses are equal and non-empty
      - preserve provided indices as-is (0-based), sanitize out-of-range
    """
    objs = CURRENT_SCENE.get("objects", [])
    n = len(objs)

    def _sanitize_indices(arr):
        out = []
        for v in (arr or []):
            try:
                i = int(v)
            except Exception:
                continue
            if 0 <= i < n:
                out.append(i)
        return out

    def _norm_sel(sel):
        sel = dict(sel or {})
        t_norm, s_norm = normalize_type_and_status(sel.get("type"), sel.get("status"))
        sel["type"] = t_norm
        sel["status"] = s_norm
        return sel

    def _norm_status(i):
        raw = objs[i].get("attributes", "")
        s = normalize_label(raw) if raw is not None else ""
        # treat empty/any as no-status => not used for equality
        if s in ("", "any"):
            return ""
        return s

    # ---- 1) Prefer indices-mode if provided ----
    a_list = _sanitize_indices(args.get("a_indices"))
    b_list = _sanitize_indices(args.get("b_indices"))

    # ---- 2) Fallback to selectors if indices not provided ----
    if not a_list:
        a_selector = _norm_sel(args.get("a_selector", {}))
        a_list = _sanitize_indices(_select_by_index_or_filter(a_selector))
    if not b_list:
        b_selector = _norm_sel(args.get("b_selector", {}))
        b_list = _sanitize_indices(_select_by_index_or_filter(b_selector))

    # ---- 3) Fail-safe: if either side empty, don't expand to whole scene ----
    if not a_list or not b_list:
        return json.dumps({
            "a_indices": a_list,
            "a_statuses": [objs[i].get("attributes", "") or "" for i in a_list],
            "b_indices": b_list,
            "b_statuses": [objs[j].get("attributes", "") or "" for j in b_list],
            "pairs_same_status": [],
            "same": False
        })

    # ---- 4) Compute pairs with same *concrete* status ----
    a_stats_norm = [_norm_status(i) for i in a_list]
    b_stats_norm = [_norm_status(j) for j in b_list]

    # index b by status for efficient pairing
    b_by_status = {}
    for j, sb in zip(b_list, b_stats_norm):
        if not sb:  # skip empty status
            continue
        b_by_status.setdefault(sb, []).append(j)

    pairs = []
    for i, sa in zip(a_list, a_stats_norm):
        if not sa:  # skip empty status
            continue
        for j in b_by_status.get(sa, []):
            pairs.append([i, j])

    # raw (un-normalized) statuses for transparency
    a_statuses_raw = [objs[i].get("attributes", "") or "" for i in a_list]
    b_statuses_raw = [objs[j].get("attributes", "") or "" for j in b_list]

    return json.dumps({
        "a_indices": a_list,
        "a_statuses": a_statuses_raw,
        "b_indices": b_list,
        "b_statuses": b_statuses_raw,
        "pairs_same_status": pairs,  # empty if none
        "same": bool(pairs)
    })
# === Updated tool schemas (ALL-return) ===
TOOLS = [
    {
  "type": "function",
  "function": {
    "name": "intersect_ctx",
    "description": "Intersect two index lists, preserving the order of 'base'. Use only on indices returned by previous tools.",
    "parameters": {
      "type": "object",
      "properties": {
        "base":  { "type": "array", "items": { "type": "integer" }, "minItems": 1, "maxItems": 1024 },
        "other": { "type": "array", "items": { "type": "integer" }, "minItems": 1, "maxItems": 16384 }
      },
      "required": ["base","other"]
    }
  }
},
    {
        "type": "function",
        "function": {
            "name": "find_by_relation_ctx",
            "description": "Return ALL objects that satisfy a spatial relation from a reference object. Use ONLY when a relation is explicitly specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_selector": {
                        "type": "object",
                        "description": "Identify the reference object (by index or by type/status).",
                        "properties": {
                            "index":  {"type": ["integer", "null"]},
                            "type":   {"type": ["string",  "null"]},
                            "status": {"type": ["string",  "null"]}
                        }
                    },
                    "relation": {
                        "type": "string",
                        "enum": ["front", "back", "front_left", "front_right", "back_left", "back_right"]
                    },
                    "type_filter":   {"type": ["string","null"], "description": "Optional target type filter"},
                    "status_filter": {"type": ["string","null"], "description": "Optional target status filter"},
                    "max_k":         {"type": ["integer","null"], "description": "Optional cutoff after distance sorting"}
                },
                "required": ["ref_selector", "relation"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "exists_ctx",
            "description": "Existence check via type/status filters OR directly over a provided list of indices (indices-mode).",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_filter":   {"type": ["string","null"]},
                    "status_filter": {"type": ["string","null"]},
                    "indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"},
                        "description": "If provided, filters are ignored and existence is tested over these indices."
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "count_ctx",
            "description": "Count via type/status filters OR directly over a provided list of indices (indices-mode).",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_filter":   {"type": ["string","null"]},
                    "status_filter": {"type": ["string","null"]},
                    "indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"},
                        "description": "If provided, filters are ignored and the count is taken over these indices."
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_type_ctx",
            "description": "Return ALL matching objects' indices and types. Prefer indices-mode if 'indices' is provided; otherwise use 'selector'. Optional target-side filters apply in indices-mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"},
                        "description": "Operate on these indices (order preserved)."
                    },
                    "selector": {
                        "type": ["object","null"],
                        "description": "Fallback selector-mode when 'indices' is not provided.",
                        "properties": {
                            "index":  {"type": ["integer","null"]},
                            "status": {"type": ["string","null"]}
                        }
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_status_ctx",
            "description": "Return ALL matching objects' indices and statuses. Prefer indices-mode if 'indices' is provided; otherwise use 'selector'. Optional target-side filters apply in indices-mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"},
                        "description": "Operate on these indices (order preserved)."
                    },
                    "selector": {
                        "type": ["object","null"],
                        "description": "Fallback selector-mode when 'indices' is not provided.",
                        "properties": {
                            "index":  {"type": ["integer","null"]},
                            "type":   {"type": ["string","null"]},
                        }
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "compare_status_ctx",
            "description": "Compare statuses between two sets. Prefer indices-mode if a_indices/b_indices are provided; otherwise use selectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a_indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"}
                    },
                    "b_indices": {
                        "type": ["array","null"],
                        "items": {"type": "integer"}
                    },
                    "a_selector": {
                        "type": ["object","null"],
                        "properties": {
                            "index":  {"type": ["integer","null"]},
                            "type":   {"type": ["string","null"]},
                            "status": {"type": ["string","null"]}
                        }
                    },
                    "b_selector": {
                        "type": ["object","null"],
                        "properties": {
                            "index":  {"type": ["integer","null"]},
                            "type":   {"type": ["string","null"]},
                            "status": {"type": ["string","null"]}
                        }
                    }
                },
                "required": []
            }
        }
    },

]
# Runtime dispatch map
DISPATCH = {
    "find_by_relation_ctx": tool_find_by_relation_ctx,
    "count_ctx": tool_count_ctx,
    "exists_ctx": tool_exists_ctx,
    "get_type_ctx": tool_get_type_ctx,
    "get_status_ctx": tool_get_status_ctx,
    "compare_status_ctx": tool_compare_status_ctx,
    "intersect_ctx": tool_intersect_ctx,
}
client = OpenAI(
    api_key="sk-709609f9bf95438ba6408edd8273f756",
    base_url="http://localhost:8000/v1",  # change for Singapore region if needed
)

def get_response(messages):
    return client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
        top_p=1.0,
        #max_tokens=4096,
        #stream=False  # 使用流式输出
    )
SYSTEM_PROMPT="""
You are a Scene-QA solver.
You cannot see raw scene data; you must use tools to retrieve facts from the currently bound scene context.
## Tools (read-only over the bound scene)
* **find_by_relation_ctx**(ref_selector, relation, type_filter?, status_filter?, **max_k?**) → indices of objects at a relation from a reference. (only use when a relation is specified)
* **exists_ctx**(type_filter?, status_filter?) → {answer yes/no, count, indices}.
* **count_ctx**(type_filter?, status_filter?, indices?) → {count, indices}.
* **get_type_ctx**(selector (**no parameter is "type"**)) → {**objects:[{index,type},...], indices**}.
* **get_status_ctx**(selector) → {**objects:[{index,status},...], indices**}.
* **compare_status_ctx**(a_selector, b_selector) → {**a_indices, a_statuses, b_indices, b_statuses, pairs_same_status**}.
* **intersect_ctx**(base, other) → {**indices**}. you can not use this tool before you get indices from other tools.

> Selectors may identify by **index** or by **(type, status)**.  
> Relation lookup must use **find_by_relation_ctx**; other tools do **not** accept relation arguments.
> You should use tools step by step.

## Routing Rules
### Conventions
* "any" or null = no filter. Normalize labels (lowercase; ignore spaces/underscores; sitting_lying_down→not_standing).
* Never return the ego as a relation target.
* You can use **find_by_relation_ctx** only if a relation (e.g., front, back, left, right) and a reference are specified.
* "type" means object category (car, truck, pedestrian, etc.); "status" means attributes (moving, standing, parked, etc.). Please do not confuse them, and do not add any limit word.
* If a type or a status is not specified, do not add any filter on it, and do not make any assumption.
* Angle buckets/relations follow the implementation’s right-closed intervals; do not reinterpret them in text.
* "Visible" is not a status, so it means 'any'.
* Do not use reference objects in comparisons.

## 1) Existence (yes/no)
**h0 (direct):**  
"Are there any parked trucks?"  
→ `exists_ctx(type_filter="truck", status_filter="parked")`
"Is there another thing as the same status as the truck?"
→ 'get_status_ctx(selector={type:"truck"})' → get status S of first truck
→ `exists_ctx(status_filter=S)`
**h1 (via relation):**  
"Is there a car to the front of the standing pedestrian?"  
→ `find_by_relation_ctx(ref_selector={type:"pedestrian", status:"standing"}, relation="front", type_filter="car")`  
→ if `objects_idx` is non-empty, answer "yes", else "no".
"Is there any object that is both in front of the traffic cone and front-left of any stopped object?"
→A = find_by_relation_ctx(ref_selector={type:"traffic cone"},relation="front")  
→B = find_by_relation_ctx(ref_selector={status:"stopped"},relation="front_left")  
→ intersect_ctx(base=A.objects_idx, other=B.objects_idx)  
→ if `indices` is non-empty, answer "yes", else "no".

## 2) Counting (number)
**h0 (direct):**  
"How many cars are parked?"  
→ `count_ctx(type_filter="car", status_filter="parked")`
"How many pedestrians are there?"
→ `count_ctx(type_filter="pedestrian")`
"How many objects are standing?"
→ `count_ctx(status_filter="standing")`
**h1 (via relation):**  
"How many cars are to the front-left of the standing pedestrian?"
→ find_by_relation_ctx(ref_selector={type:"pedestrian", status:"standing"}, relation="front_left", type_filter="car")
→ count_ctx(indices=<objects_idx from the previous step>)
"How many cars are both in front of the traffic cone and front-left of any stopped object?"
→A = find_by_relation_ctx(ref_selector={type:"traffic cone"},relation="front", type_filter="car")  
→B = find_by_relation_ctx(ref_selector={status:"stopped"},relation="front_left", type_filter="car")  
→ intersect_ctx(base=A.objects_idx, other=B.objects_idx)  
→ count_ctx(indices=<indices from the previous step>)

## 3) Query-Object (what is ...?/ ...is what? → type)
**h0 (direct selector, no relation):**  
"The parked object—what is it?"
"The parked thing is what?"  
→ `get_type_ctx(selector={status:"parked"})`  
→ take the first index from `indices` and output its `type`.
**h1 (locate by relation, then get type):**  
"There is an object to the back-right of the bus; what is it?"  
→ `find_by_relation_ctx(ref_selector={type:"bus"}, relation="back_right")`  
→ `get_type_ctx(selector={index:<first objects_idx item>})`
"There is a thing that is in front of the traffic cone and front-left of the stopped thing; what is it?"
→A = find_by_relation_ctx(ref_selector={type:"traffic cone"},relation="front")  
→B = find_by_relation_ctx(ref_selector={status:"stopped"},relation="front_left")  
→ C = intersect_ctx(base=A.objects_idx, other=B.objects_idx)  
→ `get_type_ctx(selector={index:<first C.indices item>})` if the target must have a specific status,use status_filter to filter it.

## 4) Query-Status (what status is it? → status)
**h0 (direct selector, no relation):**  
"The nearest car—what status is it?"  
→ `get_status_ctx(selector={type:"car"})`  
→ take the first index from `indices` and output its `status`.
**h1 (locate by relation, then get status):**  
"The object in front of the standing pedestrian; what status is it?"  
→ `find_by_relation_ctx(ref_selector={type:"pedestrian", status:"standing"}, relation="front")`  
→ `get_status_ctx(selector={index:<first objects_idx item>})`
"What status is the bicycle that is both in front of the traffic cone and front-left of any stopped object?"
→A = find_by_relation_ctx(ref_selector={type:"traffic cone"},relation="front", type_filter="bicycle")
→B = find_by_relation_ctx(ref_selector={status:"stopped"},relation="front_left", type_filter="bicycle")
→ C = intersect_ctx(base=A.objects_idx, other=B.objects_idx)
→ `get_status_ctx(selector={index:<first C.indices item>})`

## 5) Comparison (same status?)

**h0 (both picked directly):**  
"Is the nearest car the same status as the nearest truck?"  
→ `get_status_ctx(selector={type:"car"})` and `get_status_ctx(selector={type:"truck"})`  
→ `compare_status_ctx(a_selector={type:"car"}, b_selector={type:"truck"})`
**h1 (both via relations):**  
"Is the car front-left of the standing pedestrian the same status as the car in front of the standing pedestrian?"  
→ A: `find_by_relation_ctx(ref_selector={type:"pedestrian", status:"standing"}, relation="front_left", type_filter="car")` → take first index as A  
→ B: `find_by_relation_ctx(ref_selector={type:"pedestrian", status:"standing"}, relation="front",      type_filter="car")` → take first index as B  
→ `compare_status_ctx(a_selector={index:A}, b_selector={index:B})`

## Special handling: **ego-car referred to as "me"**

* Treat **"me"** as the ego object; use `ref_selector={type:"ego"}` or `selector={type:"ego"}` when the question refers to "me".
* "ego-car" or "me" will never be an answer.
* **When "me" is the reference** (e.g., "in front of me"): call tools with `ref_selector={type:"ego"}`; relation is computed from ego forward.
* Never expose internal IDs, coordinates, or tool arguments.

## Status
status includes: "moving","stopped","parked","with_rider","without_rider","not_standing","standing"

## Output style
* Only output the final answer text to the user; 
* for comparisons and existences, say "yes" or "no".
* for counting, output the integer only, 
* for query_status/object, output the status or type only. 
* All of your answers must be in lowercase without punctuation. Only output one or two words (e.g., "yes", "no", "3", "standing", "car").
* Never return "ego" or "me" as an answer.
"""
def normalize_label(s: str) -> str:
    """
    - Case-insensitive
    - Spaces and underscores are equivalent
    - sitting_lying_down -> not_standing
    """
    if s is None:
        return ""
    x = s.strip().lower()
    x = x.replace("_", " ")
    x = x.replace(" ", "")  # remove all spaces so "_" and " " are equivalent
    if x == "sittinglyingdown":
        x = "notstanding"
    return x

ALLOWED_ANSWERS_NORM={normalize_label(a) for a in ALLOWED_ANSWERS}
# ===== Added: Global normalization for type/status =====
_RAW_STATUS_LABELS = {
    "moving", "parked", "stopped", "standing",
    "not_standing", "with_rider", "without_rider"
}
_STATUS_CANON = { normalize_label(x) for x in _RAW_STATUS_LABELS }

def is_status_label(label: str) -> bool:
    if not label:
        return False
    n = normalize_label(label)
    return n in _STATUS_CANON

def normalize_type_and_status(t_raw, s_raw):
    """
    Normalize (type, status) with robust fallbacks:
      - type: "thing"/"any"/"" -> None
      - status: "any"/"" -> None
      - if a status-like token was put in type and status is empty -> move to status
      - if status is not a recognized status label -> None (fallback)
      - ego aliases -> "ego"
      - if remaining type starts with "not" (e.g., "notcar") -> None (fallback)
    """
    # base normalization (lowercase; remove spaces/underscores; special map for sitting_lying_down)
    t_norm = normalize_label(t_raw) if t_raw is not None else None
    s_norm = normalize_label(s_raw) if s_raw is not None else None

    # "any"/"thing"/empty -> no filter
    if t_norm in {"any", "", "thing","vehicle","object"}:
        t_norm = None
    if s_norm in {"any", ""}:
        s_norm = None

    # move status mistakenly placed in type -> status
    if (s_norm is None) and is_status_label(t_norm):
        s_norm = t_norm
        t_norm = None
    if is_status_label(t_norm):
        t_norm = None
    # Fallback: if status is not a recognized status label, drop it
    if s_norm is not None and not is_status_label(s_norm):
        s_norm = None

    # ego aliases
    if t_norm in {"me", "ego", "egocar", "ego-car"}:
        t_norm = "ego"

    # Fallback: if remaining type expresses a negation (starts with "not"), drop it
    if t_norm is not None and t_norm.startswith("not"):
        t_norm = None

    return t_norm, s_norm

# ===== End Added =====
def answers_match(model_answer: str, gold_answer: str) -> bool:
    return normalize_label(model_answer) == normalize_label(gold_answer)
def parse_final_answer(final_answer: str) -> str:
    """
    解析最终答案，处理两种情况：
    1. JSON格式的工具调用或直接答案：需要解析并执行/提取
    2. 纯文字答案：直接返回
    
    Args:
        final_answer: 从assistant.content获取的最终答案字符串
        
    Returns:
        处理后的最终答案
    """
    final_answer = final_answer.strip()
    
    # 尝试解析为JSON
    try:
        parsed = json.loads(final_answer)
        
        # 情况1: 工具调用格式 {"name": "count_ctx", "arguments": {...}}
        if "name" in parsed and "arguments" in parsed:
            tool_name = parsed["name"]
            arguments = parsed["arguments"]
            
            # 检查工具是否存在
            if tool_name not in DISPATCH:
                return final_answer  # 如果工具不存在，返回原始字符串
            
            try:
                # 调用工具获取结果
                result_str = DISPATCH[tool_name](arguments)
                result_dict = json.loads(result_str)
                
                # 根据工具类型提取最终答案
                if tool_name == "count_ctx":
                    return str(result_dict.get("count", 0))
                elif tool_name == "exists_ctx":
                    return result_dict.get("answer", "no")
                elif tool_name == "get_type_ctx":
                    objects = result_dict.get("objects", [])
                    if objects:
                        return objects[0].get("type", "unknown").lower()
                    return "none"
                elif tool_name == "get_status_ctx":
                    objects = result_dict.get("objects", [])
                    if objects:
                        return objects[0].get("status", "unknown").lower()
                    return "none"
                elif tool_name == "compare_status_ctx":
                    pairs = result_dict.get("pairs_same_status", [])
                    return "yes" if pairs else "no"
                elif tool_name == "find_by_relation_ctx":
                    objects_idx = result_dict.get("objects_idx", [])
                    return "yes" if objects_idx else "no"
                elif tool_name == "intersect_ctx":
                    indices = result_dict.get("indices", [])
                    return "yes" if indices else "no"
                else:
                    return str(result_dict)
            except Exception:
                return final_answer  # 如果工具调用失败，返回原始字符串
        
        # 情况2: 直接答案格式 {"answer": "no"}
        elif "answer" in parsed:
            return parsed["answer"]
        
        # 其他JSON格式，返回原始字符串
        else:
            return final_answer
            
    except json.JSONDecodeError:
        # 不是JSON格式，直接返回（这是纯文字答案如'no', 'yes', '1', 'car'）
        return final_answer
    except Exception:
        # 其他异常，返回原始字符串
        return final_answer
def run_one_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Bind scene
    bind_scene_from_sample(sample)

    # LLM sees only the question
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["question"]+"\n /no_think"}]

    step_logs = []  # each round: list of tool_calls and corresponding tool_results
    resp = get_response(messages)
    assistant = resp.choices[0].message
    if assistant.content is None:
        assistant.content = ""
    turns=0
    p=False
    while assistant.tool_calls:
        # 第一段：先修改 assistant.tool_calls（兜底转换），不执行
        for tc in assistant.tool_calls:
            try:
                name = getattr(tc.function, 'name', None)
                args = json.loads(getattr(tc.function, 'arguments', None) or "{}")
            except Exception:
                name, args = None, {}

            try:
                if name == "get_status_ctx" and isinstance(args, dict):
                    sel = args.get("selector") if isinstance(args.get("selector"), dict) else None
                    has_status = False
                    type_token = None

                    if sel is not None:
                        if sel.get("status") is not None and sel.get("status") != "":
                            has_status = True
                        type_token = sel.get("type")

                    if not has_status and args.get("status_filter") is not None and args.get("status_filter") != "":
                        has_status = True
                        type_token = args.get("type_filter")

                    def _is_thing_or_any(t):
                        if t is None:
                            return True
                        if isinstance(t, str):
                            tt = t.strip().lower()
                            return tt in {"thing", "any", "", "null"}
                        return False

                    if has_status and _is_thing_or_any(type_token):
                        status_val = None
                        if sel is not None and sel.get("status") is not None and sel.get("status") != "":
                            status_val = sel.get("status")
                        else:
                            status_val = args.get("status_filter")

                        if status_val is not None and status_val != "":
                            new_name = "get_type_ctx"
                            new_args = {"selector": {"status": status_val}}
                            # 就地覆盖 assistant.tool_calls 当前项
                            try:
                                tc.function.name = new_name
                                tc.function.arguments = json.dumps(new_args)
                                p=True
                            except Exception:
                                try:
                                    tc.function['name'] = new_name
                                    tc.function['arguments'] = json.dumps(new_args)
                                except Exception:
                                    pass
            except Exception:
                pass

        # 第二步：把已修改的 assistant 放入消息
        if p:
            print(assistant)
            p=False
        messages.append(assistant)

        # 第三段：执行修改后的 tool_calls，并记录日志
        round_log = {"tool_calls": [], "tool_results": []}
        for tc in assistant.tool_calls:
            name = getattr(tc.function, 'name', None)
            try:
                args = json.loads(getattr(tc.function, 'arguments', None) or "{}")
            except Exception:
                args = {}

            round_log["tool_calls"].append({"name": name, "arguments": args})
            result = DISPATCH[name](args)
            try:
                result_json = json.loads(result)
            except Exception:
                result_json = result
            round_log["tool_results"].append({"name": name, "result": result_json})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        step_logs.append(round_log)

        # 拉取下一轮回答
        resp = get_response(messages)
        assistant = resp.choices[0].message
        if assistant.content is None:
            assistant.content = ""
        turns += 1
        if turns >= 5:
            break  # safety stop

    final_answer = parse_final_answer(assistant.content)
    gold = sample.get("gold_answer", "")
    match = answers_match(final_answer, gold)

    return {
        "uid": sample["uid"],
        "sample_token": sample.get("sample_token", ""),
        "question": sample["question"],
        "gold_answer": gold,
        "model_answer": final_answer,
        "match": match,
        "steps": step_logs,
        "turns": turns}
def run_batch(dataset: List[Dict[str, Any]], out_path: str = "results.json", step: int = 5):
    """Run evaluation over dataset with optional subsampling stride `step`.

    Returns the list of per-sample result dicts.
    """
    results = []
    # 支持 step 为 1（全部）或更大（抽样）
    idx=0
    total=len(dataset)
    for sample in dataset[::step]:
        try:
            result = run_one_sample(sample)
        except Exception as e:
            # 把错误信息记录下来
            err_msg = f"{type(e).__name__}: {e}"
            err_trace = traceback.format_exc()

            print(f"[ERROR] uid={sample.get('uid')} sample_token={sample.get('sample_token')} -> {err_msg}")
            # 如果你不想打印堆栈，可以把下一行删掉
            print(err_trace)

            # 用和 run_one_sample 同样结构构造一个结果
            result = {
                "uid": sample.get("uid"),
                "sample_token": sample.get("sample_token", ""),
                "question": sample.get("question", ""),
                "gold_answer": sample.get("gold_answer", ""),
                "model_answer": f"ERROR: {err_msg}",
                "match": False,                         # 按你说的，出错一律 False
                "steps": [
                    {
                        "error": err_msg,
                        "traceback": err_trace,
                    }
                ],
                "turns": [],                            # 没有正常对话，就给空
            }

        results.append(result)
        idx += 1
        if idx % 1000 == 0 or idx==total:
            print(f"Processed {idx}/{total} samples...")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} records to {out_path}")
    return results
# ===================== Demo: LLM sees only the question =====================
if __name__ == "__main__":
    # 无需命令行输入：默认从 build_llm_prompts 的输出目录读取处理好的问题文件
    # 约定：input_dir 包含 5 类 * 2 层 = 10 个文件（*_h0.json, *_h1.json）
    input_dir = Path("dataset/extract_data/question_processed_p09r03")
    # 将结果保存在 input_dir 的旁边目录，保持组织清晰
    out_dir = input_dir.parent / (input_dir.name + "_results")
    step = 1  # 默认抽样步长（与之前保持一致）；如需全部评估可在代码中改成 1

    # 创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只处理 *_h0.json 和 *_h1.json，按文件名排序
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and (p.name.endswith("_h0.json") or p.name.endswith("_h1.json"))])
    if not files:
        print(f"No *_h0.json or *_h1.json files found in {input_dir}")
        raise SystemExit(1)

    metrics = {}
    total_processed = 0
    total_correct = 0

    for src in files:
        try:
            with open(src, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {src}: {e}")
            continue

        basename = src.stem
        result_file = out_dir / f"results_{basename}.json"
        print(f"Processing {src} -> {result_file} (step={step})")
        results = run_batch(dataset, out_path=str(result_file), step=step)

        # 计算准确率（match 字段为 True/False）
        if results:
            correct = sum(1 for r in results if r.get("match"))
            processed = len(results)
            acc = correct / processed
        else:
            correct = 0
            processed = 0
            acc = 0.0

        metrics[basename] = {
            "processed": processed,
            "correct": correct,
            "accuracy": acc
        }

        total_processed += processed
        total_correct += correct

    overall_acc = (total_correct / total_processed) if total_processed > 0 else 0.0
    metrics_summary = {
        "per_file": metrics,
        "total_processed": total_processed,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Overall accuracy: {overall_acc:.4f} ({total_correct}/{total_processed})")
    # 1. 兜底  2. relation 3. 
    