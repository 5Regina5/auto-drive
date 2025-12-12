# -*- coding: utf-8 -*-
import math
import os
import json, csv, re, random, argparse
from pathlib import Path
from collections import defaultdict, Counter
ALLOWED_ANSWERS = [
    "0","1","2","3","4","5","6","7","8","9","10",
    "no","yes",
    "barrier","bicycle","bus","car","construction vehicle","motorcycle",
    "moving","not standing","parked","pedestrian","standing","stopped",
    "traffic cone","trailer","truck","with rider","without rider","traffic_cone","construction_vehicle",
]
# ---------------- Object & Status vocab (aligned with your tables) ----------------
OBJECT_VOCAB = [
    # humans & animals
    "pedestrian","stroller","wheelchair","personal_mobility","animal",
    # vehicles
    "car","truck","bus","trailer","motorcycle","bicycle","construction_vehicle","emergency_vehicle",
    # movable/static objects
    "traffic_cone","barrier","debris","pushable_pullable","bicycle_rack",
]

STATUS_VOCAB = [
    "moving","stopped","parked",
    "with_rider","without_rider",
    "sitting_lying_down","standing"
]

ALIASES = {
    "exist": "existence",
    "count": "counting",
    "compare": "comparison",
    "query_object": "query_object",
    "query status": "query_status",
    "query_status": "query_status",
    "object": "query_object",
    "status": "query_status"
}
def normalized_type(t: str) -> str:
    t = (t or "").strip().lower().replace("-", "_")
    return ALIASES.get(t, t)
def coarse_class(category_name: str) -> str:
    if not category_name:
        return ""
    p = category_name.lower().split(".")
    s = set(p)
    if "human" in s and "pedestrian" in s:
        if "stroller" in s: return "stroller"
        if "wheelchair" in s: return "wheelchair"
        if "personal_mobility" in s: return "personal_mobility"
        return "pedestrian"
    if "animal" in s: return "animal"
    if "vehicle" in s:
        if "car" in s: return "car"
        if "truck" in s: return "truck"
        if "bus" in s: return "bus"
        if "trailer" in s: return "trailer"
        if "motorcycle" in s: return "motorcycle"
        if "bicycle" in s: return "bicycle"
        if "construction" in s: return "construction_vehicle"
        if "emergency" in s: return "emergency_vehicle"
    if "movable_object" in s:
        if "trafficcone" in s or "traffic_cone" in s: return "traffic_cone"
        if "barrier" in s: return "barrier"
        if "debris" in s: return "debris"
        if "pushable_pullable" in s: return "pushable_pullable"
    if "static_object" in s and "bicycle_rack" in s:
        return "bicycle_rack"
    return p[-1]

def motion_state(attributes):
    if not attributes:
        return "parked"
    atts = set(a.lower() for a in attributes)
    if "vehicle.moving" in atts: return "moving"
    if "vehicle.stopped" in atts: return "stopped"
    if "vehicle.parked" in atts: return "parked"
    if "cycle.with_rider" in atts: return "with_rider"
    if "cycle.without_rider" in atts: return "without_rider"
    if "pedestrian.moving" in atts: return "moving"
    if "pedestrian.sitting_lying_down" in atts: return "sitting_lying_down"
    if "pedestrian.standing" in atts: return "standing"
    return "other"

def quat_to_yaw(q):
    """
    将四元数 [qw, qx, qy, qz] 转成绕 z 轴的偏航角 yaw（弧度）
    约定：数据集中 rotation 为 (qw, qx, qy, qz)
    公式：yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    """
    if not q or len(q) != 4:
        return 0.0
    w, x, y, z = q
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return float(yaw)

def one_object_row(ann: dict):
    """
    只输出你需要的三个字段：
      - type: 粗粒度类别
      - position: [x, y, z, width, length, height, yaw]
      - attributes: 归一后的状态
    其余 annotation 字段（如 num_lidar_pts、visibility 等）不再纳入 prompt
    """
    cls = coarse_class(ann.get("category_name", ""))
    if cls not in ALLOWED_ANSWERS:
        return None
    t = ann.get("translation", [0.0, 0.0, 0.0])
    s = ann.get("size", [0.0, 0.0, 0.0])  # [width, length, height]
    r = ann.get("rotation", [1.0, 0.0, 0.0, 0.0])  # [qw,qx,qy,qz]
    yaw = quat_to_yaw(r)

    # position: [x, y, z, width, length, height, yaw]
    pos = [
        float(t[0]) if len(t) > 0 else 0.0,
        float(t[1]) if len(t) > 1 else 0.0,
        float(t[2]) if len(t) > 2 else 0.0,
        float(s[0]) if len(s) > 0 else 0.0,
        float(s[1]) if len(s) > 1 else 0.0,
        float(s[2]) if len(s) > 2 else 0.0,
        yaw,
    ]
    status = motion_state(ann.get("attribute_names", ["vehicle.parked"]))

    return {
        "type": cls,
        "position": pos,
        "attributes": status,
    }

# ---------------- Prompt 构造 ----------------
def build_prompt(question: str, ttype: str, objects_json: str) -> str:
    """
    prompt 直接嵌 objects 的 JSON 数组；并按问题类型添加输出约束
    """
    header = (
        "You are given a JSON array named OBJECTS that lists all annotated 3D objects in a scene.\n"
        "Each element has this schema:\n"
        '  {"type": <coarse_class>, "position": [x, y, z, width, length, height, yaw], "attributes": <status>}\n'
        "Objects:\n"
        f"{objects_json}\n\n"
        f"Question: {question}\n"
    )
    if ttype == "existence":
        tail = "Answer strictly with 'yes' or 'no'."
    elif ttype == "counting":
        tail = "Answer with a single non-negative integer only."
    elif ttype == "query_object":
        tail = ("Answer with ONE coarse class only from: " +
                ", ".join(OBJECT_VOCAB) + ".")
    elif ttype == "query_status":
        tail = ("Answer with ONE status only from: " +
                ", ".join(STATUS_VOCAB) + ".")
    elif ttype == "comparison":
        tail = "This is a comparison question; answer strictly with 'yes' or 'no'."
    else:
        tail = "Answer with a single concise token."
    # tail=("  Output EXACTLY ONE string chosen from the set below (case-sensitive; spaces allowed; no quotes, no punctuation):\n"
    #     "  " + ", ".join(ALLOWED_ANSWERS) + "\n")
    return header + tail

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_questions_dir", type=Path, default=Path("raw_questions"))
    ap.add_argument("--annotations_json", type=Path, default=Path("data\extracted_annotations_canbusego.json"))
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory to save per-file datasets (will contain 10 files)")
    ap.add_argument("--per_file", type=int, default=None, help="Optional: limit number of questions taken from each raw file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 读取 annotations（sample_token -> list[ann]）
    ann_map = {}
    if not args.annotations_json.exists():
        print(f"Warning: annotations_json {args.annotations_json} not found. ann_map will be empty.")
    else:
        with open(args.annotations_json, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"Warning: failed to parse line in annotations_json: {e}")
                    continue
                if isinstance(item, dict) and len(item) == 1:
                    for k, v in item.items():
                        ann_map[k] = v
                else:
                    print(f"Warning: skipping malformed line in annotations_json: {line[:100]}...")

    # helper: build output for a single questions list and write to file
    def generate_output(questions, output_file):
        out = []
        uid = 0
        for q in questions:
            token = q.get("sample_token")
            anns = ann_map.get(token, []) if token is not None else []

            # 构造 per-annotation objects
            objs = [one_object_row(a) for a in anns  if one_object_row(a) is not None]
            # 如果 anns 为空，使用默认值以避免索引错误
            if anns and isinstance(anns, list) and len(anns) > 0:
                ego_dir = anns[0].get("ego_v_dir2d", [0.0, 1.0])
                ego_pose_translation = anns[0].get("ego_pose_translation")
            else:
                ego_dir = [0.0, 1.0]
                ego_pose_translation = None

            # 将 objects 以紧凑 JSON 字符串嵌入到 prompt
            objects_json_str = json.dumps(objs, ensure_ascii=False)
            ttype = normalized_type(q.get("template_type", "unknown"))
            prompt = build_prompt(q.get("question", "").strip(), ttype, objects_json_str)

            out.append({
                "uid": uid,
                "sample_token": token,
                "template_type": ttype,
                "question": q.get("question", "").strip(),
                "gold_answer": str(q.get("answer", "")).strip(),
                "num_hop": q.get("num_hop", 0),
                "objects": objs,
                "ego_v_dir2d": ego_dir,
                "ego_pose_translation": ego_pose_translation,
                # "prompt": prompt
            })
            uid += 1

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(out)} prompts to {output_file}")
        return len(out)

    # 遍历 raw_questions_dir 下的所有 *_h0.json 和 *_h1.json 文件，逐个处理并保存到 out_dir
    if not args.raw_questions_dir.exists():
        print(f"Error: raw_questions_dir {args.raw_questions_dir} does not exist")
        return

    files = sorted([p for p in args.raw_questions_dir.iterdir() if p.is_file() and (p.name.endswith("_h0.json") or p.name.endswith("_h1.json"))])
    if not files:
        print(f"No *_h0.json or *_h1.json files found in {args.raw_questions_dir}")
        return

    total_processed = 0
    for src in files:
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {src}: {e}")
            continue

        if args.per_file is not None:
            data = data[: args.per_file]

        out_file = args.out_dir / src.name
        count = generate_output(data, out_file)
        total_processed += count

    print(f"Total questions processed across all files: {total_processed}")

if __name__ == "__main__":
    main()