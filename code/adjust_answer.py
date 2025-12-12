import json
import glob
import os
from typing import List, Dict, Any, Tuple

# === 配置 ===
DIR = "dataset/extract_data/questions_processed_results"
IN_PATTERN = os.path.join(DIR, "results_*.json")
OUT_DIR = os.path.join(DIR, "adjusted_filtered_all_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# 允许的“非数值”答案（统一小写，空格/下划线等价，由 normalize 处理）
ALLOWED_WORDS = {
    "yes", "no",
    "barrier", "bicycle", "bus", "car", "construction vehicle", "motorcycle",
    "moving", "not standing", "parked", "pedestrian", "standing", "stopped",
    "traffic cone", "trailer", "truck", "with rider", "without rider",
}

NUMERIC_RANGE = set(str(i) for i in range(0, 11))  # "0" ~ "10"


def normalize_token(s: Any) -> str:
    """统一化答案：转小写、下划线->空格、折叠多空格、去首尾空白。"""
    s = "" if s is None else str(s)
    s = s.lower().replace("_", " ")
    s = " ".join(s.split())  # 折叠多空格
    return s


def is_allowed_answer(ans: Any) -> bool:
    """判断 model_answer 是否属于允许集合（数值0~10，或 ALLOWED_WORDS），空格与下划线等价。"""
    t = normalize_token(ans)
    if t in NUMERIC_RANGE:
        return True
    if t in ALLOWED_WORDS:
        return True
    return False


def is_counting_file(filename: str) -> bool:
    return "counting" in os.path.basename(filename).lower()


def should_skip_question(q: str) -> bool:
    """对所有 results_*.json 生效的过滤：with rider/_rider + (traffic cone|barrier) + status"""
    ql = (q or "").lower()
    cond_with_rider = ("with rider" in ql) or ("with_rider" in ql)
    cond_obj = ("traffic cone" in ql) or ("barrier" in ql)
    cond_status = "status" in ql
    return cond_with_rider or (cond_obj and cond_status)


def str_equal(a: Any, b: Any) -> bool:
    return normalize_token(a) == normalize_token(b)


def maybe_decrement_for_other(entry: Dict[str, Any]) -> None:
    """仅在 counting 文件中：question 含 other 且 model_answer != '0' 时，将答案减 1（可转 int 时）。"""
    q = (entry.get("question") or "").lower()
    model = entry.get("model_answer")
    if "other" in q and model is not None and normalize_token(model) != "0":
        try:
            entry["model_answer"] = str(int(str(model).strip()) - 1)
        except ValueError:
            pass  # 非数字，忽略


def process_file(path: str) -> Tuple[int, int, int, float, int, list]:
    """
    返回：
      (原始条目数, 过滤后条目数, 命中数, 准确率, 非法答案计数, 非法答案示例列表)
    """
    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    original_total = len(data)

    # 全局过滤
    #kept: List[Dict[str, Any]] = [e for e in data if not should_skip_question(e.get("question", ""))]
    kept=data
    kept_total = len(kept)

    # counting: 对含 other 的样例做 -1
    if is_counting_file(path):
        for e in kept:
            maybe_decrement_for_other(e)

    # 统计非法答案
    invalid_count = 0
    invalid_examples = []  # 收集少量示例
    for e in kept:
        ma = e.get("model_answer")
        if not is_allowed_answer(ma):
            invalid_count += 1
            if len(invalid_examples) < 10:  # 只展示前 10 个示例，避免太长
                invalid_examples.append({
                    "uid": e.get("uid"),
                    "model_answer": ma,
                    "normalized": normalize_token(ma),
                    "question": e.get("question", "")[:120]
                })

    # 仅“other”题重算 match，其他保留旧 match（若无，则兜底算一次）
    matched = 0
    for e in kept:
        ql = (e.get("question") or "").lower()
        if "other" in ql:
            e["match"] = str_equal(e.get("model_answer"), e.get("gold_answer"))
        else:
            if "match" not in e:
                e["match"] = str_equal(e.get("model_answer"), e.get("gold_answer"))
        if bool(e.get("match")):
            matched += 1

    acc = (matched / kept_total) if kept_total else 0.0

    # 写结果
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(OUT_DIR, f"{name}.v3{ext}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"{base:35s} | orig={original_total:5d} kept={kept_total:5d} "
          f"matched={matched:5d} acc={acc:.4f}  invalid_ans={invalid_count:5d} -> {out_path}")

    # 额外输出非法答案示例文件
    ex_path = os.path.join(OUT_DIR, f"{name}.invalid_examples.json")
    if invalid_examples:
        with open(ex_path, "w", encoding="utf-8") as f:
            json.dump(invalid_examples, f, ensure_ascii=False, indent=2)

    return original_total, kept_total, matched, acc, invalid_count, invalid_examples


def main():
    files = sorted(glob.glob(IN_PATTERN))
    if not files:
        print(f"No files matched: {IN_PATTERN}")
        return

    summary = []
    total_kept = 0
    total_matched = 0
    total_invalid = 0

    total_kept_h0 = total_matched_h0 = 0
    total_kept_h1 = total_matched_h1 = 0

    for fp in files:
        orig, kept, matched, acc, inv, _ = process_file(fp)
        summary.append((os.path.basename(fp), orig, kept, matched, acc, inv))

        total_kept += kept
        total_matched += matched
        total_invalid += inv

        # h0/h1 汇总
        if "_h0" in os.path.basename(fp):
            total_kept_h0 += kept
            total_matched_h0 += matched
        if "_h1" in os.path.basename(fp):
            total_kept_h1 += kept
            total_matched_h1 += matched

    # 汇总打印
    print("\n=== Summary ===")
    for fname, orig, kept, matched, acc, inv in summary:
        print(f"{fname:35s} | kept={kept:6d} matched={matched:6d} acc={acc:.4f}  invalid_ans={inv:6d}")

    overall_acc = (total_matched / total_kept) if total_kept else 0.0
    acc_h0 = (total_matched_h0 / total_kept_h0) if total_kept_h0 else 0.0
    acc_h1 = (total_matched_h1 / total_kept_h1) if total_kept_h1 else 0.0

    print("\n=== Global Metrics ===")
    print(f"Overall: kept={total_kept} matched={total_matched} acc={overall_acc:.4f}  invalid_ans_total={total_invalid}")
    print(f"h0:      kept={total_kept_h0} matched={total_matched_h0} acc={acc_h0:.4f}")
    print(f"h1:      kept={total_kept_h1} matched={total_matched_h1} acc={acc_h1:.4f}")

    # 也把全局指标写出
    metrics_path = os.path.join(OUT_DIR, "metrics_v3.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": {
                "kept": total_kept,
                "matched": total_matched,
                "accuracy": overall_acc,
                "invalid_answers": total_invalid,
            },
            "h0": {
                "kept": total_kept_h0,
                "matched": total_matched_h0,
                "accuracy": acc_h0,
            },
            "h1": {
                "kept": total_kept_h1,
                "matched": total_matched_h1,
                "accuracy": acc_h1,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved global metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
