## 1)全局变量
### `CURRENT_SCENE` 与答案约束

* `CURRENT_SCENE` 是进程内的“当前绑定场景”，所有工具都只读它。
* `EGO_TYPE="ego"`, `ME_ALIASES={"me","ego","ego-car"}`：把“我/自车”统一成 `ego`。
* `ALLOWED_ANSWERS`：允许输出的 type/status/数字/yes/no 列表（后面会做 normalize）。

### 绑定样本到 `CURRENT_SCENE`

* `_get_ego_index()`：在 `objects` 里找 `type=="ego"` 的索引。
* `bind_scene_from_sample(sample)`：
  * 把 `uid/objects/ego_v_dir2d/ego_pose_translation` 拷贝到 `CURRENT_SCENE`。
  * 对 `ego_v_dir2d` / `ego_pose_translation` 的 `None` 做了兜底（默认方向 `[0,1]`，默认位置 `[0,0,0]`）。
  * 如果 objects 里没有 ego，就注入一个“合成 ego”，放在 index 0（后面工具都跳过 ego 作为 target）。

## 2) 几何与关系判断

### `calculate_signed_angle_degrees`

* 用 `atan2(cross, dot)` 计算有符号角度 ∈ [-180, 180]。
* 这里的 `B1` 是候选物体位置，`B2` 是参考物体位置，`V_ego` 是 ego 方向（用前两维）。

### `classify_relation`

把角度分桶成 `front/front_left/front_right/back_left/back_right/back`，并且 **区间是右闭**。

## 3) Selector 选择逻辑

这一段决定“给定 selector/type_filter/status_filter 时，哪些 objects 被命中”。

* `_normalize_ego_aliases_in_selector`：把 selector 的 `type` 如果是 me/ego-car 等别名，统一成 `ego`。
* `_select_by_type_status(selector)`：

  * 先用 `normalize_type_and_status` 规范化 type/status。
  * 遍历 `CURRENT_SCENE["objects"]`，匹配 type/status。
  * 关键：当 `t_norm != "ego"` 时，会跳过 ego 物体（不把自车当成普通候选）。
  * 还有一条和 parked 相关的过滤：`if o_type not in ALLOWED_ANSWERS_NORM and s_norm == "parked": continue`。 
* `_select_by_index_or_filter`：如果 selector 里给了 `index`，就直接用 index（并做边界检查）；否则走 type/status 过滤。

## 3) 工具实现

这些工具是 LLM 唯一能访问场景事实的途径，全部只读 `CURRENT_SCENE`。

### `intersect_ctx`

* 输入两组 indices：`base` 与 `other`，输出交集（保留 base 顺序，且 base 去重）。
* 失败安全：任一为空直接返回空交集，避免“误用导致假阳性”。

###  `find_by_relation_ctx`

* 先用 `ref_selector` 找 reference（如果多个命中，只取第一个 ref）。
* 对每个候选物体：
  * 跳过 ref 本身、跳过 ego。
  * 计算相对角度 → 分桶 → 筛 relation。
  * 可选再按 `type_filter/status_filter` 二次过滤。

###  `count_ctx`

两种模式：

* A) `indices` 模式：直接数你给的 indices（做类型转换和越界过滤）。
* B) filter 模式：按 `type_filter/status_filter` 遍历场景计数，并对 `parked` 有额外过滤（过滤 barrier/trafficcone 以及不在 allowed 的类型）。

###  `exists_ctx`

* 本质是 `count_ctx` 的封装：count>0 → yes，否则 no，同时返回 count 和 indices。

### `get_type_ctx`

两种模式：

* A) `indices` 模式：对给定 indices 按顺序返回 `{index,type}`，并支持可选过滤（type/status），且会过滤掉不在 `ALLOWED_ANSWERS_NORM` 里的 type。
* B) `selector` 模式：用 selector 找 indices，再返回 type 列表。

###  `get_status_ctx`

与 `get_type_ctx` 基本同构：

* `indices` 模式支持 type/status 过滤；
* `selector` 模式按 selector 找到 indices 后返回 status。

###  `compare_status_ctx`

* 优先用 `a_indices/b_indices`（如果提供且非空），否则用 `a_selector/b_selector` 解析。
* 失败安全：任一侧为空直接判 `same=False`，不做“扩展到全场景”的危险推断。
* status 归一化后比较，**空/any 不参与匹配**；输出所有匹配 pairs（笛卡尔式配对）。

## 4) Tools Schema 与运行时派发

### `TOOLS`

* 这是给 LLM 的 tool schema（function calling 定义）。
* 每个 tool 的参数约束都在这里（比如 relation enum、indices 类型、minItems 等）。

### `DISPATCH`

* Python 侧把 tool name → 实现函数映射起来，供执行 tool_calls 时调用。


## 5) LLM 客户端与系统提示词

### `OpenAI` client + `get_response`

* `api_key` 在代码里写死，并且 `base_url="http://localhost:8000/v1"`：vllm起了一个兼容 OpenAI 的服务后才能用，如何起服务写在外面的readme里了，一定要注意端口要对齐，比如我习惯起服务时设置端口为8000。
* `model="Qwen/Qwen3-8B"`，`tools=TOOLS`，`temperature=0.0`：偏确定性评测配置。

### `SYSTEM_PROMPT`

* 系统提示词

## 6) 归一化与答案解析

### normalize 系列

* `normalize_label`：大小写无关、空格/下划线等价、`sitting_lying_down → not_standing`。
* `ALLOWED_ANSWERS_NORM`：对允许答案做 normalize 后的集合。
* `normalize_type_and_status`：这是 selector/filter 的关键兜底逻辑：
  * "any/thing/object/vehicle/空" → 不作为 filter；
  * 如果把 status 误填进 type，会自动挪到 status；
  * status 不在认可集合就丢弃；
  * ego 别名归一到 `ego`；
    这些逻辑直接影响 `_select_by_type_status`、`find_by_relation_ctx` 等工具的命中行为。

### `answers_match`

* 用 normalize 后字符串是否相等判断对错。

###：`parse_final_answer`

* 这是“最终答案兜底解析器”，处理两类情况：

  1. LLM 最终输出居然是 JSON（例如 `{ "name": "count_ctx", "arguments": {...}}`），它会直接调用本地 `DISPATCH` 再提取答案；
  2. LLM 直接输出纯文本（例如 `no / 3 / car`），就原样返回。


## 7) run_one_sample

### 准备 messages

* `bind_scene_from_sample(sample)` 绑定场景。
* messages：system prompt + user question（还拼了一个 `\n /no_think`，像是给某些模型的控制 token）。

### 处理 tool_calls 的多轮循环

循环逻辑是标准 function calling：

1. LLM 产出 tool_calls
2. Python 执行 tool_calls，追加 tool message
3. 再让 LLM 接着生成
   直到没有 tool_calls 或到达 safety stop（最多 5 轮）

其中有一段特殊“修补 tool_calls”的逻辑：

* 如果 LLM 调用 `get_status_ctx` 但 selector/type_token 是 “thing/any/空”，同时又给了 status，这段代码会把调用强行改写成 `get_type_ctx(selector={status: ...})`。

### 产出最终结构化结果

* `final_answer = parse_final_answer(assistant.content)`
* `match = answers_match(final_answer, gold)`
* 返回一个 dict：含 steps（每轮 tool_calls+tool_results）和 turns。

## 8) 批量评测与主入口

### `run_batch`

* 支持 `step` 抽样（`dataset[::step]`）。
* 单样本异常会被捕获并写入结果（`match=False`，带 traceback）。

### `__main__`

* 默认读取 `dataset/extract_data/question_processed_p09r03` 目录里所有 `*_h0.json` 和 `*_h1.json`。
* 对每个文件跑 `run_batch`，分别写 `results_*.json`，并汇总写 `metrics.json`。

每个 h0和h1的JSON 文件应是一个 list，每个元素（sample）至少包含：

* `uid`
* `objects`：list[dict]，每个 dict 有 `type`, `position`, `attributes`
* `ego_v_dir2d`
* `ego_pose_translation`
* `question`
* `gold_answer`


运行后输出：

* 每个输入文件对应一个 `results_*.json`
* 一个汇总 `metrics.json`
* 终端打印 overall accuracy 

