test_h0.json:h0的测试集

test_h1.json:h1的测试集

data:qwen3-32b-thinking的结果，以及extracted_annotations

data_nothink_h0_singlelabel或data_nothink_h1_singlelabel:新鲜出炉的测试结果

code/build_llm_prompts_v3.py: 应该是最新的构建prompt的代码
code/extract_data_withego.py: 提取了所有annatation以及ego的代码
code/adjust_answer.py: 在得到结果后，对问题中的含有"other"的结果处理
code/agent.py: 完整的运行代码（有待整理）

最好的那个data:不知道是哪个pvc里的前缀为question_后缀为_parked_new的文件夹(来不及找了啊啊啊)

vllm命令行运行命令：

```bash
nohup vllm serve Qwen/Qwen3-8B --download-dir models/qwen3-8b  --enable-auto-tool-choice --tool-call-parser hermes  --reasoning-parser deepseek_r1 --tensor_parallel_size 4  --port 8000 >/dev/null 2>&1 &
```
