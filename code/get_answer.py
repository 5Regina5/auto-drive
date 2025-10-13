import json
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
import argparse

class BaiLianLLMClient:
    """阿里百炼平台大模型客户端"""
    
    def __init__(self, api_key: str, model_name: str = "qwen-turbo"):
        """
        初始化客户端
        
        Args:
            api_key: 阿里百炼平台的API密钥
            model_name: 使用的模型名称，默认为qwen-turbo
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        生成模型回复
        
        Args:
            prompt: 输入的提示文本
            max_tokens: 最大生成token数
            
        Returns:
            包含模型回复的字典
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"enable_thinking": False},
                stream=True  # 使用流式输出
            )
            
            is_answering = False  # 是否进入回复阶段
            reasoning_content = ""  # 思考过程内容
            response_content = ""   # 回复内容
            
            for chunk in completion:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    response_content += delta.content
            return {
                "success": True,
                "reasoning": reasoning_content,  # 思考过程
                "response": response_content,    # 最终回复
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"请求异常: {str(e)}",
                "response": ""
            }

def load_prompts(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载提示数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        提示数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条提示数据")
        return data
    except Exception as e:
        print(f"加载提示文件失败: {e}")
        return []

def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    保存结果到JSON文件
    
    Args:
        results: 结果列表
        output_path: 输出文件路径
    """
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")

def process_prompts_with_llm(
    prompts_file: str,
    api_key: str,
    output_file: str,
    model_name: str = "qwen-turbo",
    start_index: int = 0,
    end_index: int = None
):
    """
    处理提示数据并获取大模型回复
    
    Args:
        prompts_file: 提示数据文件路径
        api_key: 阿里百炼API密钥
        output_file: 输出文件路径
        model_name: 模型名称
        start_index: 开始处理的索引（包含）
        end_index: 结束处理的索引（不包含），None表示处理到最后
    """
    # 加载提示数据
    prompts_data = load_prompts(prompts_file)
    if not prompts_data:
        return
    
    # 确定处理范围
    total_count = len(prompts_data)
    if end_index is None:
        end_index = total_count
    
    # 切片数据
    prompts_slice = prompts_data[start_index:end_index]
    slice_count = len(prompts_slice)
    
    print(f"总数据量: {total_count}")
    print(f"处理范围: {start_index} - {end_index-1} (共 {slice_count} 条)")
    
    # 初始化客户端
    client = BaiLianLLMClient(api_key, model_name)
    
    results = []
    
    print(f"开始处理 {slice_count} 条提示数据...")
    
    for i, prompt_item in enumerate(prompts_slice):
        global_index = start_index + i  # 全局索引
        print(f"处理进度: {i+1}/{slice_count} (全局索引: {global_index})")
        
        # 构建提示文本
        if isinstance(prompt_item, dict):
            # 如果是字典格式，提取question作为提示
            if 'prompt' in prompt_item:
                prompt_text = prompt_item['prompt']
            else:
                # 如果没有question字段，将整个对象转为JSON字符串作为提示
                prompt_text = json.dumps(prompt_item, ensure_ascii=False)
        else:
            # 如果是字符串，直接使用
            prompt_text = str(prompt_item)
        
        # 获取模型回复
        response = client.generate_response(prompt_text)
        
        # 构建结果记录
        # 处理ground_truth和llm_response，去除空格和下划线后再比较
        def normalize_answer(ans):
            return str(ans).replace(" ", "").replace("_", "").strip().lower()

        ground_truth = prompt_item.get("gold_answer", "")
        llm_response = response["response"] if response["success"] else ""

        result_item = {
            "index": global_index,  # 使用全局索引
            "original_token": prompt_item["sample_token"],
            "ground_truth": ground_truth,
            "prompt": prompt_text,
            "llm_reasoning": response.get("reasoning", "") if response["success"] else "",  # 思考过程
            "llm_response": llm_response,           # 最终回复
            "success": response["success"],
            "error": response.get("error", ""),
            "match_ground_truth": (
            normalize_answer(llm_response) == normalize_answer(ground_truth)
            ) if response["success"] else False
        }
        
        results.append(result_item)
        
        
        # 每处理完一条就立即保存结果
        save_results(results, output_file)
    
    # 保存最终结果
    save_results(results, output_file)
    
    # 统计信息
    success_count = sum(1 for r in results if r["success"])
    print(f"\n处理完成!")
    print(f"处理范围: {start_index} - {end_index-1}")
    print(f"处理数量: {slice_count}")
    print(f"成功: {success_count}")
    print(f"失败: {slice_count - success_count}")
    print(f"结果已保存到: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="调用阿里百炼大模型批量处理提示")
    parser.add_argument("--prompts_file", type=str, default="prompt_changetail.json", help="提示数据文件路径")
    parser.add_argument("--output_file", type=str, default="f:/auto-drive/data_nothink_changetail/llm_responses.json", help="输出文件路径")
    parser.add_argument("--model_name", type=str, default="qwen3-8b", help="模型名称")
    parser.add_argument("--start_index", type=int, default=0, help="开始处理的索引（包含）")
    parser.add_argument("--end_index", type=int, default=None, help="结束处理的索引（不包含）")
    parser.add_argument("--process_id", type=int, default=None, help="进程ID（1-5），自动计算处理范围")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 从环境变量获取API密钥，或者直接设置
    API_KEY = "sk-709609f9bf95438ba6408edd8273f756"
    
    if not API_KEY:
        print("请设置DASHSCOPE_API_KEY环境变量，或在代码中直接设置API_KEY")
        print("例如: set DASHSCOPE_API_KEY=your_api_key_here")
        return
    
    # 如果指定了process_id，自动计算处理范围（假设总共2500条数据，每个进程处理500条）
    if args.process_id is not None:
        if args.process_id < 1 or args.process_id > 5:
            print("process_id 必须在 1-5 之间")
            return
        
        ITEMS_PER_PROCESS = 500
        args.start_index = (args.process_id - 1) * ITEMS_PER_PROCESS
        args.end_index = args.process_id * ITEMS_PER_PROCESS
        
        # 为每个进程生成不同的输出文件名
        base_name = args.output_file.replace('.json', '')
        args.output_file = f"{base_name}_part{args.process_id}.json"
        
    
    print(f"配置信息:")
    print(f"  提示文件: {args.prompts_file}")
    print(f"  输出文件: {args.output_file}")
    print(f"  模型名称: {args.model_name}")
    if args.process_id:
        print(f"  进程ID: {args.process_id}")
    print(f"  处理范围: {args.start_index} - {args.end_index if args.end_index else '结尾'}")
    print()
    
    # 开始处理
    process_prompts_with_llm(
        prompts_file=args.prompts_file,
        api_key=API_KEY,
        output_file=args.output_file,
        model_name=args.model_name,
        start_index=args.start_index,
        end_index=args.end_index
    )

if __name__ == "__main__":
    main()
