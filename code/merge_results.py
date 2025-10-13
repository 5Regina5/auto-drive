import json
import glob
import os
from collections import defaultdict

def load_original_questions(questions_file="f:/auto-drive/data/prompt.json"):
    """加载原始问题文件，获取每个uid对应的template_type"""
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 构建uid到template_type的映射
        uid_to_template = {}
        template_counter = {}
        
        for item in data:
            if 'uid' in item and 'template_type' in item:
                uid_to_template[item['uid']] = item['template_type']
                template_type = item['template_type']
                template_counter[template_type] = template_counter.get(template_type, 0) + 1
        
        print(f"加载原始问题文件: {len(uid_to_template)} 条记录")
        print("Template Type分布:")
        for template_type, count in sorted(template_counter.items()):
            print(f"  {template_type}: {count}")
        
        return uid_to_template
    except Exception as e:
        print(f"加载原始问题文件失败: {e}")
        return {}

def calculate_accuracy(result_item):
    """计算单个问题的准确率 - 直接检查match字段"""
    return result_item.get('match_ground_truth', False)

def save_errors_by_category(all_results):
    """按类别保存错误结果"""
    # 按template_type分组错误结果
    errors_by_type = defaultdict(list)
    
    for result in all_results:
        template_type = result.get('template_type', 'unknown')
        is_correct = result.get('is_correct', False)
        
        # 只保存错误的结果
        if not is_correct:
            errors_by_type[template_type].append(result)
    
    # 为每个类别保存错误文件
    print(f"\n保存分类别错误结果:")
    for template_type, error_results in errors_by_type.items():
        if error_results:  # 只有存在错误时才保存
            error_file = f"f:/auto-drive/data_nothink_changetail/errors_{template_type}.json"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_results, f, ensure_ascii=False, indent=2)
                print(f"  {template_type}: {len(error_results)} 条错误 -> {error_file}")
            except Exception as e:
                print(f"  {template_type}: 保存失败 - {e}")
        else:
            print(f"  {template_type}: 0 条错误")
    
    # 保存所有错误的汇总文件
    all_errors = []
    for error_results in errors_by_type.values():
        all_errors.extend(error_results)
    
    if all_errors:
        all_errors_file = "f:/auto-drive/data_nothink_changetail/all_errors.json"
        try:
            with open(all_errors_file, 'w', encoding='utf-8') as f:
                json.dump(all_errors, f, ensure_ascii=False, indent=2)
            print(f"  所有错误汇总: {len(all_errors)} 条 -> {all_errors_file}")
        except Exception as e:
            print(f"  所有错误汇总: 保存失败 - {e}")

def merge_results():
    """合并所有分片处理的结果并计算准确率"""
    
    # 加载原始问题文件
    uid_to_template = load_original_questions()
    
    # 查找所有分片文件
    pattern = "f:/auto-drive/data_nothink_changetail/llm_responses_part*.json"
    part_files = glob.glob(pattern)
    
    if not part_files:
        print("没有找到分片文件")
        return
    
    print(f"找到 {len(part_files)} 个分片文件:")
    for f in sorted(part_files):
        print(f"  {f}")
    
    all_results = []
    
    # 读取所有分片文件
    for part_file in sorted(part_files):
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.extend(data)
                print(f"加载 {part_file}: {len(data)} 条记录")
        except Exception as e:
            print(f"加载 {part_file} 失败: {e}")
    
    # 按index排序
    all_results.sort(key=lambda x: x.get('index', 0))
    
    # 为每个结果添加template_type信息并计算准确率
    template_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'success': 0})
    
    # 调试信息：检查映射情况
    unknown_count = 0
    sample_indices = []
    
    for result in all_results:
        # 获取index对应的template_type（index应该对应prompt.json中的uid）
        index = result.get('index', -1)
        template_type = uid_to_template.get(index, 'unknown')
        result['template_type'] = template_type
        
        if template_type == 'unknown':
            unknown_count += 1
        
        # 收集前10个样本用于调试
        if len(sample_indices) < 10:
            sample_indices.append({
                'index': index,
                'template_type': template_type,
                'original_token': result.get('original_token', '')
            })
    
    print(f"映射调试信息:")
    print(f"  未知template_type的数量: {unknown_count}")
    print(f"  前10个样本映射:")
    for sample in sample_indices:
        print(f"    index={sample['index']}, template_type={sample['template_type']}, token={sample['original_token'][:20]}...")
    
    # 重新遍历所有结果进行统计
    for result in all_results:
        template_type = result.get('template_type', 'unknown')
        
        # 计算准确率 - 直接使用match字段
        is_correct = calculate_accuracy(result)
        result['is_correct'] = is_correct
        
        # 统计信息
        template_stats[template_type]['total'] += 1
        if result.get('success', False):
            template_stats[template_type]['success'] += 1
        if is_correct:
            template_stats[template_type]['correct'] += 1
    
    # 保存合并结果
    output_file = "f:/auto-drive/data_nothink_changetail/llm_responses_merged.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n合并完成!")
        print(f"总记录数: {len(all_results)}")
        print(f"合并文件: {output_file}")
        
        # 按类别保存错误结果
        save_errors_by_category(all_results)
        
        # 统计成功失败情况
        success_count = sum(1 for r in all_results if r.get('success', False))
        correct_count = sum(1 for r in all_results if r.get('is_correct', False))
        
        print(f"\n整体统计:")
        print(f"成功处理: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
        print(f"回答正确: {correct_count}/{success_count} ({correct_count/success_count*100:.1f}%)" if success_count > 0 else "回答正确: 0/0 (0.0%)")
        print(f"总体准确率: {correct_count}/{len(all_results)} ({correct_count/len(all_results)*100:.1f}%)")
        
        # 按模板类型统计准确率
        print(f"\n各类型准确率:")
        print("-" * 60)
        print(f"{'Template Type':<20} {'Total':<8} {'Success':<8} {'Correct':<8} {'Accuracy':<10}")
        print("-" * 60)
        
        for template_type in sorted(template_stats.keys()):
            stats = template_stats[template_type]
            total = stats['total']
            success = stats['success']
            correct = stats['correct']
            accuracy = (correct / success * 100) if success > 0 else 0.0
            
            print(f"{template_type:<20} {total:<8} {success:<8} {correct:<8} {accuracy:<10.1f}%")
        
        # 保存详细统计信息
        stats_output = {
            "overall": {
                "total": len(all_results),
                "success": success_count,
                "correct": correct_count,
                "success_rate": success_count/len(all_results)*100 if len(all_results) > 0 else 0,
                "accuracy": correct_count/len(all_results)*100 if len(all_results) > 0 else 0,
                "accuracy_among_success": correct_count/success_count*100 if success_count > 0 else 0
            },
            "by_template": {}
        }
        
        for template_type, stats in template_stats.items():
            stats_output["by_template"][template_type] = {
                "total": stats['total'],
                "success": stats['success'],
                "correct": stats['correct'],
                "success_rate": stats['success']/stats['total']*100 if stats['total'] > 0 else 0,
                "accuracy": stats['correct']/stats['total']*100 if stats['total'] > 0 else 0,
                "accuracy_among_success": stats['correct']/stats['success']*100 if stats['success'] > 0 else 0
            }
        
        stats_file = "f:/auto-drive/data_nothink_changetail/accuracy_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)
        print(f"\n详细统计信息已保存到: {stats_file}")
        
    except Exception as e:
        print(f"保存合并文件失败: {e}")

if __name__ == "__main__":
    merge_results()
