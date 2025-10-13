import json
from collections import Counter

def count_template_types():
    """统计prompt.json中各个template_type的数量"""
    try:
        with open("f:/auto-drive/data/prompt.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计各个template_type的数量
        template_counter = Counter()
        uid_to_template = {}
        
        for item in data:
            if 'template_type' in item:
                template_type = item['template_type']
                template_counter[template_type] += 1
            
            if 'uid' in item and 'template_type' in item:
                uid_to_template[item['uid']] = item['template_type']
        
        print("Template Type统计:")
        print("-" * 40)
        for template_type, count in sorted(template_counter.items()):
            print(f"{template_type:<15}: {count}")
        
        print(f"\n总数: {sum(template_counter.values())}")
        print(f"UID映射数量: {len(uid_to_template)}")
        
        # 检查前几个记录的结构
        print(f"\n前3个记录的结构:")
        for i, item in enumerate(data[:3]):
            print(f"记录{i}: uid={item.get('uid')}, sample_token={item.get('sample_token')}, template_type={item.get('template_type')}")
        
    except Exception as e:
        print(f"统计失败: {e}")

if __name__ == "__main__":
    count_template_types()
