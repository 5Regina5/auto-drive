import json
import pandas as pd
import os
import glob

def convert_error_files_to_xlsx():
    """
    专门用于转换错误分析JSON文件到Excel格式
    """
    # 定义要转换的文件
    error_files = [
        "f:/auto-drive/data/errors_counting.json",
        "f:/auto-drive/data/errors_comparison.json", 
        "f:/auto-drive/data/errors_existence.json",
        "f:/auto-drive/data/errors_query_object.json",
        "f:/auto-drive/data/errors_query_status.json",
        "f:/auto-drive/data/all_errors.json",
        "f:/auto-drive/data/llm_responses_merged.json"
    ]
    
    print("开始转换错误分析文件...")
    
    for json_file in error_files:
        if os.path.exists(json_file):
            try:
                # 读取JSON文件
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 转换为DataFrame
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    
                    # 为特定列设置更好的显示格式
                    if 'llm_reasoning' in df.columns:
                        # 限制推理内容长度，避免Excel显示问题
                        df['llm_reasoning_preview'] = df['llm_reasoning'].astype(str).str[:200] + '...'
                    
                    if 'llm_response' in df.columns:
                        # 限制回复内容长度
                        df['llm_response_preview'] = df['llm_response'].astype(str).str[:100] + '...'
                    
                    # 生成Excel文件路径
                    xlsx_file = json_file.replace('.json', '.xlsx')
                    
                    # 保存为Excel
                    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                        
                        # 调整列宽
                        worksheet = writer.sheets['Data']
                        for column in worksheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            
                            # 设置合适的列宽，最大50个字符
                            adjusted_width = min(max_length + 2, 50)
                            worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    print(f"✓ 转换完成: {os.path.basename(json_file)} -> {os.path.basename(xlsx_file)}")
                    print(f"  数据行数: {len(df)}, 列数: {len(df.columns)}")
                    
                else:
                    print(f"✗ 跳过: {os.path.basename(json_file)} (数据格式不支持)")
                    
            except Exception as e:
                print(f"✗ 转换失败: {os.path.basename(json_file)} - {e}")
        else:
            print(f"- 文件不存在: {os.path.basename(json_file)}")
    
    print("\n转换完成!")

def convert_single_json_to_xlsx(json_file_path):
    """
    转换单个JSON文件为Excel
    
    Args:
        json_file_path: JSON文件路径
    """
    if not os.path.exists(json_file_path):
        print(f"文件不存在: {json_file_path}")
        return
    
    try:
        print(f"正在转换: {json_file_path}")
        
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # 如果是字典，尝试展平
            df = pd.json_normalize(data)
        else:
            print("不支持的数据格式")
            return
        
        # 生成Excel文件路径
        xlsx_file = json_file_path.replace('.json', '.xlsx')
        
        # 保存为Excel
        with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
            
            # 调整列宽
            worksheet = writer.sheets['Data']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                # 设置合适的列宽
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✓ 转换完成: {xlsx_file}")
        print(f"  数据行数: {len(df)}")
        print(f"  数据列数: {len(df.columns)}")
        print(f"  列名: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式：转换指定文件
        json_file = sys.argv[1]
        convert_single_json_to_xlsx(json_file)
    else:
        # 交互模式
        print("JSON转Excel工具")
        print("=" * 30)
        print("1. 转换所有错误分析文件")
        print("2. 转换指定文件")
        
        choice = input("请选择模式 (1/2): ").strip()
        
        if choice == '1':
            convert_error_files_to_xlsx()
        elif choice == '2':
            json_file = input("请输入JSON文件路径: ").strip()
            # 处理拖拽文件的引号
            json_file = json_file.strip('"\'')
            convert_single_json_to_xlsx(json_file)
        else:
            print("无效选择!")
