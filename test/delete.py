import sys

def remove_duplicates(file_path):
    """读取文件，删除重复行（基于文件名），并写回文件"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 使用字典去重（保留第一个出现的行）
        unique_lines = {}
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            filename = line.split(',')[0]
            if filename not in unique_lines:
                unique_lines[filename] = line
        
        # 将去重后的行写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_lines.values()) + '\n')
        
        print(f"处理完成！已从 {file_path} 中删除重复行。")
        print(f"原始行数: {len(lines)}, 去重后行数: {len(unique_lines)}")
        print(f"共删除 {len(lines) - len(unique_lines)} 个重复项。")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误：{e}")
        sys.exit(1)

if __name__ == "__main__":

    
    file_path = '/big-data/dataset-IJCAI/liwei/test/submission/prediction.txt'
    remove_duplicates(file_path)