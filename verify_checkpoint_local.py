#!/usr/bin/env python3
"""
验证 checkpoint 是否保存到本地的脚本
运行 node1 和 node2 后，用这个脚本验证 checkpoint 是否真的保存到了本地文件系统
"""
import os
import sys
import pickle
from pathlib import Path

def verify_checkpoint_local():
    """验证 checkpoint 是否保存在本地"""
    base_dir = "checkpoints"
    project_root = Path(__file__).parent.absolute()
    checkpoint_path = project_root / base_dir
    
    print("=" * 80)
    print("🔍 验证 Checkpoint 本地保存")
    print("=" * 80)
    print(f"\n项目根目录: {project_root}")
    print(f"Checkpoint 目录: {checkpoint_path}")
    print(f"绝对路径: {checkpoint_path.absolute()}\n")
    
    if not checkpoint_path.exists():
        print("❌ Checkpoint 目录不存在")
        print("\n💡 提示:")
        print("  1. 确保已经运行了 node1.py 和 node2.py")
        print("  2. 确保已经启动了训练（调用 node.set_start_learning()）")
        print("  3. 确保至少完成了一轮训练")
        return False
    
    print("✅ Checkpoint 目录存在\n")
    
    # 列出所有实验
    experiments = []
    for item in checkpoint_path.iterdir():
        if item.is_dir():
            experiments.append(item)
    
    if not experiments:
        print("⚠️  目录存在但没有实验数据")
        return False
    
    print(f"📁 找到 {len(experiments)} 个实验:\n")
    
    total_files = 0
    total_size = 0
    
    for exp_dir in experiments:
        print(f"🔬 实验: {exp_dir.name}")
        print(f"   路径: {exp_dir.absolute()}\n")
        
        # 列出所有节点
        nodes = []
        for item in exp_dir.iterdir():
            if item.is_dir():
                nodes.append(item)
        
        for node_dir in nodes:
            print(f"   📍 节点: {node_dir.name}")
            print(f"      路径: {node_dir.absolute()}")
            
            # 列出所有 checkpoint 文件
            checkpoint_files = list(node_dir.glob("*.pkl"))
            
            if checkpoint_files:
                print(f"      ✅ 找到 {len(checkpoint_files)} 个 checkpoint 文件:")
                for ckpt_file in sorted(checkpoint_files):
                    size = ckpt_file.stat().st_size
                    size_mb = size / (1024 * 1024)
                    total_files += 1
                    total_size += size
                    
                    # 尝试读取 checkpoint 验证内容
                    try:
                        with open(ckpt_file, 'rb') as f:
                            data = pickle.load(f)
                            metadata = data.get('metadata', {})
                            round_num = metadata.get('round', '?')
                            node_id = metadata.get('node_id', '?')
                            checkpoint_type = data.get('checkpoint_type', 'unknown')
                            
                        print(f"         • {ckpt_file.name}")
                        print(f"           大小: {size_mb:.2f} MB")
                        print(f"           轮次: {round_num}")
                        print(f"           类型: {checkpoint_type}")
                        print(f"           节点ID: {node_id}")
                        print(f"           验证: ✅ 文件可读，内容有效")
                    except Exception as e:
                        print(f"         • {ckpt_file.name}")
                        print(f"           大小: {size_mb:.2f} MB")
                        print(f"           验证: ⚠️  文件存在但无法读取: {e}")
                print()
            else:
                print(f"      ❌ 没有 checkpoint 文件\n")
    
    print("=" * 80)
    print(f"📊 总计:")
    print(f"   - 实验数: {len(experiments)}")
    print(f"   - Checkpoint 文件数: {total_files}")
    print(f"   - 总大小: {total_size / (1024 * 1024):.2f} MB")
    print("=" * 80)
    
    if total_files > 0:
        print("\n✅ 验证成功！Checkpoint 确实保存在本地文件系统中")
        print(f"\n💡 你可以用以下命令查看:")
        print(f"   ls -lh {checkpoint_path}")
        print(f"   find {checkpoint_path} -name '*.pkl'")
        return True
    else:
        print("\n⚠️  没有找到 checkpoint 文件")
        return False

if __name__ == "__main__":
    success = verify_checkpoint_local()
    sys.exit(0 if success else 1)

