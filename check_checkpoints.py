#!/usr/bin/env python3
"""
检查 checkpoint 保存位置的脚本
"""
import os
import sys
from pathlib import Path

def check_checkpoints():
    """检查 checkpoint 目录"""
    base_dir = "checkpoints"
    
    print("=" * 80)
    print("🔍 Checkpoint 检查工具")
    print("=" * 80)
    print(f"\n检查目录: {os.path.abspath(base_dir)}\n")
    
    if not os.path.exists(base_dir):
        print(f"❌ Checkpoint 目录不存在: {base_dir}")
        print("\n可能的原因：")
        print("  1. 实验还没有运行")
        print("  2. checkpoint 保存失败（检查日志中的错误信息）")
        print("  3. checkpoint 保存在其他位置")
        return
    
    # 列出所有实验
    experiments = []
    for item in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, item)
        if os.path.isdir(exp_path):
            experiments.append(item)
    
    if not experiments:
        print(f"⚠️  Checkpoint 目录存在，但没有实验数据")
        print(f"   目录: {os.path.abspath(base_dir)}")
        return
    
    print(f"✅ 找到 {len(experiments)} 个实验:\n")
    
    total_checkpoints = 0
    for exp_name in experiments:
        exp_path = os.path.join(base_dir, exp_name)
        print(f"📁 实验: {exp_name}")
        print(f"   路径: {os.path.abspath(exp_path)}")
        
        # 列出所有节点
        nodes = []
        for item in os.listdir(exp_path):
            node_path = os.path.join(exp_path, item)
            if os.path.isdir(node_path):
                nodes.append(item)
        
        if not nodes:
            print("   ⚠️  没有节点目录")
            continue
        
        print(f"   📍 节点数: {len(nodes)}")
        
        for node_addr in nodes:
            node_path = os.path.join(exp_path, node_addr)
            print(f"\n   🔹 节点: {node_addr}")
            print(f"      路径: {os.path.abspath(node_path)}")
            
            # 列出所有 checkpoint 文件
            checkpoint_files = []
            if os.path.exists(node_path):
                for filename in os.listdir(node_path):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(node_path, filename)
                        size = os.path.getsize(filepath)
                        # 尝试解析 checkpoint 类型
                        checkpoint_type = "unknown"
                        if "_local.pkl" in filename:
                            checkpoint_type = "local"
                        elif "_aggregated.pkl" in filename:
                            checkpoint_type = "aggregated"
                        elif "_round_finished.pkl" in filename:
                            checkpoint_type = "round_finished"
                        checkpoint_files.append((filename, size, checkpoint_type))
            
            if checkpoint_files:
                print(f"      ✅ 找到 {len(checkpoint_files)} 个 checkpoint 文件:")
                for filename, size, ckpt_type in sorted(checkpoint_files):
                    size_mb = size / (1024 * 1024)
                    print(f"         • {filename} ({size_mb:.2f} MB) [{ckpt_type}]")
                total_checkpoints += len(checkpoint_files)
            else:
                print(f"      ❌ 没有 checkpoint 文件")
        
        print()
    
    print("=" * 80)
    print(f"📊 总计: {len(experiments)} 个实验, {total_checkpoints} 个 checkpoint 文件")
    print("=" * 80)
    
    # 检查日志中的 checkpoint 信息
    print("\n💡 提示:")
    print("  如果看不到 checkpoint，请检查:")
    print("  1. 运行日志中是否有 'Checkpoint saved' 消息")
    print("  2. 是否有 'Failed to save checkpoint' 警告")
    print("  3. 实验是否成功完成（至少完成 1 轮训练）")
    print("\n  查看日志:")
    print("     grep -r 'Checkpoint saved' logs/")
    print("     grep -r 'Failed to save checkpoint' logs/")

if __name__ == "__main__":
    check_checkpoints()

