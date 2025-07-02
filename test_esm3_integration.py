#!/usr/bin/env python3
"""
测试ESM3在SaprotHub中的集成
"""

import os
import sys
import torch
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_esm3_import():
    """测试ESM3导入"""
    print("🔍 测试ESM3导入...")
    
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        from esm3_protein_encoder import (
            read_pdb_simple, 
            preprocess_coordinates, 
            encode_sequence_only, 
            encode_sequence_and_structure,
            get_esm3_encoding,
            get_esm3_tokenizer,
            encode_pdb_to_tokens
        )
        print("✅ ESM3导入成功!")
        return True
    except ImportError as e:
        print(f"❌ ESM3导入失败: {e}")
        return False

def test_pdb_reading():
    """测试PDB文件读取"""
    print("\n🔍 测试PDB文件读取...")
    
    # 创建一个简单的测试PDB文件
    test_pdb_content = """ATOM      1  N   ALA A   1      27.462  14.105   5.468  1.00 20.00
ATOM      2  CA  ALA A   1      26.213  14.871   5.548  1.00 20.00
ATOM      3  C   ALA A   1      25.085  14.146   6.285  1.00 20.00
ATOM      4  O   ALA A   1      24.985  14.254   7.505  1.00 20.00
ATOM      5  N   GLY A   2      24.154  13.481   5.612  1.00 20.00
ATOM      6  CA  GLY A   2      23.012  12.723   6.156  1.00 20.00
ATOM      7  C   GLY A   2      22.864  11.318   5.612  1.00 20.00
ATOM      8  O   GLY A   2      22.864  11.254   4.392  1.00 20.00
END"""
    
    test_pdb_file = "test_protein.pdb"
    with open(test_pdb_file, 'w') as f:
        f.write(test_pdb_content)
    
    try:
        from esm3_protein_encoder import read_pdb_simple, preprocess_coordinates
        
        sequence, coordinates = read_pdb_simple(test_pdb_file)
        coordinates = preprocess_coordinates(coordinates)
        
        print(f"✅ PDB读取成功!")
        print(f"   序列: {sequence}")
        print(f"   坐标形状: {coordinates.shape}")
        
        # 清理测试文件
        os.remove(test_pdb_file)
        return True
        
    except Exception as e:
        print(f"❌ PDB读取失败: {e}")
        if os.path.exists(test_pdb_file):
            os.remove(test_pdb_file)
        return False

def test_esm3_encoding():
    """测试ESM3编码"""
    print("\n🔍 测试ESM3编码...")
    
    if not test_esm3_import():
        return False
    
    try:
        from esm3_protein_encoder import get_esm3_encoding, get_esm3_tokenizer
        
        # 创建一个简单的测试序列
        test_sequence = "AG"
        
        # 创建测试PDB文件
        test_pdb_content = """ATOM      1  N   ALA A   1      27.462  14.105   5.468  1.00 20.00
ATOM      2  CA  ALA A   1      26.213  14.871   5.548  1.00 20.00
ATOM      3  C   ALA A   1      25.085  14.146   6.285  1.00 20.00
ATOM      4  O   ALA A   1      24.985  14.254   7.505  1.00 20.00
ATOM      5  N   GLY A   2      24.154  13.481   5.612  1.00 20.00
ATOM      6  CA  GLY A   2      23.012  12.723   6.156  1.00 20.00
ATOM      7  C   GLY A   2      22.864  11.318   5.612  1.00 20.00
ATOM      8  O   GLY A   2      22.864  11.254   4.392  1.00 20.00
END"""
        
        test_pdb_file = "test_protein.pdb"
        with open(test_pdb_file, 'w') as f:
            f.write(test_pdb_content)
        
        # 测试编码
        encoding_results, sequence, coordinates = get_esm3_encoding(
            test_pdb_file, 
            model=None, 
            device="cpu",  # 使用CPU避免GPU问题
            use_structure=True
        )
        
        print(f"✅ ESM3编码成功!")
        print(f"   序列: {sequence}")
        print(f"   编码结果键: {list(encoding_results.keys())}")
        
        # 测试tokenizer
        tokenizer = get_esm3_tokenizer()
        tokens = tokenizer(sequence, return_tensors="pt")
        print(f"   Token形状: {tokens['input_ids'].shape}")
        
        # 清理测试文件
        os.remove(test_pdb_file)
        return True
        
    except Exception as e:
        print(f"❌ ESM3编码失败: {e}")
        if os.path.exists("test_protein.pdb"):
            os.remove("test_protein.pdb")
        return False

def test_saprot_integration():
    """测试与SaprotHub的集成"""
    print("\n🔍 测试SaprotHub集成...")
    
    try:
        # 导入saprot-esm中的函数
        from saprot_esm import (
            get_esm3_model_and_tokenizer,
            encode_pdb_with_esm3,
            batch_encode_pdbs_with_esm3,
            ESM3_AVAILABLE
        )
        
        print(f"✅ SaprotHub集成成功!")
        print(f"   ESM3可用: {ESM3_AVAILABLE}")
        
        if ESM3_AVAILABLE:
            # 测试模型和tokenizer加载
            model, tokenizer = get_esm3_model_and_tokenizer()
            if model is not None and tokenizer is not None:
                print("   ✅ 模型和tokenizer加载成功")
            else:
                print("   ⚠️ 模型和tokenizer加载失败")
        
        return True
        
    except Exception as e:
        print(f"❌ SaprotHub集成失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始ESM3集成测试...\n")
    
    tests = [
        ("ESM3导入", test_esm3_import),
        ("PDB文件读取", test_pdb_reading),
        ("ESM3编码", test_esm3_encoding),
        ("SaprotHub集成", test_saprot_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! ESM3集成成功!")
    else:
        print("⚠️ 部分测试失败，请检查安装和配置")

if __name__ == "__main__":
    main() 