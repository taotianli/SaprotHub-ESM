# ESM3在SaprotHub中的集成

本文档说明如何在SaprotHub中集成ESM3（Evolutionary Scale Modeling 3）来替代原有的FoldSeek方法进行蛋白质序列和结构编码。

## 概述

ESM3是Meta AI开发的最新蛋白质语言模型，能够同时处理序列和结构信息。通过集成ESM3，我们可以：

1. **更准确的编码**：ESM3使用最新的深度学习技术，提供更准确的蛋白质表示
2. **结构感知**：能够同时利用序列和3D结构信息
3. **更好的tokenizer**：使用专门的蛋白质tokenizer
4. **现代化架构**：基于最新的Transformer架构

## 文件结构

```
SaprotHub/
├── esm3_protein_encoder.py      # ESM3编码器核心模块
├── saprot-esm.py               # 修改后的主文件（集成ESM3）
├── test_esm3_integration.py    # 集成测试脚本
├── example_esm3_usage.py       # 使用示例脚本
└── README_ESM3_Integration.md  # 本文档
```

## 安装依赖

### 1. 安装ESM3

```bash
pip install fair-esm
```

### 2. 安装其他依赖

```bash
pip install torch transformers numpy
```

## 核心功能

### 1. PDB文件读取

```python
from esm3_protein_encoder import read_pdb_simple, preprocess_coordinates

# 读取PDB文件
sequence, coordinates = read_pdb_simple("protein.pdb")
coordinates = preprocess_coordinates(coordinates)

print(f"序列: {sequence}")
print(f"坐标形状: {coordinates.shape}")
```

### 2. ESM3编码

```python
from esm3_protein_encoder import get_esm3_encoding

# 获取ESM3编码
encoding_results, sequence, coordinates = get_esm3_encoding(
    pdb_file="protein.pdb",
    model=None,  # 自动加载模型
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_structure=True  # 使用结构信息
)

print(f"编码结果: {list(encoding_results.keys())}")
```

### 3. Tokenization

```python
from esm3_protein_encoder import get_esm3_tokenizer

# 获取tokenizer
tokenizer = get_esm3_tokenizer()

# 对序列进行tokenization
tokens = tokenizer(sequence, return_tensors="pt")
print(f"Token形状: {tokens['input_ids'].shape}")
```

### 4. 完整编码流程

```python
from esm3_protein_encoder import encode_pdb_to_tokens

# 完整的PDB到token的编码
result = encode_pdb_to_tokens(
    pdb_file="protein.pdb",
    model=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_structure=True
)

print(f"序列: {result['sequence']}")
print(f"坐标: {result['coordinates'].shape}")
print(f"Token: {result['tokens']['input_ids'].shape}")
print(f"编码: {list(result['encoding'].keys())}")
```

## 在SaprotHub中的使用

### 1. 自动检测

SaprotHub会自动检测ESM3是否可用：

```python
# 在saprot-esm.py中
if ESM3_AVAILABLE:
    print("使用ESM3进行编码")
    # 使用ESM3方法
else:
    print("回退到FoldSeek方法")
    # 使用原有方法
```

### 2. 替换原有函数

以下函数已被ESM3版本替换：

- `pdb2sequence()`: 使用ESM3编码替代FoldSeek
- `pdb2sa()`: 使用ESM3进行批量编码

### 3. 新增函数

- `get_esm3_model_and_tokenizer()`: 获取ESM3模型和tokenizer
- `encode_pdb_with_esm3()`: 使用ESM3编码单个PDB文件
- `batch_encode_pdbs_with_esm3()`: 批量编码PDB文件

## 测试和验证

### 运行测试

```bash
cd SaprotHub
python test_esm3_integration.py
```

### 运行示例

```bash
cd SaprotHub
python example_esm3_usage.py
```

## 性能对比

### ESM3 vs FoldSeek

| 特性 | ESM3 | FoldSeek |
|------|------|----------|
| 编码质量 | 更高 | 标准 |
| 结构感知 | ✅ | ✅ |
| 序列编码 | ✅ | ✅ |
| 现代化架构 | ✅ | ❌ |
| 依赖复杂度 | 中等 | 低 |
| GPU加速 | ✅ | ❌ |

### 使用建议

1. **推荐使用ESM3**：如果ESM3可用，建议使用ESM3进行编码
2. **回退机制**：如果ESM3不可用，系统会自动回退到FoldSeek
3. **GPU加速**：ESM3支持GPU加速，建议在有GPU的环境中运行

## 故障排除

### 常见问题

1. **ESM3导入失败**
   ```bash
   pip install fair-esm --upgrade
   ```

2. **CUDA内存不足**
   ```python
   # 使用CPU
   device = "cpu"
   ```

3. **PDB文件格式错误**
   - 确保PDB文件格式正确
   - 检查是否包含必要的原子信息

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细输出
from esm3_protein_encoder import get_esm3_encoding
result = get_esm3_encoding("protein.pdb", verbose=True)
```

## 高级用法

### 自定义编码

```python
from esm3_protein_encoder import encode_sequence_only, encode_sequence_and_structure

# 仅序列编码
seq_results = encode_sequence_only(model, sequence, device)

# 序列+结构编码
struct_results = encode_sequence_and_structure(model, sequence, coordinates, device)
```

### 批量处理

```python
from saprot_esm import batch_encode_pdbs_with_esm3

pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
results = batch_encode_pdbs_with_esm3(pdb_files, use_structure=True)
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

遵循SaprotHub的原有许可证。 