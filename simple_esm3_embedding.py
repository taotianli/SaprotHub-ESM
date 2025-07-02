import torch
import numpy as np
import pandas as pd
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

# 简单的PDB处理
def get_sequence_from_pdb(pdb_file_path):
    """从PDB文件提取序列和坐标"""
    try:
        from Bio.PDB import PDBParser
        
        # 氨基酸映射
        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file_path)
        
        sequence = ""
        coordinates = []
        
        # 取第一个模型的第一条链
        chain = list(structure[0].get_chains())[0]
        
        for residue in chain:
            if residue.id[0] == ' ':  # 标准残基
                resname = residue.get_resname()
                if resname in aa_dict:
                    sequence += aa_dict[resname]
                    
                    # 获取CA原子坐标
                    if 'CA' in residue:
                        coordinates.append(residue['CA'].get_coord())
        
        coordinates = np.array(coordinates)
        
        print(f"✅ 序列长度: {len(sequence)}")
        print(f"✅ 坐标形状: {coordinates.shape}")
        print(f"✅ 序列: {sequence[:50]}...")
        
        return sequence, coordinates
        
    except ImportError:
        print("❌ 需要安装BioPython: pip install biopython")
        return None, None
    except Exception as e:
        print(f"❌ 处理PDB文件失败: {str(e)}")
        return None, None

def get_esm3_embedding(sequence):
    """获取ESM3 embedding - 改进版本"""
    print(f"🧬 获取序列embedding...")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM3.from_pretrained("esm3-open").to(device).eval()
    
    # 创建蛋白质对象
    protein = ESMProtein(sequence=sequence)
    
    with torch.no_grad():
        # 方法1: 使用ESM3InferenceClient
        print("🔸 尝试方法1: 使用ESM3InferenceClient...")
        try:
            from esm.sdk.api import ESM3InferenceClient
            
            # 创建推理客户端
            client = ESM3InferenceClient(model)
            
            # 使用客户端进行推理
            result = client.encode(protein)
            
            if result is not None:
                print(f"✅ 客户端结果类型: {type(result)}")
                
                # 检查结果的属性
                for attr_name in dir(result):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(result, attr_name)
                            if hasattr(attr_value, 'shape') and len(attr_value.shape) >= 2:
                                print(f"✅ 找到embedding: {attr_name} {attr_value.shape}")
                                
                                # 生成序列级别embedding
                                if len(attr_value.shape) == 3:  # [batch, seq, hidden]
                                    content_embedding = attr_value[:, 1:-1, :].mean(dim=1) if attr_value.shape[1] > 2 else attr_value.mean(dim=1)
                                    cls_embedding = attr_value[:, 0, :]
                                elif len(attr_value.shape) == 2:  # [seq, hidden]
                                    content_embedding = attr_value[1:-1, :].mean(dim=0, keepdim=True) if attr_value.shape[0] > 2 else attr_value.mean(dim=0, keepdim=True)
                                    cls_embedding = attr_value[0:1, :]
                                else:
                                    content_embedding = attr_value
                                    cls_embedding = attr_value
                                
                                return {
                                    'sequence_embedding': content_embedding.cpu().numpy(),
                                    'cls_embedding': cls_embedding.cpu().numpy(),
                                    'hidden_states': attr_value.cpu().numpy(),
                                    'tokens': result.sequence.cpu().numpy() if hasattr(result, 'sequence') else None
                                }
                        except:
                            continue
                
        except Exception as e:
            print(f"⚠️ 方法1失败: {str(e)}")
        
        # 方法2: 直接使用embedding层
        print("🔸 尝试方法2: 直接使用embedding层...")
        try:
            encoded = model.encode(protein)
            tokens = encoded.sequence
            print(f"✅ Tokens形状: {tokens.shape}")
            
            # 查找并使用embedding层
            embedding_layer = None
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    embedding_layer = module
                    print(f"✅ 找到embedding层: {name}, 权重形状: {module.weight.shape}")
                    break
            
            if embedding_layer is not None:
                # 直接通过embedding层获取token embeddings
                token_embeddings = embedding_layer(tokens)
                print(f"✅ Token embeddings形状: {token_embeddings.shape}")
                
                # 生成序列级别embedding
                if len(token_embeddings.shape) == 2:  # [seq_len, hidden_dim]
                    # 排除特殊tokens的平均池化
                    if token_embeddings.shape[0] > 2:
                        content_embedding = token_embeddings[1:-1, :].mean(dim=0, keepdim=True)
                    else:
                        content_embedding = token_embeddings.mean(dim=0, keepdim=True)
                    
                    # CLS token embedding
                    cls_embedding = token_embeddings[0:1, :]
                    
                    return {
                        'sequence_embedding': content_embedding.cpu().numpy(),
                        'cls_embedding': cls_embedding.cpu().numpy(),
                        'token_embeddings': token_embeddings.cpu().numpy(),
                        'tokens': tokens.cpu().numpy()
                    }
            
        except Exception as e:
            print(f"⚠️ 方法2失败: {str(e)}")
        
        # 方法3: 使用tokens创建简单embedding
        print("🔸 方法3: 从tokens创建embedding...")
        try:
            encoded = model.encode(protein)
            tokens = encoded.sequence
            
            # 查找模型中的embedding层
            embedding_layer = None
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    embedding_layer = module
                    print(f"✅ 找到embedding层: {name}")
                    break
            
            if embedding_layer is not None:
                # 使用找到的embedding层
                token_embeddings = embedding_layer(tokens)
                print(f"✅ Token embeddings: {token_embeddings.shape}")
                
                # 添加batch维度并平均池化
                if len(token_embeddings.shape) == 2:
                    sequence_embedding = token_embeddings.mean(dim=0, keepdim=True)
                else:
                    sequence_embedding = token_embeddings.mean(dim=1)
                
                return {
                    'sequence_embedding': sequence_embedding.cpu().numpy(),
                    'token_embeddings': token_embeddings.cpu().numpy(),
                    'tokens': tokens.cpu().numpy()
                }
            else:
                # 最后的备选：返回tokens的one-hot编码
                print("⚠️ 使用tokens作为基础特征")
                vocab_size = tokens.max().item() + 1
                one_hot = torch.zeros(len(tokens), vocab_size, device=tokens.device)
                one_hot[torch.arange(len(tokens)), tokens] = 1
                
                # 简单平均
                sequence_embedding = one_hot.mean(dim=0, keepdim=True)
                
                return {
                    'sequence_embedding': sequence_embedding.cpu().numpy(),
                    'tokens': tokens.cpu().numpy(),
                    'one_hot': one_hot.cpu().numpy()
                }
                
        except Exception as e:
            print(f"⚠️ 方法3失败: {str(e)}")
        
        print("❌ 所有方法都失败了")
        return None

def main():
    """主函数"""
    print("🚀 简单ESM3 Embedding提取")
    print("="*50)
    
    # 1. 处理PDB文件
    pdb_file = "/content/1qsf.pdb"  # 修改为您的PDB文件路径
    
    # 如果文件不存在，下载1QSF
    import os
    if not os.path.exists(pdb_file):
        print("📥 下载1QSF.pdb...")
        import urllib.request
        url = "https://files.rcsb.org/download/1QSF.pdb"
        urllib.request.urlretrieve(url, pdb_file)
    
    sequence, coordinates = get_sequence_from_pdb(pdb_file)
    
    if sequence is None:
        print("❌ 无法获取序列")
        return
    
    # 2. 获取embedding
    embeddings = get_esm3_embedding(sequence)
    
    if embeddings is None:
        print("❌ 无法获取embedding")
        return
    
    # 3. 保存结果
    print("\n💾 保存结果...")
    
    # 保存为numpy文件
    result_data = {
        'sequence': sequence,
        'coordinates': coordinates,
        **embeddings
    }
    
    np.save('protein_embeddings.npy', result_data, allow_pickle=True)
    print("✅ 已保存到: protein_embeddings.npy")
    
    # 保存序列级别embedding为CSV
    if 'sequence_embedding' in embeddings:
        df = pd.DataFrame(
            embeddings['sequence_embedding'], 
            columns=[f'dim_{i}' for i in range(embeddings['sequence_embedding'].shape[1])]
        )
        df.to_csv('sequence_embedding.csv', index=False)
        print("✅ 序列embedding已保存到: sequence_embedding.csv")
    
    print("\n🎉 完成!")
    print(f"📊 结果包含: {list(embeddings.keys())}")

if __name__ == "__main__":
    main() 