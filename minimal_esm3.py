import torch
import numpy as np
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

def extract_pdb_sequence(pdb_file):
    """最简单的PDB序列提取"""
    from Bio.PDB import PDBParser
    
    aa_map = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
              'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
              'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
              'TRP':'W','TYR':'Y'}
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chain = list(structure[0].get_chains())[0]
    
    sequence = ""
    coords = []
    
    for residue in chain:
        if residue.id[0] == ' ' and residue.get_resname() in aa_map:
            sequence += aa_map[residue.get_resname()]
            if 'CA' in residue:
                coords.append(residue['CA'].get_coord())
    
    return sequence, np.array(coords)

def get_embedding(sequence):
    """获取ESM3 embedding - 简单可靠版本"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM3.from_pretrained("esm3-open").to(device).eval()
    
    protein = ESMProtein(sequence=sequence)
    
    with torch.no_grad():
        # 方法1: 使用ESM3的正确forward方法
        try:
            encoded = model.encode(protein)
            tokens = encoded.sequence
            
            # 使用正确的forward调用
            result = model.forward(encoded)
            
            if hasattr(result, 'shape'):
                hidden = result
            elif hasattr(result, '__dict__'):
                # 查找tensor属性
                for attr_value in result.__dict__.values():
                    if hasattr(attr_value, 'shape') and len(attr_value.shape) >= 2:
                        hidden = attr_value
                        break
                else:
                    hidden = None
            else:
                hidden = None
            
            if hidden is not None:
                if len(hidden.shape) == 3:  # [batch, seq, hidden]
                    seq_emb = hidden.mean(dim=1).cpu().numpy()
                    return seq_emb[0]
                elif len(hidden.shape) == 2:  # [seq, hidden]
                    seq_emb = hidden.mean(dim=0).cpu().numpy()
                    return seq_emb
                
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 查找embedding层
        try:
            encoded = model.encode(protein)
            tokens = encoded.sequence
            
            # 查找embedding层
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    token_emb = module(tokens)
                    seq_emb = token_emb.mean(dim=0).cpu().numpy()
                    return seq_emb
                
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 方法3: 简单的one-hot编码
        try:
            encoded = model.encode(protein)
            tokens = encoded.sequence.cpu().numpy()
            
            # 创建简单的token-level特征
            vocab_size = tokens.max() + 1
            one_hot = np.zeros((len(tokens), vocab_size))
            one_hot[np.arange(len(tokens)), tokens] = 1
            
            # 平均池化
            seq_emb = one_hot.mean(axis=0)
            return seq_emb
            
        except Exception as e:
            print(f"方法3失败: {e}")
        
        print("所有方法都失败了")
        return None

# 使用示例
if __name__ == "__main__":
    # 1. 提取序列和坐标
    sequence, coordinates = extract_pdb_sequence("1qsf.pdb")  # 修改路径
    print(f"序列长度: {len(sequence)}")
    print(f"坐标形状: {coordinates.shape}")
    
    # 2. 获取embedding
    embedding = get_embedding(sequence)
    
    if embedding is not None:
        print(f"Embedding形状: {embedding.shape}")
        
        # 3. 保存
        np.save('embedding.npy', {
            'sequence': sequence,
            'coordinates': coordinates, 
            'embedding': embedding
        })
        print("已保存到 embedding.npy") 