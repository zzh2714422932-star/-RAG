#!/usr/bin/env python3
"""
向量化与索引构建 - 使用 ChromaDB 存储文本块向量
"""

import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================
# 数据路径
CHUNKS_DATA_PATH = "./chunking_output/text_chunks.parquet"   # 上一阶段生成的文本块
OUTPUT_DIR = "./vector_index_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 嵌入模型选择（根据内存/显存调整）
# - BAAI/bge-small-en-v1.5: 384维，~1GB内存，推荐
# - BAAI/bge-base-en-v1.5: 768维，~8GB内存
# - BAAI/bge-large-en-v1.5: 1024维，~16GB内存
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # small 对应384，base为768，large为1024

# ChromaDB 持久化路径
PERSIST_DIR = os.path.join(OUTPUT_DIR, "chroma_db")
COLLECTION_NAME = "medical_chunks"

# 批处理大小（根据内存调整，越小越省内存）
BATCH_SIZE = 64

# 查询测试参数
TEST_QUERY = "What are the latest treatments for acute myocardial infarction?"
TOP_K = 5

# ============================================================
# 1. 加载文本块数据集
# ============================================================

def load_chunks(file_path: str) -> pd.DataFrame:
    """加载 Parquet 格式的文本块"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文本块数据：{file_path}")
    df = pd.read_parquet(file_path)
    print(f"加载 {len(df)} 个文本块")
    return df

# ============================================================
# 2. 嵌入模型加载（封装）
# ============================================================

class BGEEmbedder:
    """BGE 嵌入模型封装，支持批量编码和指令前缀（用于查询）"""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"加载模型 {model_name}，维度 {self.dimension}，设备 {device}")
    
    def encode_documents(self, texts, batch_size=64) -> np.ndarray:
        """编码文档（不加指令）"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # BGE 建议归一化
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """编码查询（添加 BGE 指令前缀）"""
        # BGE 查询指令（根据官方建议）
        instruction = "Represent this sentence for searching relevant passages: "
        text_with_instruction = instruction + query
        return self.model.encode(
            [text_with_instruction],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

# ============================================================
# 3. ChromaDB 集合构建
# ============================================================

def create_chroma_collection(persist_dir: str, collection_name: str, embedding_dim: int):
    """
    创建 ChromaDB 持久化集合
    使用余弦相似度（默认），支持元数据过滤
    """
    # 初始化 ChromaDB 持久化客户端
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 检查集合是否已存在，若存在则删除（重新构建）
    try:
        client.delete_collection(collection_name)
        print(f"已删除旧集合 {collection_name}")
    except:
        pass
    
    # 创建新集合
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )
    print(f"创建集合 {collection_name}，相似度度量：cosine")
    return collection

# ============================================================
# 4. 批量向量化并添加到 Chroma
# ============================================================

def index_chunks(chunks_df: pd.DataFrame, embedder: BGEEmbedder, collection, batch_size: int = 64):
    """
    将文本块向量化并添加到 Chroma 集合
    元数据包含：doc_id, chunk_index, total_chunks, journal, pub_year, token_count, source_title
    """
    total = len(chunks_df)
    # 准备数据
    ids = []
    metadatas = []
    documents = []
    
    # 遍历所有块，分批编码
    for i in tqdm(range(0, total, batch_size), desc="批量编码与索引"):
        batch = chunks_df.iloc[i:i+batch_size]
        texts = batch['text'].tolist()
        
        # 生成向量（文档编码）
        embeddings = embedder.encode_documents(texts, batch_size=batch_size)
        
        # 准备 ids 和 metadatas
        batch_ids = batch['chunk_id'].tolist()
        batch_metadatas = []
        for _, row in batch.iterrows():
            meta = {
                "doc_id": row['doc_id'],
                "chunk_index": int(row['chunk_index']),
                "total_chunks": int(row['total_chunks']),
                "source_title": row['source_title'][:200] if row['source_title'] else "",  # 限制长度
                "token_count": int(row['token_count']),
            }
            # 可选元数据
            if 'journal' in row and pd.notna(row['journal']):
                meta['journal'] = row['journal'][:100]
            if 'pub_year' in row and pd.notna(row['pub_year']):
                meta['pub_year'] = str(row['pub_year'])
            batch_metadatas.append(meta)
        
        # 添加到 Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas,
            documents=texts,
        )
    
    # 获取最终集合大小
    count = collection.count()
    print(f"索引完成，共 {count} 个向量")

# ============================================================
# 5. 查询函数
# ============================================================

def query_collection(collection, embedder, query_text: str, n_results: int = 5, where_filter: dict = None):
    """
    执行查询，支持元数据过滤
    """
    # 生成查询向量（带 BGE 指令）
    query_embedding = embedder.encode_query(query_text).tolist()
    
    # 执行查询
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    return results

# ============================================================
# 6. 质量验证与测试
# ============================================================

def run_quality_validation(collection, embedder, chunks_df: pd.DataFrame):
    """
    执行质量验证：
    1. 基本统计（向量数量、样本元数据）
    2. 自相似性测试（从索引中抽取文本作为查询）
    3. 边界情况（空查询、超长查询）
    4. 元数据过滤功能测试
    """
    print("\n" + "="*60)
    print("质量验证报告")
    print("="*60)
    
    # 1. 基本统计
    total_vectors = collection.count()
    print(f"1. 向量总数: {total_vectors}")
    print(f"   文本块总数: {len(chunks_df)}")
    assert total_vectors == len(chunks_df), "向量数量与文本块数量不一致！"
    
    # 查看一个样本元数据
    sample = collection.get(limit=1, include=["metadatas"])
    if sample['metadatas']:
        print(f"   样本元数据示例: {sample['metadatas'][0]}")
    
    # 2. 自相似性测试（随机抽取一个块，查询自身）
    random_chunk = chunks_df.sample(1).iloc[0]
    query_text = random_chunk['text']
    print(f"\n2. 自相似性测试 - 查询文本块 (doc_id={random_chunk['doc_id']})")
    results = query_collection(collection, embedder, query_text, n_results=3)
    print(f"   返回 {len(results['ids'][0])} 个结果")
    print(f"   最相似块 doc_id: {results['metadatas'][0][0]['doc_id']}")
    print(f"   相似度距离: {results['distances'][0][0]:.4f} (余弦距离越小越相似)")
    
    # 3. 边界情况测试
    print("\n3. 边界情况测试")
    # 空查询
    try:
        empty_results = query_collection(collection, embedder, "", n_results=1)
        print("   ✓ 空查询处理正常")
    except Exception as e:
        print(f"   ✗ 空查询异常: {e}")
    
    # 超长查询（>512 tokens 可能会截断，但BGE支持长文本）
    long_query = " ".join(["cancer treatment"] * 1000)  # 约2000 tokens
    try:
        long_results = query_collection(collection, embedder, long_query, n_results=1)
        print("   ✓ 超长查询处理正常")
    except Exception as e:
        print(f"   ✗ 超长查询异常: {e}")
    
    # 4. 元数据过滤测试
    print("\n4. 元数据过滤测试")
    # 假设存在 journal 字段
    if 'journal' in chunks_df.columns:
        # 找一个出现过的期刊
        sample_journal = chunks_df['journal'].dropna().iloc[0] if len(chunks_df) > 0 else None
        if sample_journal:
            where_filter = {"journal": sample_journal}
            print(f"   过滤条件: journal = {sample_journal}")
            filtered_results = query_collection(collection, embedder, "therapy", n_results=2, where_filter=where_filter)
            print(f"   返回结果数: {len(filtered_results['ids'][0])}")
            if len(filtered_results['ids'][0]) > 0:
                print(f"   结果中的 journal: {filtered_results['metadatas'][0][0].get('journal')}")
        else:
            print("   跳过（无 journal 数据）")
    else:
        print("   跳过（数据集中无 journal 字段）")
    
    # 5. 测试查询示例
    print("\n5. 示例查询 - " + TEST_QUERY)
    test_results = query_collection(collection, embedder, TEST_QUERY, n_results=TOP_K)
    print(f"   返回 {len(test_results['ids'][0])} 条结果")
    for i, (doc_id, meta, dist) in enumerate(zip(test_results['ids'][0], test_results['metadatas'][0], test_results['distances'][0])):
        print(f"   {i+1}. doc_id={doc_id}, 距离={dist:.4f}, 标题={meta.get('source_title', 'N/A')[:50]}...")

# ============================================================
# 7. 保存统计信息
# ============================================================

def save_stats(collection, chunks_df: pd.DataFrame, embedder, output_dir: str):
    """保存索引统计信息到 JSON"""
    stats = {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "index_built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_size_stats": {
            "mean": float(chunks_df['token_count'].mean()),
            "max": int(chunks_df['token_count'].max()),
            "min": int(chunks_df['token_count'].min()),
        } if 'token_count' in chunks_df.columns else {},
        "metadata_fields": list(chunks_df.columns.intersection(['doc_id', 'chunk_index', 'total_chunks', 'source_title', 'journal', 'pub_year', 'token_count'])),
        "db_persist_dir": PERSIST_DIR,
        "batch_size": BATCH_SIZE,
    }
    stats_path = os.path.join(output_dir, "index_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"\n统计信息已保存至: {stats_path}")

# ============================================================
# 主程序
# ============================================================

def main():
    print("开始向量化与索引构建...")
    
    # 1. 加载文本块
    chunks_df = load_chunks(CHUNKS_DATA_PATH)
    
    # 2. 加载嵌入模型
    embedder = BGEEmbedder(MODEL_NAME)
    
    # 3. 创建 ChromaDB 集合
    collection = create_chroma_collection(PERSIST_DIR, COLLECTION_NAME, EMBEDDING_DIM)
    
    # 4. 索引文本块
    index_chunks(chunks_df, embedder, collection, BATCH_SIZE)
    
    # 5. 保存统计信息
    save_stats(collection, chunks_df, embedder, OUTPUT_DIR)
    
    # 6. 执行质量验证与测试
    run_quality_validation(collection, embedder, chunks_df)
    
    print("\n✅ 向量化与索引构建完成！")
    print(f"向量数据库目录: {PERSIST_DIR}")
    print(f"统计信息: {OUTPUT_DIR}/index_stats.json")

if __name__ == "__main__":
    main()