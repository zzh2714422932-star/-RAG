#!/usr/bin/env python3
"""
文档解析与分割 - 将PMC文献的标题+摘要分割成适合RAG检索的文本块
根据上周分析报告中的长度分布自动选择分割策略
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================
EMBEDDING_MODEL_LIMIT = 512          # 嵌入模型的token上限
CHUNK_OVERLAP = 50                   # 重叠token数
TOKENIZER_MODEL = "cl100k_base"      # OpenAI embedding使用的编码

# 输入输出路径（根据你的实际情况修改）
INPUT_CSV_PATH = "cleaned_articles.csv"   # 若已有清洗后的CSV，直接读取
# 如果没有CSV，可以指定XML目录重新生成DataFrame（见下面函数）
XML_DATA_DIR = os.path.expanduser("/Users/zhou/med_rag_data/pubmed_oa/")
OUTPUT_DIR = "./chunking_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 第一部分：数据加载与清洗（若已有清洗后的DataFrame可跳过XML解析）
# ============================================================

def load_or_parse_data() -> pd.DataFrame:
    """
    优先读取已有的清洗后CSV，否则从XML目录重新解析并清洗
    """
    if os.path.exists(INPUT_CSV_PATH):
        print(f"加载已有清洗数据: {INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH)
        # 确保必要的列存在
        required_cols = ['doc_id', 'title', 'abstract', 'journal', 'pub_year']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV中缺少列: {col}")
        return df
    else:
        print("未找到清洗后的CSV，开始从XML解析...")
        # 这里复用之前的解析函数（你需要确保parse_pmc_xml函数存在）
        # 如果之前没有保存清洗后的数据，建议先运行一次数据清洗并保存CSV
        # 下面给出一个简化的加载流程，实际你可能需要将之前的load_data_from_xml等函数复制过来
        from data_analysis import load_data_from_xml, clean_dataframe  # 假设之前脚本中有这些函数
        df_raw = load_data_from_xml(XML_DATA_DIR, max_files=None)  # 全部加载
        df_clean = clean_dataframe(df_raw)
        # 生成doc_id
        df_clean = add_doc_id(df_clean)
        # 保存为CSV供下次使用
        df_clean.to_csv(INPUT_CSV_PATH, index=False)
        print(f"已保存清洗后数据至 {INPUT_CSV_PATH}")
        return df_clean


def add_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    """为每篇文献生成唯一标识符，直接使用行索引，确保非空"""
    df['doc_id'] = 'DOC_' + df.index.astype(str)
    return df


# ============================================================
# 第二部分：文本分割器（支持Token计数和智能分割）
# ============================================================

class TokenTextSplitter:
    """
    封装LangChain的RecursiveCharacterTextSplitter，使用tiktoken计算token长度
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(TOKENIZER_MODEL)
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            add_start_index=True,
        )
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.encoding.encode(text))
    
    def split_document(self, title: str, abstract: str) -> List[str]:
        """将标题+摘要拼接后分割成多个文本块"""
        full_text = f"{title}\n{abstract}" if title else abstract
        chunks = self.splitter.split_text(full_text)
        return chunks


def decide_strategy(df: pd.DataFrame, tokenizer) -> Dict[str, Any]:
    """
    根据上周分析报告中的长度分布自动选择分割策略
    返回: {'strategy': 'no_split' or 'sliding_window', 
           'chunk_size': int, 
           'chunk_overlap': int}
    """
    # 计算拼接后的token长度（抽样或全量，这里抽样加速）
    sample_texts = (df['title'].fillna('') + ' ' + df['abstract']).tolist()[:500]
    token_counts = [tokenizer._count_tokens(t) for t in sample_texts]
    p95 = np.percentile(token_counts, 95)
    
    if p95 <= EMBEDDING_MODEL_LIMIT:
        strategy = "no_split"
        chunk_size = EMBEDDING_MODEL_LIMIT
        chunk_overlap = 0
        print(f"策略: 整体不分割 (95%分位数={p95:.0f} ≤ {EMBEDDING_MODEL_LIMIT})")
    else:
        strategy = "sliding_window"
        chunk_size = EMBEDDING_MODEL_LIMIT
        chunk_overlap = CHUNK_OVERLAP
        print(f"策略: 重叠滑动窗口 (95%分位数={p95:.0f} > {EMBEDDING_MODEL_LIMIT})")
    
    return {
        'strategy': strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'p95_token': p95
    }


# ============================================================
# 第三部分：执行分割并生成块数据集
# ============================================================

def create_chunks_for_document(doc: Dict, splitter: TokenTextSplitter, strategy: str) -> List[Dict]:
    """
    对单篇文献生成文本块
    doc包含: doc_id, title, abstract, journal, pub_year
    """
    title = doc.get('title', '')
    abstract = doc.get('abstract', '')
    full_text = f"{title}\n{abstract}" if title else abstract
    
    if strategy == "no_split":
        # 整体不分割
        token_count = splitter._count_tokens(full_text)
        chunk = {
            "chunk_id": doc['doc_id'],
            "text": full_text,
            "doc_id": doc['doc_id'],
            "chunk_index": 0,
            "total_chunks": 1,
            "source_title": title,
            "token_count": token_count,
            "journal": doc.get('journal', ''),
            "pub_year": doc.get('pub_year', '')
        }
        return [chunk]
    else:
        # 滑动窗口分割
        chunks_text = splitter.split_document(title, abstract)
        chunks = []
        for i, text in enumerate(chunks_text):
            chunk = {
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text": text,
                "doc_id": doc['doc_id'],
                "chunk_index": i,
                "total_chunks": len(chunks_text),
                "source_title": title,
                "token_count": splitter._count_tokens(text),
                "journal": doc.get('journal', ''),
                "pub_year": doc.get('pub_year', '')
            }
            chunks.append(chunk)
        return chunks


def process_chunking(df: pd.DataFrame, strategy_info: Dict) -> pd.DataFrame:
    """处理所有文献，返回所有块组成的DataFrame"""
    splitter = TokenTextSplitter(
        chunk_size=strategy_info['chunk_size'],
        chunk_overlap=strategy_info['chunk_overlap']
    )
    
    all_chunks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="分割文献"):
        doc = row.to_dict()
        chunks = create_chunks_for_document(doc, splitter, strategy_info['strategy'])
        all_chunks.extend(chunks)
    
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.reset_index(drop=True, inplace=True)
    return chunks_df


# ============================================================
# 第四部分：保存结果与统计报告
# ============================================================

def save_chunks(chunks_df: pd.DataFrame, output_dir: str) -> str:
    """保存文本块数据集为Parquet文件（高效压缩）"""
    output_path = os.path.join(output_dir, "text_chunks.parquet")
    chunks_df.to_parquet(output_path, index=False)
    print(f"文本块数据集已保存至: {output_path}")
    return output_path


def generate_stats_report(df_raw: pd.DataFrame, chunks_df: pd.DataFrame, 
                          strategy_info: Dict, output_dir: str) -> None:
    """生成处理日志和统计报告（JSON格式）- 修复Index错误版本"""
    # 确保 token_count 是数值型 Series
    token_counts = chunks_df['token_count'].astype(float).dropna()
    
    stats = {
        "processed_date": datetime.now().isoformat(),
        "data_split": "train",
        "original_documents": len(df_raw),
        "total_chunks": len(chunks_df),
        "chunks_per_doc": round(len(chunks_df) / len(df_raw), 2) if len(df_raw) > 0 else 0,
        "chunk_size": strategy_info['chunk_size'],
        "chunk_overlap": strategy_info['chunk_overlap'],
        "strategy": strategy_info['strategy'],
        "p95_token_original": float(strategy_info['p95_token']),
        "token_stats": {
            "min_token": float(token_counts.min()),
            "max_token": float(token_counts.max()),
            "mean_token": round(float(token_counts.mean()), 2),
            "median_token": float(token_counts.median()),
            "p95_token_chunks": float(token_counts.quantile(0.95))
        },
        "output_file": "text_chunks.parquet"
    }
    
    report_path = os.path.join(output_dir, "chunking_stats.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计报告已保存至: {report_path}")
    
    # 文本报告
    text_report = os.path.join(output_dir, "chunking_report.txt")
    with open(text_report, 'w', encoding='utf-8') as f:
        f.write("文档分割处理报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理时间: {stats['processed_date']}\n")
        f.write(f"原始文献数: {stats['original_documents']}\n")
        f.write(f"生成文本块总数: {stats['total_chunks']}\n")
        f.write(f"平均每篇文献块数: {stats['chunks_per_doc']}\n")
        f.write(f"分割策略: {stats['strategy']}\n")
        f.write(f"chunk_size: {stats['chunk_size']}, chunk_overlap: {stats['chunk_overlap']}\n")
        f.write(f"\n文本块Token长度统计:\n")
        f.write(f"  最小: {stats['token_stats']['min_token']}\n")
        f.write(f"  最大: {stats['token_stats']['max_token']}\n")
        f.write(f"  平均: {stats['token_stats']['mean_token']}\n")
        f.write(f"  中位数: {stats['token_stats']['median_token']}\n")
        f.write(f"  95%分位数: {stats['token_stats']['p95_token_chunks']}\n")
    print(f"文本报告已保存至: {text_report}")


# ============================================================
# 第五部分：质量验证
# ============================================================

def quality_validation(chunks_df: pd.DataFrame, strategy_info: Dict, sample_size: int = 10) -> None:
    """
    执行质量验证：抽样检查文本块，重点关注多块文献的重叠部分
    """
    print("\n" + "=" * 60)
    print("质量验证报告")
    print("=" * 60)
    
    token_counts = chunks_df['token_count'].astype(float)

    # 1. 基本统计
    total_chunks = len(chunks_df)
    print(f"总块数: {total_chunks}")
    print(f"超过模型限制(>{EMBEDDING_MODEL_LIMIT})的块数: {(chunks_df['token_count'] > EMBEDDING_MODEL_LIMIT).sum()}")
    print(f"其中最大token数: {chunks_df['token_count'].max()}")
    
    # 2. 是否包含标题（检查source_title是否非空）
    missing_title = chunks_df['source_title'].isna() | (chunks_df['source_title'] == '')
    print(f"缺失标题的块数: {missing_title.sum()}")
    
    # 3. 抽样检查（随机抽取sample_size个块）
    sample_chunks = chunks_df.sample(min(sample_size, len(chunks_df)), random_state=42)
    print("\n--- 随机抽样文本块预览 ---")
    for idx, row in sample_chunks.iterrows():
        print(f"\n[Chunk: {row['chunk_id']}]")
        print(f"  归属文献: {row['doc_id']} | 块序号: {row['chunk_index']}/{row['total_chunks']}")
        print(f"  Token数: {row['token_count']}")
        print(f"  文本预览: {row['text'][:150]}...")
    
    # 4. 多块文献重点检查（找到总块数>=2的文献）
    multi_chunk_docs = chunks_df[chunks_df['total_chunks'] >= 2]['doc_id'].unique()
    print(f"\n--- 多块文献检查 (共有{len(multi_chunk_docs)}篇文献被分成2块以上) ---")
    for doc_id in multi_chunk_docs[:5]:  # 只显示前5个示例
        doc_chunks = chunks_df[chunks_df['doc_id'] == doc_id].sort_values('chunk_index')
        print(f"\n文献 {doc_id} 被分成 {len(doc_chunks)} 块")
        for i, row in doc_chunks.iterrows():
            print(f"  块{row['chunk_index']}: Token数={row['token_count']}, 文本开头={row['text'][:80]}...")
        # 检查重叠（若策略为滑动窗口，相邻块应有重复文本）
        if strategy_info['strategy'] == 'sliding_window' and len(doc_chunks) >= 2:
            # 简单检查：第一个块的结尾和第二个块的开头是否有相似内容
            first_end = doc_chunks.iloc[0]['text'][-100:]
            second_start = doc_chunks.iloc[1]['text'][:100]
            print(f"  重叠检查: 块0结尾 vs 块1开头 -> {first_end[:50]}... vs ...{second_start[:50]}")
    
    # 5. 不完整截断检查（检查文本是否以句号或段落结束，简单启发）
    suspicious = chunks_df[~chunks_df['text'].str.endswith(('.', '。', '?', '!', '\n'))]
    print(f"\n可能不完整截断的块数（末尾无句号/段落）: {len(suspicious)}")
    if len(suspicious) > 0:
        print("示例:", suspicious[['chunk_id', 'text']].iloc[0]['text'][-100:])


# ============================================================
# 主程序
# ============================================================

def main():
    print("开始文档解析与分割...")
    
    # 1. 加载数据
    df = load_or_parse_data()
    print(f"加载文献数: {len(df)}")
    
    # 2. 确保有doc_id
    if 'doc_id' not in df.columns:
        df = add_doc_id(df)
    
    # 3. 初始化临时tokenizer用于决策
    temp_splitter = TokenTextSplitter(chunk_size=EMBEDDING_MODEL_LIMIT, chunk_overlap=0)
    strategy_info = decide_strategy(df, temp_splitter)
    
    # 4. 执行分割
    chunks_df = process_chunking(df, strategy_info)
    print(f"生成文本块数: {len(chunks_df)}")
    
    # 5. 保存结果
    output_path = save_chunks(chunks_df, OUTPUT_DIR)
    
    # 6. 生成统计报告
    generate_stats_report(df, chunks_df, strategy_info, OUTPUT_DIR)
    
    # 7. 质量验证
    quality_validation(chunks_df, strategy_info, sample_size=10)
    
    print("\n✅ 全部完成！")
    print(f"文本块数据集: {output_path}")
    print(f"处理报告: {OUTPUT_DIR}/chunking_stats.json 和 chunking_report.txt")


if __name__ == "__main__":
    main()