import os
import glob
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from lxml import etree
import tiktoken
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载：从本地 PMC XML 文件提取字段
# ============================================================

def parse_pmc_xml(file_path: str) -> Optional[Dict[str, Any]]:
    """解析单个 PMC XML 文件，提取标题、摘要、期刊、年份、PMCID"""
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()

        # 命名空间处理（PMC 标准）
        ns = {'pmc': 'http://www.ncbi.nlm.nih.gov/pmc/articles'}

        # 标题
        title_elem = root.find('.//article-title', namespaces=ns)
        title = ' '.join(title_elem.itertext()).strip() if title_elem is not None else ''

        # 摘要
        abstract_elem = root.find('.//abstract', namespaces=ns)
        abstract = ' '.join(abstract_elem.itertext()).strip() if abstract_elem is not None else ''

        # 期刊名称
        journal_title_elem = root.find('.//journal-title', namespaces=ns)
        journal = ' '.join(journal_title_elem.itertext()).strip() if journal_title_elem is not None else ''

        # PMC ID (如 PMC1234567)
        pmcid_elem = root.find(".//article-id[@pub-id-type='pmcid']", namespaces=ns)
        pmid = pmcid_elem.text if pmcid_elem is not None else ''

        # 出版年份
        pub_year_elem = root.find('.//pub-date/year', namespaces=ns)
        pub_year = pub_year_elem.text if pub_year_elem is not None else ''

        return {
            'title': title,
            'abstract': abstract,
            'journal': journal,
            'pmid': pmid,
            'pub_year': pub_year,
            'file_path': file_path
        }
    except Exception as e:
        # 忽略解析失败的单个文件
        return None


def load_data_from_xml(data_dir: str, max_files: int = 5000000) -> pd.DataFrame:
    """
    从指定目录递归查找所有 .nxml 文件，加载并解析
    
    max_files: 限制加载文件数，便于快速测试
    """
    xml_files = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
    if not xml_files:
        xml_files = glob.glob(os.path.join(data_dir, '*.xml'))

    if max_files and len(xml_files) > max_files:
        xml_files = xml_files[:max_files]

    print(f"找到 {len(xml_files)} 个 XML 文件，开始解析...")
    records = []
    for f in tqdm(xml_files, desc="解析进度"):
        rec = parse_pmc_xml(f)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"成功解析 {len(df)} 条记录")
    return df


# ============================================================
# 2. 字段完整性检查与清洗
# ============================================================

def check_field_completeness(df: pd.DataFrame) -> Dict[str, Dict]:
    """计算每个字段的缺失率并给出清洗建议"""
    results = {}
    for col in df.columns:
        total = len(df)
        missing = df[col].isna().sum()
        empty = (df[col] == '').sum() if df[col].dtype == 'object' else 0
        effective_missing = missing + empty
        rate = round(effective_missing / total * 100, 2)

        if rate > 20:
            strategy = "丢弃该字段"
        elif rate > 1:
            strategy = "填充或考虑丢弃"
        else:
            strategy = "保留"

        results[col] = {
            'total': total,
            'missing_count': missing,
            'empty_count': empty,
            'missing_rate': rate,
            'strategy': strategy
        }
    return results


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """过滤掉 abstract 缺失或过短的记录"""
    df_clean = df[df['abstract'].notna() & (df['abstract'] != '')].copy()
    df_clean['abstract_len'] = df_clean['abstract'].str.len()
    df_clean = df_clean[df_clean['abstract_len'] >= 50]   # 至少 50 字符
    df_clean.drop(columns=['abstract_len'], inplace=True)
    print(f"清洗后保留 {len(df_clean)} 条记录 (原始 {len(df)})")
    return df_clean


# ============================================================
# 3. 关键字段分析 (journal, pub_year, pmid)
# ============================================================

def analyze_key_fields(df: pd.DataFrame) -> Dict:
    """评估期刊、年份、PMID 的元数据过滤能力"""
    result = {}

    # 期刊
    if 'journal' in df.columns:
        result['total_journals'] = df['journal'].nunique()
        top5 = df['journal'].value_counts().head(5).to_dict()
        result['top_journals'] = top5

    # 年份
    if 'pub_year' in df.columns:
        valid_years = df[df['pub_year'].notna() & (df['pub_year'] != '')]
        if len(valid_years) > 0:
            try:
                years = valid_years['pub_year'].astype(int)
                result['year_min'] = int(years.min())
                result['year_max'] = int(years.max())
                recent = years[years >= 2020]
                result['recent_5y_count'] = len(recent)
                result['recent_5y_pct'] = round(len(recent) / len(years) * 100, 2)
            except:
                result['year_parse_error'] = True

    # PMID 覆盖率
    if 'pmid' in df.columns:
        has_pmid = df['pmid'].notna() & (df['pmid'] != '')
        result['pmid_coverage'] = round(has_pmid.sum() / len(df) * 100, 2)

    # 实际筛选示例：近5年 Nature 期刊文献数量
    if 'journal' in df.columns and 'pub_year' in df.columns:
        mask = (df['journal'] == 'Nature') & (df['pub_year'].astype(str).str.isdigit()) & (df['pub_year'].astype(int) >= 2020)
        result['nature_recent_count'] = mask.sum()

    return result


# ============================================================
# 4. 领域内容理解：缩写、IMRaD、概念变体、高频词
# ============================================================

def extract_abbreviations(text: str) -> List[str]:
    """提取连续大写字母组成的缩写 (2-5个字符)"""
    if not isinstance(text, str):
        return []
    pattern = r'\b[A-Z]{2,5}\b'
    return re.findall(pattern, text)


def detect_imrad_structure(text: str) -> Dict[str, bool]:
    """检测摘要中是否包含典型的 IMRaD 章节关键词"""
    patterns = {
        'Background': r'\b(Background|Introduction|OBJECTIVE|Aim)\b',
        'Methods': r'\b(Methods|Methodology|Patients and Methods|Design|Procedure)\b',
        'Results': r'\b(Results|Findings|Outcomes)\b',
        'Conclusions': r'\b(Conclusions|Conclusion|Discussion|Interpretation)\b'
    }
    detected = {}
    for sec, pat in patterns.items():
        detected[sec] = bool(re.search(pat, text, re.IGNORECASE))
    return detected


def find_concept_variations(df: pd.DataFrame) -> Dict:
    """查找同一医学概念的不同表述出现频率"""
    variations = {
        'heart attack': ['heart attack', 'myocardial infarction', 'MI', 'acute coronary syndrome'],
        'cancer': ['cancer', 'malignancy', 'tumor', 'neoplasm', 'carcinoma'],
        'high blood pressure': ['high blood pressure', 'hypertension', 'HBP']
    }
    results = {}
    for concept, var_list in variations.items():
        counts = {}
        for var in var_list:
            cnt = df['abstract'].str.contains(var, case=False, na=False).sum()
            if cnt > 0:
                counts[var] = int(cnt)
        results[concept] = counts
    return results


def analyze_text_features(df: pd.DataFrame, sample_size_per_group: int = 30) -> Dict:
    """
    按文本长度分层抽样，分析缩写密度、IMRaD 结构覆盖率
    """
    df = df.copy()
    df['len'] = df['abstract'].str.len()
    # 三等分
    df['len_group'] = pd.qcut(df['len'], q=3, labels=['short', 'medium', 'long'], duplicates='drop')
    groups = {}
    all_abbreviations = []

    for grp in ['short', 'medium', 'long']:
        sub = df[df['len_group'] == grp]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(sample_size_per_group, len(sub)), random_state=42)
        groups[grp] = sample
        for _, row in sample.iterrows():
            if row['abstract']:
                all_abbreviations.extend(extract_abbreviations(row['abstract']))

    abbrev_counter = Counter([a for a in all_abbreviations if len(a) >= 2])
    top_abbreviations = abbrev_counter.most_common(20)

    # IMRaD 覆盖率
    imrad_sample = df.sample(min(100, len(df)), random_state=42)
    imrad_detected = 0
    for _, row in imrad_sample.iterrows():
        if row['abstract']:
            det = detect_imrad_structure(row['abstract'])
            if any(det.values()):
                imrad_detected += 1
    imrad_coverage = round(imrad_detected / len(imrad_sample) * 100, 2)

    return {
        'top_abbreviations': top_abbreviations,
        'total_unique_abbreviations': len(abbrev_counter),
        'imrad_coverage': imrad_coverage
    }


def generate_word_frequency(df: pd.DataFrame, top_n: int = 30) -> List[tuple]:
    """统计英文摘要中的高频词（去除停用词）"""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()

    # 补充医学常见无意义词
    extra_stop = {'et', 'al', 'fig', 'table', 'supplementary', 'online', 'using', 'also', 'however', 'may'}
    stop_words.update(extra_stop)

    word_counter = Counter()
    for abstract in df['abstract'].dropna().tolist():
        text = abstract.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        word_counter.update(filtered)

    return word_counter.most_common(top_n)


# ============================================================
# 5. Token 长度分析及分割策略
# ============================================================

def token_lengths(texts: List[str], encoding_name: str = "cl100k_base") -> List[int]:
    """使用 tiktoken 计算每个文本的 token 数量"""
    enc = tiktoken.get_encoding(encoding_name)
    lengths = []
    for t in tqdm(texts, desc="计算Token长度"):
        lengths.append(len(enc.encode(t)))
    return lengths


def analyze_length_distribution(df: pd.DataFrame, embedding_limit: int = 512) -> Dict:
    """
    分析拼接标题+摘要后的 token 长度分布，给出分割策略建议
    """
    full_texts = (df['title'].fillna('') + ' ' + df['abstract']).tolist()
    token_counts = token_lengths(full_texts)
    arr = np.array(token_counts)

    dist = {
        'total': len(token_counts),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': round(arr.mean(), 2),
        'median': int(np.median(arr)),
        'p50': int(np.percentile(arr, 50)),
        'p75': int(np.percentile(arr, 75)),
        'p90': int(np.percentile(arr, 90)),
        'p95': int(np.percentile(arr, 95)),
        'p99': int(np.percentile(arr, 99)),
        'exceed_pct': round((arr > embedding_limit).sum() / len(arr) * 100, 2),
        'embedding_limit': embedding_limit
    }

    # 策略决定
    if dist['p95'] <= embedding_limit:
        strategy = "整体不分割"
        detail = "RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)"
    elif dist['p99'] <= embedding_limit * 2:
        strategy = "重叠滑动窗口"
        detail = f"chunk_size={embedding_limit}, chunk_overlap=50"
    else:
        strategy = "分层处理：短文档整体嵌入，长文档按语义分割"
        detail = "使用 MarkdownHeaderTextSplitter 或自定义章节分割"

    dist['recommended_strategy'] = strategy
    dist['strategy_detail'] = detail
    return dist


# ============================================================
# 6. 生成最终报告
# ============================================================

def generate_report(df_original: pd.DataFrame,
                    df_clean: pd.DataFrame,
                    completeness: Dict,
                    key_fields: Dict,
                    lang_features: Dict,
                    length_dist: Dict,
                    high_freq: List[tuple],
                    concept_vars: Dict,
                    output_file: str = "RAG数据分析与设计说明.txt") -> None:
    """将所有分析结果写入文本文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("《RAG数据分析与设计说明》\n")
        f.write("=" * 60 + "\n\n")

        # 1. 数据集
        f.write("## 一、数据集\n\n")
        f.write(f"原始记录数: {len(df_original)}\n")
        f.write(f"清洗后有效记录数: {len(df_clean)}\n")
        f.write(f"清洗率: {round((1 - len(df_clean)/len(df_original))*100, 2)}%\n\n")
        f.write("### 字段完整性\n")
        f.write("| 字段 | 缺失率 | 清洗策略 |\n")
        f.write("|------|--------|----------|\n")
        for col, info in completeness.items():
            f.write(f"| {col} | {info['missing_rate']}% | {info['strategy']} |\n")
        f.write("\n")

        # 2. 关键字段分析
        f.write("## 二、关键字段与元数据过滤能力\n\n")
        f.write(f"- 期刊种数: {key_fields.get('total_journals', 'N/A')}\n")
        if 'top_journals' in key_fields:
            f.write(f"- 高频期刊示例: {list(key_fields['top_journals'].keys())[:3]}\n")
        f.write(f"- 出版年份范围: {key_fields.get('year_min', 'N/A')} – {key_fields.get('year_max', 'N/A')}\n")
        f.write(f"- 近5年文献占比: {key_fields.get('recent_5y_pct', 'N/A')}%\n")
        f.write(f"- PMID覆盖率: {key_fields.get('pmid_coverage', 'N/A')}%\n")
        f.write(f"- 可以查询 '近5年Nature上的文献'？实际样本中存在 {key_fields.get('nature_recent_count', 0)} 篇\n\n")

        # 3. 文本长度分布
        f.write("## 三、文本长度分布 (Token, embedding上限=512)\n\n")
        f.write(f"- 文档总数: {length_dist['total']}\n")
        f.write(f"- Token 最小值: {length_dist['min']}\n")
        f.write(f"- Token 最大值: {length_dist['max']}\n")
        f.write(f"- Token 平均值: {length_dist['mean']}\n")
        f.write(f"- 95% 分位数: {length_dist['p95']}\n")
        f.write(f"- 99% 分位数: {length_dist['p99']}\n")
        f.write(f"- 超过 512 token 的占比: {length_dist['exceed_pct']}%\n\n")
        f.write(f"### 分割策略\n")
        f.write(f"- 推荐策略: {length_dist['recommended_strategy']}\n")
        f.write(f"- 具体配置: {length_dist['strategy_detail']}\n\n")

        # 4. 领域语言特性
        f.write("## 四、领域语言特性\n\n")
        f.write(f"- IMRaD 结构覆盖率: {lang_features['imrad_coverage']}%\n")
        f.write(f"- 不同缩写数量: {lang_features['total_unique_abbreviations']}\n")
        f.write("- 高频缩写 Top 10:\n")
        for i, (ab, cnt) in enumerate(lang_features['top_abbreviations'][:10], 1):
            f.write(f"  {i}. {ab} ({cnt}次)\n")
        f.write("\n- 高频医学词汇 (Top 20):\n")
        for i, (word, cnt) in enumerate(high_freq[:20], 1):
            f.write(f"  {i}. {word} ({cnt})\n")
        f.write("\n- 概念表述变体示例:\n")
        for concept, variants in concept_vars.items():
            f.write(f"  {concept}: {variants}\n")
        f.write("\n")

        # 5. 补充说明
        f.write("## 五、补充说明\n\n")
        f.write("- 嵌入模型上限为512 token，已考虑标题+摘要拼接。\n")
        f.write("- 后续开发中，建议在向量库中存储 journal、pub_year 作为元数据，便于过滤检索。\n")
        f.write("- 对于极少数超长摘要 (>1024 token)，可采用滑动窗口分割并保留重叠部分。\n")
        f.write("- 医学缩写词典可后续扩展，提高实体识别准确率。\n")
        f.write(f"\n*报告生成时间: {pd.Timestamp.now()}*\n")

    print(f"报告已保存至: {output_file}")


# ============================================================
# 主函数
# ============================================================

def main():
    # 配置路径 - 请根据你的实际数据存放路径修改
    # 例如: /Users/你的用户名/med_rag_data/pubmed_oa/comm_aa
    DATA_DIR = os.path.expanduser("/Users/zhou/med_rag_data/pubmed_oa/")
    MAX_FILES = 50000000   # 可调整，若性能好可增至1000+

    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在 -> {DATA_DIR}")
        print("请修改 DATA_DIR 为实际路径")
        return

    print("=" * 60)
    print("PMC OA 医学文献数据分析 (RAG 准备工作)")
    print("=" * 60)

    # 1. 加载
    df_raw = load_data_from_xml(DATA_DIR, max_files=MAX_FILES)
    if df_raw.empty:
        print("未解析到任何数据，请检查 XML 文件格式。")
        return

    # 2. 完整性
    completeness = check_field_completeness(df_raw)

    # 3. 清洗
    df_clean = clean_dataframe(df_raw)
    if df_clean.empty:
        print("清洗后无有效数据，请降低清洗阈值或检查数据。")
        return

    # 4. 关键字段
    key_fields = analyze_key_fields(df_clean)

    # 5. 语言特征
    lang_features = analyze_text_features(df_clean, sample_size_per_group=30)

    # 6. 高频词
    high_freq = generate_word_frequency(df_clean, top_n=30)

    # 7. 概念变体
    concept_vars = find_concept_variations(df_clean)

    # 8. 长度分布
    length_dist = analyze_length_distribution(df_clean, embedding_limit=512)

    # 9. 生成报告
    generate_report(
        df_original=df_raw,
        df_clean=df_clean,
        completeness=completeness,
        key_fields=key_fields,
        lang_features=lang_features,
        length_dist=length_dist,
        high_freq=high_freq,
        concept_vars=concept_vars,
        output_file="RAG数据分析与设计说明.txt"
    )

    print("\n✅ 分析完成！请查看 RAG数据分析与设计说明.txt")


if __name__ == "__main__":
    main()