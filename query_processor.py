#!/usr/bin/env python3
"""
查询理解与增强模块
用于RAG系统的医学查询预处理，输出增强后的查询对象
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 静态医学同义词词典（可扩展，建议从MeSH/UMLS导入）
# ============================================================
MEDICAL_SYNONYMS = {
    # 心血管疾病
    "mi": ["myocardial infarction", "heart attack"],
    "heart attack": ["myocardial infarction", "acute coronary syndrome"],
    "myocardial infarction": ["heart attack", "mi"],
    "acs": ["acute coronary syndrome"],
    
    # 糖尿病
    "dm": ["diabetes mellitus", "diabetes"],
    "t2dm": ["type 2 diabetes", "type 2 diabetes mellitus"],
    
    # 高血压
    "htn": ["hypertension", "high blood pressure"],
    "hypertension": ["high blood pressure", "htn"],
    
    # 药物
    "metformin": ["glucophage"],
    "aspirin": ["acetylsalicylic acid"],
    "atorvastatin": ["lipitor"],
    "warfarin": ["coumadin"],
    
    # 常用缩写
    "cvd": ["cardiovascular disease"],
    "chf": ["congestive heart failure", "heart failure"],
    "copd": ["chronic obstructive pulmonary disease"],
    "pci": ["percutaneous coronary intervention"],
    "cabg": ["coronary artery bypass grafting"],
    "ecg": ["electrocardiogram"],
    "eeg": ["electroencephalogram"],
    "mri": ["magnetic resonance imaging"],
    "ct": ["computed tomography"],
    "pet": ["positron emission tomography"],
    "biopsy": ["tissue sample"],
    "stat": ["immediately"],
    "prn": ["as needed"],
}


# ============================================================
# 2. 医学实体识别模式（支持自定义扩展）
# ============================================================
MEDICAL_PATTERNS = {
    'drug': r'\b(aspirin|metformin|atorvastatin|warfarin|insulin|clopidogrel|heparin|enoxaparin|furosemide|lisinopril|amlodipine|simvastatin|rosuvastatin|glipizide|pioglitazone|sitagliptin|empagliflozin|dapagliflozin)\b',
    'disease': r'\b(myocardial infarction|heart attack|stroke|diabetes|hypertension|heart failure|arrhythmia|angina|atherosclerosis|thrombosis|embolism|pneumonia|sepsis|cancer|tumor|neoplasm|dementia|alzheimer|parkinson)\b',
    'procedure': r'\b(pci|cabg|stent|bypass|angioplasty|echocardiogram|catheterization|coronary angiography|ventilator|intubation|resuscitation|defibrillation)\b',
    'measurement': r'\b(blood pressure|heart rate|oxygen saturation|bmi|cholesterol|triglycerides|glucose|HbA1c|eGFR|creatinine)\b',
}


# ============================================================
# 3. 同义词扩展器
# ============================================================
class SynonymExpander:
    """将查询中的术语替换为同义词集合"""
    
    def __init__(self, synonyms: Dict[str, List[str]] = None):
        self.synonyms = synonyms or MEDICAL_SYNONYMS
        # 构建反向映射（变体 → 标准词），便于标准化
        self._build_reverse_map()
    
    def _build_reverse_map(self):
        """构建从所有变体到标准词（第一个定义）的映射"""
        self.variant_to_standard = {}
        for standard, variants in self.synonyms.items():
            for var in variants:
                self.variant_to_standard[var.lower()] = standard.lower()
    
    def expand(self, query: str) -> Dict[str, List[str]]:
        """
        返回每个术语及其同义词列表
        格式: {'original_term': ['syn1', 'syn2', ...]}
        """
        # 按词边界分割（保留单词，忽略标点）
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        expansion = {}
        for w in words:
            # 如果该词是某个标准词的同义词变体，映射到标准词
            standard = self.variant_to_standard.get(w, w)
            # 获取该标准词的所有同义词（包括自身）
            if standard in self.synonyms:
                expansion[w] = self.synonyms[standard] + [standard]
            else:
                # 如果该词没有同义词，保留原词
                expansion[w] = [w]
        return expansion
    
    def generate_query_variants(self, query: str) -> List[str]:
        """
        生成查询变体：将每个词替换为其同义词组合
        示例: "mi treatment" -> ["myocardial infarction treatment", "heart attack treatment", ...]
        """
        expansion = self.expand(query)
        # 获取每个位置的所有替换选项
        options = [opts for opts in expansion.values()]
        # 生成笛卡尔积（如果词汇量不大）
        import itertools
        variants = []
        for combo in itertools.product(*options):
            # 重建查询（保持原始词序，但替换为同义词）
            # 注意：这里我们简单按空格拼接，不保留原始介词等
            # 更好的做法是基于原始查询进行替换，但此处简化
            variant = " ".join(combo)
            variants.append(variant)
        # 去重并返回
        return list(set(variants))


# ============================================================
# 4. 实体识别器
# ============================================================
class MedicalEntityExtractor:
    """基于正则识别医学实体类型"""
    
    def __init__(self, patterns: Dict[str, str] = None):
        self.patterns = patterns or MEDICAL_PATTERNS
        self.compiled = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
    
    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        返回识别的实体，按类型分组
        例如: {'drug': ['metformin'], 'disease': ['myocardial infarction']}
        """
        entities = {}
        for typ, pattern in self.compiled.items():
            matches = pattern.findall(query)
            if matches:
                entities[typ] = list(set(matches))  # 去重
        return entities


# ============================================================
# 5. 过滤条件提取器（时间范围、期刊等）
# ============================================================
class FilterExtractor:
    """从查询中提取元数据过滤条件"""
    
    def __init__(self):
        # 年份提取模式
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        self.relative_time_patterns = [
            (r'(?:last|past|previous)\s+(\d+)\s+year', 'years_ago'),  # "last 5 years"
            (r'after\s+(\d{4})', 'after_year'),                       # "after 2015"
            (r'before\s+(\d{4})', 'before_year'),                     # "before 2020"
            (r'between\s+(\d{4})\s+and\s+(\d{4})', 'between_years'),  # "between 2010 and 2020"
        ]
        # 期刊过滤（简单示例）
        self.journal_pattern = re.compile(r'(?:in|from)\s+([A-Za-z\s]+Journal[A-Za-z\s]*|Nature|NEJM|Lancet|JAMA|BMJ)', re.IGNORECASE)
    
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        返回过滤条件字典，兼容ChromaDB的where条件
        示例: {"pub_year": {"$gte": 2020, "$lte": 2025}}
        """
        filters = {}
        
        # 1. 提取年份
        # 检查相对时间短语
        for pattern, typ in self.relative_time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if typ == 'years_ago':
                    years = int(match.group(1))
                    current_year = datetime.now().year
                    start_year = current_year - years
                    filters['pub_year'] = {"$gte": start_year, "$lte": current_year}
                elif typ == 'after_year':
                    year = int(match.group(1))
                    filters['pub_year'] = {"$gte": year}
                elif typ == 'before_year':
                    year = int(match.group(1))
                    filters['pub_year'] = {"$lte": year}
                elif typ == 'between_years':
                    start = int(match.group(1))
                    end = int(match.group(2))
                    filters['pub_year'] = {"$gte": start, "$lte": end}
                break  # 只处理第一个匹配
        
        # 如果未匹配相对时间，尝试直接提取绝对年份
        if 'pub_year' not in filters:
            years = [int(y) for y in self.year_pattern.findall(query)]
            if years:
                if len(years) == 1:
                    filters['pub_year'] = {"$eq": years[0]}
                elif len(years) > 1:
                    filters['pub_year'] = {"$gte": min(years), "$lte": max(years)}
        
        # 2. 提取期刊（简单的正则）
        journal_match = self.journal_pattern.search(query)
        if journal_match:
            journal = journal_match.group(1).strip()
            filters['journal'] = {"$eq": journal}
        
        return filters


# ============================================================
# 6. 主查询处理器
# ============================================================
class QueryProcessor:
    """
    主查询处理器：整合同义词扩展、实体识别、过滤提取，
    输出增强的查询对象
    """
    
    def __init__(self, 
                 synonym_dict: Dict[str, List[str]] = None,
                 entity_patterns: Dict[str, str] = None):
        self.syn_expander = SynonymExpander(synonym_dict)
        self.entity_extractor = MedicalEntityExtractor(entity_patterns)
        self.filter_extractor = FilterExtractor()
    
    def process(self, raw_query: str) -> Dict[str, Any]:
        """
        处理原始查询，返回增强查询对象
        
        返回格式:
        {
            "original_query": str,
            "cleaned_query": str,          # 基础清洗后的查询
            "vector_query": str,           # 用于向量检索（已加指令）
            "keyword_query": str,          # 用于关键词检索
            "query_variants": List[str],   # 同义词扩展的变体
            "entities": Dict[str, List[str]],
            "filters": Dict,               # 元数据过滤条件
            "synonyms_expanded": Dict[str, List[str]]  # 每个词的扩展
        }
        """
        # 1. 基础清洗（去除多余空格，转小写）
        cleaned = " ".join(raw_query.strip().split())
        
        # 2. 实体识别
        entities = self.entity_extractor.extract(cleaned)
        
        # 3. 同义词扩展
        synonyms_expanded = self.syn_expander.expand(cleaned)
        # 生成查询变体
        query_variants = self.syn_expander.generate_query_variants(cleaned)
        
        # 4. 向量查询（BGE 最佳实践：添加指令前缀）
        vector_query = f"Represent this question for searching relevant passages: {cleaned}"
        
        # 5. 关键词查询（保留原始文本，可用于 BM25 等）
        keyword_query = cleaned
        
        # 6. 提取过滤条件
        filters = self.filter_extractor.extract_filters(cleaned)
        
        # 7. 构建输出
        result = {
            "original_query": raw_query,
            "cleaned_query": cleaned,
            "vector_query": vector_query,
            "keyword_query": keyword_query,
            "query_variants": query_variants[:10],  # 限制变体数量，避免过多
            "entities": entities,
            "filters": filters,
            "synonyms_expanded": synonyms_expanded
        }
        
        logger.info(f"处理查询: {raw_query[:50]}...")
        logger.info(f"识别实体: {entities}")
        logger.info(f"提取过滤条件: {filters}")
        
        return result


# ============================================================
# 7. 快速测试（可独立运行）
# ============================================================
if __name__ == "__main__":
    # 创建处理器
    processor = QueryProcessor()
    
    # 测试查询
    test_queries = [
        "二甲双胍对心血管疾病有何影响？",
        "mi treatment in last 5 years",
        "aspirin for heart attack after 2015",
        "hypertension and diabetes management in NEJM"
    ]
    
    for q in test_queries:
        print("\n" + "="*60)
        print(f"原始查询: {q}")
        result = processor.process(q)
        print(f"向量查询: {result['vector_query']}")
        print(f"同义词变体: {result['query_variants'][:3]}...")
        print(f"实体: {result['entities']}")
        print(f"过滤条件: {result['filters']}")