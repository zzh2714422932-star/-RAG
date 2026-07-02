import glob
import os

data_dir = os.path.expanduser("~/med_rag_data/pubmed_oa/")   # 改成你的实际父目录
files = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
print(f"找到 {len(files)} 个 .xml 文件")
for f in files[:5]:
    print(f)