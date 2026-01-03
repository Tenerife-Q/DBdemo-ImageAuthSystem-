"""
数据清洗和预处理 - 为模型训练做准备
目标：将爬虫数据转换成模型可用的格式
"""

import pandas as pd
import os
import json
from datetime import datetime
import re
from collections import Counter

class DataPreprocessor: 
    """数据预处理器"""
    
    def __init__(self, csv_file='reddit_posts.csv'):
        self.csv_file = csv_file
        self.df = None
        self.output_dir = 'processed_data'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ 初始化完成: {self.output_dir}/")
    
    def load_data(self):
        """加载CSV数据"""
        print("\n" + "="*70)
        print("📂 第1步：加载数据")
        print("="*70)
        
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ 加载成功")
        print(f"   行数:  {len(self.df)}")
        print(f"   列数: {len(self.df.columns)}")
        print(f"   列名: {list(self.df.columns)}")
    
    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 移除特殊字符和链接
        text = re.sub(r'http\S+|www\S+', '', text)  # 移除URL
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 保留只有字母、数字、空格
        
        # 移除多余空格
        text = ' '.  join(text.split())
        
        # 转小写
        text = text.lower()
        
        return text
    
    def preprocess_text(self):
        """清洗所有文本"""
        print("\n" + "="*70)
        print("✂️  第2步：清洗文本数据")
        print("="*70)
        
        print("正在清洗标题...")
        self.df['title_clean'] = self.df['title'].apply(self.clean_text)
        
        print("正在清洗文本内容...")
        self.df['text_clean'] = self.df['text'].apply(self.clean_text)
        
        # 计算文本长度
        self.df['title_length'] = self.df['title_clean'].str.len()
        self.df['text_length'] = self. df['text_clean'].  str.len()
        
        print(f"✅ 清洗完成")
        print(f"   平均标题长度: {self.df['title_length'].mean():.0f} 字")
        print(f"   平均文本长度: {self.df['text_length'].mean():.0f} 字")
    
    def create_labels(self):
        """创建标签（真/假分类）"""
        print("\n" + "="*70)
        print("🏷️  第3步：创建标签")
        print("="*70)
        
        print("基于数据源创建标签...")
        
        # 简单的标签规则（实际项目会更复杂）
        def assign_label(row):
            # nottheonion 是"不是洋葱新闻"，通常是真实但荒谬的新闻
            # news 和 worldnews 一般是真实新闻
            # 我们这里简化处理
            
            subreddit = row['subreddit']. lower()
            
            if subreddit == 'nottheonion': 
                return 1  # 真实（但需要验证）
            elif subreddit in ['news', 'worldnews']:
                return 1  # 真实新闻
            else:
                return 0  # 未知
        
        self.df['label'] = self.df. apply(assign_label, axis=1)
        
        # 标签统计
        label_counts = self.df['label'].value_counts()
        print(f"✅ 标签创建完成")
        print(f"   真实新闻: {label_counts. get(1, 0)} 个")
        print(f"   其他:  {label_counts.get(0, 0)} 个")
    
    def filter_valid_data(self):
        """过滤有效数据"""
        print("\n" + "="*70)
        print("🔍 第4步：过滤有效数据")
        print("="*70)
        
        print("过滤前:")
        print(f"  总数: {len(self.df)}")
        
        # 过滤掉没有文本或标题的
        self.df = self.df[self.df['title_length'] > 5]
        
        print("过滤后:")
        print(f"  总数: {len(self.df)}")
        print(f"✅ 过滤完成")
    
    def analyze_cleaned_data(self):
        """分析清洗后的数据"""
        print("\n" + "="*70)
        print("📊 第5步：数据分析")
        print("="*70)
        
        print(f"\n文本统计:")
        print(f"  标题平均长度: {self.df['title_length'].mean():.0f} 字")
        print(f"  文本平均长度: {self.df['text_length'].mean():.0f} 字")
        print(f"  最长标题: {self.df['title_length'].max()} 字")
        print(f"  最长文本:  {self.df['text_length'].max()} 字")
        
        print(f"\n赞数统计:")
        print(f"  平均赞数: {self.df['score'].mean():.0f}")
        print(f"  中位赞数: {self.df['score'].median():.0f}")
        print(f"  最高赞数: {self. df['score'].max()}")
        
        print(f"\n图片分布:")
        has_image = len(self.df[self.df['image_url'].notna() & (self.df['image_url']. str.len() > 0)])
        print(f"  有图片: {has_image} 个 ({has_image/len(self.df)*100:.1f}%)")
        print(f"  无图片: {len(self. df) - has_image} 个 ({(len(self.df)-has_image)/len(self.df)*100:.1f}%)")
        
        print(f"\n关键词分析 (标题中最常见的词):")
        all_words = ' '.join(self.df['title_clean']).split()
        word_counts = Counter(all_words)
        
        # 移除常用词
        stopwords = {'the', 'a', 'and', 'or', 'of', 'in', 'to', 'is', 'that', 'for', 'on'}
        common_words = [word for word, count in word_counts.most_common(10) if word not in stopwords]
        
        for i, word in enumerate(common_words[: 5], 1):
            count = word_counts[word]
            print(f"  {i}. {word}:  {count} 次")
    
    def save_processed_data(self):
        """保存处理后的数据"""
        print("\n" + "="*70)
        print("💾 第6步：保存处理后的数据")
        print("="*70)
        
        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'reddit_posts_cleaned.csv')
        self.df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV已保存:  {csv_path}")
        
        # 保存为JSON（包含所有信息）
        json_path = os.path.join(self. output_dir, 'reddit_posts_cleaned.json')
        self.df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"✅ JSON已保存: {json_path}")
        
        # 创建统计报告
        report_path = os.path.join(self.output_dir, 'data_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Reddit数据预处理报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f. write("数据统计:\n")
            f.write(f"  总样本数: {len(self. df)}\n")
            f.write(f"  真实新闻: {len(self.df[self.df['label']==1])}\n")
            f.write(f"  其他: {len(self.df[self.df['label']==0])}\n\n")
            
            f.write("文本统计:\n")
            f.write(f"  标题平均长度: {self.df['title_length'].mean():.0f} 字\n")
            f.write(f"  文本平均长度: {self.df['text_length'].mean():.0f} 字\n\n")
            
            f. write("社交指标:\n")
            f.write(f"  平均赞数: {self. df['score'].mean():.0f}\n")
            f.write(f"  平均评论:  {self.df['comments'].mean():.0f}\n\n")
            
            f. write("数据质量:\n")
            f.write(f"  包含图片: {len(self.df[self.df['image_url'].notna()]) / len(self.df) * 100:.1f}%\n")
            f.write(f"  有效行: {len(self.df)}\n")
        
        print(f"✅ 报告已保存: {report_path}")
    
    def display_samples(self, num_samples=3):
        """显示清洗后的样本"""
        print("\n" + "="*70)
        print(f"📋 样本数据 (前{num_samples}个)")
        print("="*70)
        
        for idx, row in self.df.head(num_samples).iterrows():
            print(f"\n[样本 {idx+1}]")
            print(f"  标题: {row['title'][:   60]}")
            print(f"  清洗后: {row['title_clean'][:  60]}")
            print(f"  标签: {'真实' if row['label'] == 1 else '其他'} ({row['label']})")
            print(f"  赞:  {row['score']} | 评论: {row['comments']}")
            print(f"  文本长度: {row['text_length']}")
    
    def run(self):
        """执行完整的预处理流程"""
        print("\n🚀 开始数据预处理.. .\n")
        
        self.load_data()
        self.preprocess_text()
        self.create_labels()
        self.filter_valid_data()
        self.analyze_cleaned_data()
        self.display_samples()
        self.save_processed_data()
        
        print("\n" + "="*70)
        print("✅ 数据预处理完成!")
        print("="*70)
        print(f"\n输出文件位置: {self.output_dir}/")
        print(f"  ├── reddit_posts_cleaned.csv  (可用Excel打开)")
        print(f"  ├── reddit_posts_cleaned.json (JSON格式)")
        print(f"  └── data_report. txt (统计报告)")

# ==================== 运行 ====================
if __name__ == "__main__":
    preprocessor = DataPreprocessor('reddit_posts.csv')
    preprocessor.run()