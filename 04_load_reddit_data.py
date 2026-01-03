"""
加载Reddit爬虫数据 - 为后续模型训练做准备
"""

import json
import os
import pandas as pd
from datetime import datetime

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir='reddit_data'):
        self.data_dir = data_dir
    
    def load_json_files(self):
        """加载所有JSON文件"""
        print("="*70)
        print("📂 加载Reddit数据")
        print("="*70)
        
        all_posts = []
        
        # 遍历data_dir中的所有JSON文件
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path. join(self.data_dir, filename)
                print(f"\n📖 读取: {filename}")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    posts = json.load(f)
                    all_posts.extend(posts)
                    print(f"   ✓ 加载 {len(posts)} 个帖子")
        
        print(f"\n✅ 总共加载:  {len(all_posts)} 个帖子")
        return all_posts
    
    def convert_to_dataframe(self, posts):
        """转换为Pandas DataFrame（便于分析）"""
        print("\n" + "="*70)
        print("📊 转换为数据表")
        print("="*70)
        
        df = pd.DataFrame(posts)
        
        print(f"\n数据表信息:")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")
        
        return df
    
    def analyze_data(self, df):
        """分析数据"""
        print("\n" + "="*70)
        print("📈 数据分析")
        print("="*70)
        
        print(f"\n基础统计:")
        print(f"  平均赞数: {df['score']. mean():.1f}")
        print(f"  平均评论:  {df['comments'].mean():.1f}")
        print(f"  最高赞数: {df['score'].max()}")
        print(f"  最低赞数: {df['score'].min()}")
        
        print(f"\n文本分析:")
        print(f"  有文本的帖子: {len(df[df['text'].str.len() > 0])} 个")
        print(f"  有图片的帖子: {len(df[df['image_url'].str.len() > 0])} 个")
        print(f"  平均文本长度: {df['text'].str.len().mean():.0f} 字")
        
        print(f"\n子版块分布:")
        print(df['subreddit'].value_counts())
        
        return df
    
    def save_to_csv(self, df, output_file='reddit_posts.csv'):
        """保存为CSV（便于Excel打开）"""
        print(f"\n💾 保存为CSV: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存")
    
    def display_samples(self, df, num_samples=3):
        """显示样本数据"""
        print("\n" + "="*70)
        print(f"📋 数据样本 (前{num_samples}个)")
        print("="*70)
        
        for idx, row in df.head(num_samples).iterrows():
            print(f"\n[{idx+1}]")
            print(f"  标题: {row['title'][:  60]}")
            print(f"  作者: {row['author']}")
            print(f"  赞:  {row['score']} | 评论: {row['comments']}")
            print(f"  文本: {row['text'][: 80]}...")
            print(f"  图片: {'有' if row['image_url'] else '无'}")

# ==================== 运行 ====================
if __name__ == "__main__": 
    loader = DataLoader('reddit_data')
    
    # 1. 加载JSON文件
    posts = loader.load_json_files()
    
    # 2. 转换为DataFrame
    df = loader.convert_to_dataframe(posts)
    
    # 3. 分析数据
    df = loader.analyze_data(df)
    
    # 4. 显示样本
    loader.display_samples(df, num_samples=3)
    
    # 5. 保存为CSV
    loader.save_to_csv(df, 'reddit_posts.csv')
    
    print("\n" + "="*70)
    print("✅ 数据加载完成!")
    print("="*70)
    print("\n可以用Excel打开 reddit_posts.csv 查看数据")