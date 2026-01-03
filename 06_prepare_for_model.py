"""
为模型训练准备数据
生成：训练集、验证集、测试集
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class ModelDataPreparation:
    """为模型准备数据"""
    
    def __init__(self, csv_file='processed_data/reddit_posts_cleaned.csv'):
        self.csv_file = csv_file
        self.df = None
        self.output_dir = 'model_data'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ 初始化完成:  {self.output_dir}/")
    
    def load_processed_data(self):
        """加载预处理的数据"""
        print("\n" + "="*70)
        print("📂 第1步：加载预处理数据")
        print("="*70)
        
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ 加载成功: {len(self.df)} 行数据")
    
    def split_data(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """将数据分为训练集、验证集、测试集"""
        print("\n" + "="*70)
        print("✂️  第2步：划分数据集")
        print("="*70)
        
        print(f"划分比例: 训练{train_size*100:.0f}% | 验证{val_size*100:.0f}% | 测试{test_size*100:.0f}%")
        
        # 第一次分割：分离测试集
        train_val, test = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=42,
            stratify=self.df['label']  # 保持标签比例
        )
        
        # 第二次分割：分离训练集和验证集
        train, val = train_test_split(
            train_val,
            test_size=val_size/(train_size+val_size),
            random_state=42,
            stratify=train_val['label']
        )
        
        print(f"✅ 划分完成:")
        print(f"  训练集: {len(train)} 样本 ({len(train)/len(self.df)*100:.1f}%)")
        print(f"  验证集: {len(val)} 样本 ({len(val)/len(self.df)*100:.1f}%)")
        print(f"  测试集: {len(test)} 样本 ({len(test)/len(self.df)*100:.1f}%)")
        
        # 显示标签分布
        print(f"\n标签分布:")
        for name, data in [('训练', train), ('验证', val), ('测试', test)]:
            label_0 = len(data[data['label']==0])
            label_1 = len(data[data['label']==1])
            print(f"  {name}集: 真实{label_1} | 其他{label_0}")
        
        return train, val, test
    
    def prepare_text_features(self, df):
        """提取文本特征"""
        features = pd.DataFrame()
        
        features['title_length'] = df['title_clean'].str.len()
        features['text_length'] = df['text_clean'].str.len()
        features['title_word_count'] = df['title_clean'].str.split().str.len()
        features['text_word_count'] = df['text_clean'].str.split().str.len()
        
        # 社交特征
        features['score'] = df['score']
        features['comments'] = df['comments']
        
        # 标准化（将数据缩放到相同范围）
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features = pd.DataFrame(features_scaled, columns=features.columns)
        
        return features
    
    def save_datasets(self, train, val, test):
        """保存数据集"""
        print("\n" + "="*70)
        print("💾 第3步：保存数据集")
        print("="*70)
        
        # 保存为CSV
        for name, data in [('train', train), ('val', val), ('test', test)]:
            path = os.path.join(self.output_dir, f'{name}_set.csv')
            data.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ 保存:  {name}_set.csv ({len(data)} 样本)")
        
        # 保存标签
        train_labels = train['label'].values
        val_labels = val['label'].values
        test_labels = test['label'].values
        
        np.save(os.path.join(self.output_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(self.output_dir, 'val_labels.npy'), val_labels)
        np.save(os.path.join(self.output_dir, 'test_labels.npy'), test_labels)
        
        print(f"✅ 保存标签: train/val/test_labels.npy")
    
    def create_metadata(self, train, val, test):
        """创建元数据文件"""
        metadata = {
            'created_at': datetime.now(). isoformat(),
            'total_samples': len(self.df),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples':  len(test),
            'features': [
                'title_clean',
                'text_clean',
                'score',
                'comments',
                'label'
            ],
            'label_mapping': {
                '0': '其他',
                '1': '真实新闻'
            },
            'class_distribution': {
                'train': {
                    'label_0': int(len(train[train['label']==0])),
                    'label_1':  int(len(train[train['label']==1]))
                },
                'val': {
                    'label_0': int(len(val[val['label']==0])),
                    'label_1': int(len(val[val['label']==1]))
                },
                'test': {
                    'label_0': int(len(test[test['label']==0])),
                    'label_1':  int(len(test[test['label']==1]))
                }
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存元数据: metadata.json")
    
    def display_statistics(self, train, val, test):
        """显示统计信息"""
        print("\n" + "="*70)
        print("📊 第4步：数据统计")
        print("="*70)
        
        print(f"\n样本统计:")
        print(f"  训练集: {len(train)}")
        print(f"  验证集: {len(val)}")
        print(f"  测试集: {len(test)}")
        
        print(f"\n标题统计 (字数):")
        for name, data in [('训练', train), ('验证', val), ('测试', test)]:
            print(f"  {name}集:")
            print(f"    平均:  {data['title_length'].mean():.0f}")
            print(f"    最大: {data['title_length'].max()}")
            print(f"    最小: {data['title_length'].min()}")
        
        print(f"\n文本统计 (字数):")
        for name, data in [('训练', train), ('验证', val), ('测试', test)]:
            print(f"  {name}集:")
            print(f"    平均: {data['text_length'].mean():.0f}")
            print(f"    最大: {data['text_length']. max()}")
            print(f"    最小: {data['text_length'].min()}")
    
    def run(self):
        """执行完整流程"""
        print("\n🚀 为模型训练准备数据.. .\n")
        
        self.load_processed_data()
        train, val, test = self.split_data()
        self.save_datasets(train, val, test)
        self.create_metadata(train, val, test)
        self.display_statistics(train, val, test)
        
        print("\n" + "="*70)
        print("✅ 数据准备完成!")
        print("="*70)
        print(f"\n输出文件位置: {self.output_dir}/")
        print(f"  ├── train_set.csv")
        print(f"  ├── val_set.csv")
        print(f"  ├── test_set.csv")
        print(f"  ├── train_labels.npy")
        print(f"  ├── val_labels.npy")
        print(f"  ├── test_labels.npy")
        print(f"  └── metadata.json")
        print(f"\n现在可以用这些数据训练模型了!")

# ==================== 运行 ====================
if __name__ == "__main__":
    prep = ModelDataPreparation('processed_data/reddit_posts_cleaned.csv')
    prep.run()