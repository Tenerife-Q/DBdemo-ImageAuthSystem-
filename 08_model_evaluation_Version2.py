"""
模型评估和预测
"""

import torch
import torch. nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 导入模型类 ====================
class MultimodalFakeNewsDetector(nn. Module):
    """多模态虚假信息检测器"""
    
    def __init__(self, text_feature_dim=4, hidden_dim=128):
        super(MultimodalFakeNewsDetector, self).__init__()
        
        self.text_branch = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        self.image_branch = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, text_features, image_features=None):
        text_out = self.text_branch(text_features)
        
        if image_features is None:
            image_features = torch.randn(text_features.size(0), 10, device=text_features.device)
        
        image_out = self.image_branch(image_features)
        combined = torch.cat([text_out, image_out], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output

# ==================== 评估器 ====================
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path='model_data/best_model.pth'):
        print("="*70)
        print("📊 模型评估")
        print("="*70)
        
        # 加载模型
        self.model = MultimodalFakeNewsDetector()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"\n✓ 模型已加载: {model_path}")
    
    def evaluate_on_test_set(self, test_csv):
        """在测试集上评估"""
        print(f"\n[1/3] 加载测试集...")
        
        test_df = pd.read_csv(test_csv)
        print(f"✓ 测试集大小: {len(test_df)}")
        
        # 准备数据
        text_features = torch.tensor(
            test_df[['title_length', 'text_length', 'score', 'comments']].values,
            dtype=torch.float32
        ).to(device)
        labels = torch.tensor(test_df['label'].values, dtype=torch.long).to(device)
        
        # 预测
        print(f"\n[2/3] 进行预测...")
        with torch.no_grad():
            outputs = self.model(text_features)
            probabilities = torch. softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # 计算准确率
        accuracy = (predictions == labels).float().mean().item() * 100
        
        print(f"✓ 预测完成")
        print(f"\n[3/3] 计算指标...")
        
        # 详细指标
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 显示结果
        print("\n" + "="*70)
        print("📈 评估结果")
        print("="*70)
        
        print(f"\n准确率指标:")
        print(f"  准确率(Accuracy): {accuracy:.2f}%")
        print(f"  精确率(Precision): {precision:.2f}")
        print(f"  召回率(Recall): {recall:.2f}")
        print(f"  F1分数:  {f1:.2f}")
        
        print(f"\n混淆矩阵:")
        print(f"  TP(真正): {tp}")
        print(f"  TN(真负): {tn}")
        print(f"  FP(假正): {fp}")
        print(f"  FN(假负): {fn}")
        
        return {
            'accuracy': accuracy,
            'precision':  precision,
            'recall': recall,
            'f1':  f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def predict_samples(self, test_csv, num_samples=5):
        """对样本进行预测展示"""
        print(f"\n" + "="*70)
        print(f"📋 样本预测 (前{num_samples}个)")
        print("="*70)
        
        test_df = pd.read_csv(test_csv)
        
        for idx in range(min(num_samples, len(test_df))):
            row = test_df.iloc[idx]
            
            # 准备数据
            text_features = torch.tensor(
                [[row['title_length'], row['text_length'], row['score'], row['comments']]],
                dtype=torch. float32
            ).to(device)
            
            # 预测
            with torch.no_grad():
                output = self.model(text_features)
                prob = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(output, dim=1)[0].item()
            
            true_label = row['label']
            pred_label = '真实' if pred == 1 else '其他'
            true_label_text = '真实' if true_label == 1 else '其他'
            confidence = prob[pred].item() * 100
            
            print(f"\n[样本 {idx+1}]")
            print(f"  标题: {row['title'][: 60]}")
            print(f"  真实标签: {true_label_text}")
            print(f"  预测标签: {pred_label}")
            print(f"  置信度: {confidence:.2f}%")
            print(f"  正确: {'✓' if pred == true_label else '✗'}")

# ==================== 主程序 ====================
def main():
    # 1. 评估
    evaluator = ModelEvaluator('model_data/best_model.pth')
    results = evaluator.evaluate_on_test_set('model_data/test_set.csv')
    
    # 2. 显示样本预测
    evaluator. predict_samples('model_data/test_set.csv', num_samples=5)
    
    # 3. 保存评估结果
    print(f"\n" + "="*70)
    print("✅ 评估完成!")
    print("="*70)
    
    with open('model_data/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 评估结果已保存:  model_data/evaluation_results.json")

if __name__ == "__main__":
    main()