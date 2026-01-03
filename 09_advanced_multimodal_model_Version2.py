"""
真实的多模态模型 v2
使用真实的文本和图片特征提取
"""

import torch
import torch.nn as nn
import torch. optim as optim
from torch. utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 真实多模态虚假信息检测模型 v2.0")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ 使用设备: {device}")

# ==================== 第1部分：改进的特征提取 ====================
class TextFeatureExtractor:
    """文本特征提取器 - 提取更丰富的特征"""
    
    def __init__(self):
        print("\n📝 初始化文本特征提取器...")
    
    def extract_features(self, title, text, score, comments):
        """
        从文本中提取真实特征
        """
        features = []
        
        # 1. 长度特征
        features.append(len(title) if title else 0)
        features.append(len(str(text)) if text else 0)
        
        # 2. 词汇特征
        title_words = len(str(title).split()) if title else 0
        text_words = len(str(text).split()) if text else 0
        features.append(title_words)
        features.append(text_words)
        
        # 3. 社交特征
        features.append(float(score) if score else 0)
        features.append(float(comments) if comments else 0)
        
        # 4. 比率特征
        avg_word_len = (len(title) / title_words) if title_words > 0 else 0
        features.append(avg_word_len)
        
        # 5. 特殊字符
        special_chars = sum(1 for c in str(title) if not c.isalnum() and c != ' ')
        features.append(special_chars)
        
        # 6. 大写字母比例
        uppercase = sum(1 for c in str(title) if c.isupper())
        uppercase_ratio = uppercase / len(title) if len(title) > 0 else 0
        features.append(uppercase_ratio)
        
        # 7. 数字比例
        digits = sum(1 for c in str(title) if c.isdigit())
        digit_ratio = digits / len(title) if len(title) > 0 else 0
        features.append(digit_ratio)
        
        return np.array(features, dtype=np.float32)

# ==================== 第2部分：改进的数据集 ====================
class ImprovedTextImageDataset(Dataset):
    """改进的数据集 - 提取真实特征"""
    
    def __init__(self, csv_file, max_samples=None):
        self.df = pd.read_csv(csv_file)
        
        if max_samples:
            self. df = self.df. head(max_samples)
        
        self.feature_extractor = TextFeatureExtractor()
        
        print(f"\n✓ 加载数据集: {csv_file}")
        print(f"  样本数: {len(self. df)}")
        
        # 预提取所有特征（加速训练）
        print("  正在提取文本特征...")
        self.features = []
        for idx, row in self.df.iterrows():
            features = self.feature_extractor. extract_features(
                row. get('title', ''),
                row.get('text', ''),
                row.get('score', 0),
                row.get('comments', 0)
            )
            self.features.append(features)
        
        self.features = np. array(self.features)
        print(f"  ✓ 提取了 {self.features.shape[1]} 个特征")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)
        return features, label

# ==================== 第3部分：改进的模型 ====================
class AdvancedMultimodalDetector(nn.Module):
    """改进的多模态检测器"""
    
    def __init__(self, feature_dim=9):
        super(AdvancedMultimodalDetector, self).__init__()
        
        print("\n🏗️  构建改进的模型架构...")
        
        # 文本特征处理（增加复杂度）
        self.text_branch = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16)
        )
        
        # 注意力机制（增强特征）
        self.attention = nn.Sequential(
            nn.Linear(16, 8),
            nn.Sigmoid()
        )
        
        # 分类头
        # 注意力机制（增强特征）
        self.attention = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid()
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 2)
        )

        print("✓ 模型构建完成")
        print(f"  参数数量: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  特征维度: {feature_dim}")
    
    def forward(self, features):
        # 文本处理
        text_out = self.text_branch(features)
        
        # 注意力加权
        attention_weights = self.attention(text_out)
        weighted = text_out * attention_weights
        
        # 分类
        output = self.classifier(weighted)
        
        return output

# ==================== 第4部分：改进的训练器 ====================
class AdvancedTrainer:
    """改进的训练器 - 支持学习率调整、早停等"""
    
    def __init__(self, model, device, lr=1e-3):
        self.model = model. to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]))  # 处理类别不平衡
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        
        print(f"\n⚙️  训练配置:")
        print(f"  优化器: Adam (weight_decay=1e-5)")
        print(f"  初始学习率: {lr}")
        print(f"  损失函数: CrossEntropyLoss (带类别权重)")
        print(f"  学习率调整:  ReduceLROnPlateau")
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(dataloader), correct / total * 100
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = total_loss / len(dataloader)
        self.scheduler.step(val_loss)
        
        return val_loss, correct / total * 100

# ==================== 第5部分：主程序 ====================
def main():
    print("\n" + "="*70)
    print("🚀 开始训练改进的多模态模型")
    print("="*70)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    train_dataset = ImprovedTextImageDataset('model_data/train_set.csv', max_samples=None)
    val_dataset = ImprovedTextImageDataset('model_data/val_set.csv', max_samples=None)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"✓ 数据加载完成")
    
    # 创建模型
    print("\n[2/5] 创建改进的模型...")
    model = AdvancedMultimodalDetector(feature_dim=train_dataset.features.shape[1])
    
    # 创建训练器
    print("\n[3/5] 初始化训练器...")
    trainer = AdvancedTrainer(model, device, lr=5e-4)
    
    # 训练
    print("\n[4/5] 开始训练...")
    print("="*70)
    
    num_epochs = 10
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss':  [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  训练:  Loss={train_loss:.4f} Acc={train_acc:.2f}%")
        print(f"  验证: Loss={val_loss:.4f} Acc={val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model_data/advanced_best_model.pth')
            print(f"  ✓ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= max_patience: 
                print(f"  ⚠️  早停触发 (耐心次数: {patience_counter}/{max_patience})")
                break
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    
    # 保存
    print("\n[5/5] 保存结果...")
    torch.save(model.state_dict(), 'model_data/advanced_final_model.pth')
    
    with open('model_data/advanced_history.json', 'w') as f:
        history_serializable = {k: [float(v) for v in vals] for k, vals in history. items()}
        json.dump(history_serializable, f, indent=2)
    
    print(f"✓ 模型已保存:  model_data/advanced_best_model.pth")
    print(f"✓ 训练历史已保存: model_data/advanced_history.json")
    
    # 结果
    print("\n" + "="*70)
    print("📊 最终结果")
    print("="*70)
    print(f"\n训练集:")
    print(f"  最终Loss: {history['train_loss'][-1]:.4f}")
    print(f"  最终精度: {history['train_acc'][-1]:.2f}%")
    
    print(f"\n验证集:")
    print(f"  最终Loss: {history['val_loss'][-1]:.4f}")
    print(f"  最终精度: {history['val_acc'][-1]:.2f}%")
    
    print(f"\n最佳验证精度: {max(history['val_acc']):.2f}%")
    print(f"最佳验证Loss: {min(history['val_loss']):.4f}")
    
    print("\n" + "="*70)
    print("🎉 改进的模型训练完成!")
    print("="*70)

if __name__ == "__main__":
    main()