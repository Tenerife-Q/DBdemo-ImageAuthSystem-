"""
多模态虚假信息检测模型
融合：BERT（文本） + ResNet50（图片）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch. utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

print("="*70)
print("🤖 多模态虚假信息检测模型")
print("="*70)

# ==================== 第1部分：检查GPU ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ 使用设备:  {device}")
if device.type == 'cuda': 
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")

# ==================== 第2部分：数据集类 ====================
class TextImageDataset(Dataset):
    """文本+图片数据集"""
    
    def __init__(self, csv_file, max_samples=None):
        """
        参数: 
        - csv_file: CSV文件路径
        - max_samples: 限制样本数（测试用）
        """
        self. df = pd.read_csv(csv_file)
        
        if max_samples:
            self.df = self.df. head(max_samples)
        
        print(f"\n✓ 加载数据集: {csv_file}")
        print(f"  样本数: {len(self. df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 文本特征（简化版：只用长度和词数）
        title_len = float(row. get('title_length', 0))
        text_len = float(row.get('text_length', 0))
        score = float(row.get('score', 0))
        comments = float(row.get('comments', 0))
        
        # 组合成文本特征向量（4维）
        text_features = torch.tensor([title_len, text_len, score, comments], dtype=torch.float32)
        
        # 标签
        label = torch.tensor(row. get('label', 0), dtype=torch.long)
        
        return text_features, label

# ==================== 第3部分：模型架构 ====================
class MultimodalFakeNewsDetector(nn.Module):
    """多模态虚假信息检测器"""
    
    def __init__(self, text_feature_dim=4, hidden_dim=128):
        """
        参数:
        - text_feature_dim: 文本特征维度
        - hidden_dim: 隐藏层维度
        """
        super(MultimodalFakeNewsDetector, self).__init__()
        
        print("\n🏗️  构建模型架构...")
        
        # 文本处理分支
        self.text_branch = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # 图片特征模拟分支（在真实项目中会用ResNet50）
        self.image_branch = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 假设图片特征是10维
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn. Sequential(
            nn.Linear(128, 64),  # 64+64=128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 分类层
        self.classifier = nn. Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # 二分类：真/假
        )
        
        print("✓ 模型构建完成")
        print(f"  参数数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, text_features, image_features=None):
        """
        前向传播
        """
        # 文本处理
        text_out = self.text_branch(text_features)
        
        # 图片处理（如果没有图片，用随机向量）
        if image_features is None:
            image_features = torch.randn(text_features.size(0), 10, device=text_features.device)
        
        image_out = self.image_branch(image_features)
        
        # 融合
        combined = torch.cat([text_out, image_out], dim=1)
        fused = self.fusion(combined)
        
        # 分类
        output = self.classifier(fused)
        
        return output

# ==================== 第4部分：训练器 ====================
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n⚙️  训练配置:")
        print(f"  优化器: Adam")
        print(f"  学习率: {lr}")
        print(f"  损失函数: CrossEntropyLoss")
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (text_features, labels) in enumerate(dataloader):
            text_features = text_features.to(self. device)
            labels = labels. to(self.device)
            
            # 前向传播
            outputs = self.model(text_features)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs. data, 1)
            correct += (predicted == labels).sum().item()
            total += labels. size(0)
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_features, labels in dataloader:
                text_features = text_features. to(self.device)
                labels = labels.to(self. device)
                
                outputs = self.model(text_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy

# ==================== 第5部分：主程序 ====================
def main():
    print("\n" + "="*70)
    print("🚀 开始训练多模态模型")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/5] 加载训练数据...")
    train_dataset = TextImageDataset('model_data/train_set.csv', max_samples=100)
    val_dataset = TextImageDataset('model_data/val_set.csv', max_samples=30)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ 数据加载完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 2. 创建模型
    print("\n[2/5] 创建模型...")
    model = MultimodalFakeNewsDetector(text_feature_dim=4, hidden_dim=128)
    
    # 3. 创建训练器
    print("\n[3/5] 初始化训练器...")
    trainer = ModelTrainer(model, device, lr=1e-3)
    
    # 4. 训练循环
    print("\n[4/5] 开始训练...")
    print("="*70)
    
    num_epochs = 5
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss':  [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # 验证
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 显示
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  训练:  Loss={train_loss:.4f} Acc={train_acc:.2f}%")
        print(f"  验证: Loss={val_loss:.4f} Acc={val_acc:.2f}%")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_data/best_model.pth')
            print(f"  ✓ 保存最佳模型")
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    
    # 5. 保存结果
    print("\n[5/5] 保存模型和历史...")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'model_data/final_model.pth')
    
    # 保存训练历史
    with open('model_data/training_history.json', 'w') as f:
        # 转换为列表便于JSON序列化
        history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_serializable, f, indent=2)
    
    print(f"✓ 模型已保存:  model_data/best_model.pth")
    print(f"✓ 训练历史已保存:  model_data/training_history.json")
    
    # 显示最终结果
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
    
    print("\n" + "="*70)
    print("🎉 模型训练完成!")
    print("="*70)

if __name__ == "__main__": 
    main()