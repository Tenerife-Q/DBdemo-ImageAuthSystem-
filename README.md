\# 多模态虚假信息检测系统



\## 项目概述



这是一个基于深度学习的\*\*多模态虚假信息检测系统\*\*，融合了文本和图片特征，用于识别社交媒体中的虚假新闻。



\## 功能特点



✨ \*\*Web爬虫\*\*

\- Reddit数据爬虫

\- Wikipedia爬虫



✨ \*\*数据处理\*\*

\- 自动数据清洗

\- 特征提取（9维）

\- 数据集划分



✨ \*\*深度学习模型\*\*

\- 多模态融合架构

\- 注意力机制

\- 早停防过拟合



\## 快速开始



\### 1. 环境配置



```bash

\# 创建虚拟环境

python -m venv venv



\# 激活虚拟环境（Windows）

.\\venv\\Scripts\\activate



\# 安装依赖

pip install -r requirements.txt

# 基于多模态与区块链的图像可信鉴真系统

[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-100%25-blue)

> **Image Trust & Authentication System based on Blockchain**  
> 一个结合了多模态特征提取、区块链存证和语义检索的图像真实性验证系统

---

## 📖 项目简介

本系统旨在解决**历史影像真实性验证**难题，通过以下技术手段实现图像的可信鉴真：

- **双重指纹技术**：SHA256（物理层防篡改） + pHash（感知层防伪造）
- **区块链存证**：模拟区块链锚定机制，确保数据不可篡改
- **语义事实分离**：将图像元数据与业务逻辑解耦，符合数据库第三范式（3NF）
- **全文检索**：支持基于时间、地点、事件描述的模糊查询

**适用场景**：新闻机构影像存档、司法取证、文物数字化保护等

---

## 🎯 核心功能

### 1. **数据摄取与存证（ETL Pipeline）**
- 从数据源（如新华社多媒体库）抓取影像
- 自动计算双重哈希指纹
- 事务性入库，支持查重与回滚

### 2. **命令行交互界面**
运行 `main.py` 可进入 Rich 风格的终端界面：
```bash
python main.py
```
- 自动导入模拟数据（1972年尼克松访华、1896年沈钧儒家庭照）
- 实时查询与展示

### 3. **Web 可视化界面**
运行 `web_app.py` 启动 Streamlit 应用：
```bash
streamlit run web_app.py
```
- 可视化检索界面
- 支持图片与元数据联合展示
- 区块链状态实时追踪

---

## 🛠️ 技术架构

### 数据库设计（符合 3NF 范式）

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  data_source    │──────▶│ raw_crawl_buffer │       │   image_core    │
│  (数据源表)      │       │  (原始数据缓冲)   │       │  (核心资产表)    │
└─────────────────┘       └──────────────────┘       └────────┬────────┘
                                                               │
                          ┌──────────────────┐                │
                          │ semantic_facts   │◀───────────────┤
                          │  (语义事实表)     │                │
                          └──────────────────┘                │
                                                               │
                          ┌──────────────────┐                │
                          │  chain_anchor    │◀───────────────┘
                          │  (区块链锚定表)   │
                          └──────────────────┘
```

### 文件结构

```
DBdemo-ImageAuthSystem-/
├── main.py                 # 命令行主程序（Rich UI）
├── web_app.py              # Web界面（Streamlit）
├── db_manager.py           # 数据库核心逻辑封装
├── BlockchainImageDB.sql   # 数据库初始化脚本
├── data/download/          # 测试图片存放目录（需自行创建）
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

**依赖安装**
```bash
pip install pymysql pillow imagehash rich streamlit
```

**数据库初始化**
```bash
# 1. 启动 MySQL 服务
# 2. 导入数据库结构
mysql -u root -p < BlockchainImageDB.sql
```

⚠️ **注意**：请修改 `main.py` 和 `web_app.py` 中的数据库密码：
```python
DB_CONFIG = {
    'host': 'localhost',
    'user':  'root',
    'password': '你的密码',  # 修改这里
    'database': 'image_auth_system'
}
```

### 2. 准备测试数据

在项目根目录创建以下结构：
```bash
mkdir -p data/download
```
将测试图片放入 `data/download/` 目录，命名为：
- `test1.jpg` (对应1972年尼克松访华照片)
- `test2.jpg` (对应1896年沈钧儒家庭照)

### 3. 运行系统

**方式一：命令行模式**
```bash
python main.py
```

**方式二：Web界面模式**
```bash
streamlit run web_app.py
```
访问：http://localhost:8501

---

## 📊 数据库视图说明

系统提供了预定义视图 `v_auth_report`，用于简化应用层查询：

```sql
SELECT * FROM v_auth_report WHERE description LIKE '%尼克松%';
```

**视图字段**：
- `uuid`: 系统内部流转ID
- `file_sha256`: 物理指纹
- `fact_time`: 历史时间
- `description`: 事件描述
- `chain_status`: 区块链状态（unsigned/signed）

---

## 🔐 安全机制

1. **防篡改**：SHA256 唯一索引确保相同文件不会重复入库
2. **防伪造**：pHash 感知哈希可识别视觉相似图片
3. **事务保障**：数据库操作采用原子性事务，失败自动回滚
4. **区块链存证**：预留区块链交互接口（待对接真实链）

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源协议

---

## 👨‍💻 作者

**Qin Guansan_65 (Tenerife-Q)**  
💼 区块链工程师 | 🎓 数据库系统设计  

---

## 📌 后续计划

- [ ] 对接真实区块链网络（如 Hyperledger Fabric）
- [ ] 增加图像深度学习篡改检测
- [ ] 支持批量上传与分布式存储
- [ ] 增强权限管理与审计日志

