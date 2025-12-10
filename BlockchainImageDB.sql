/* 项目名称：基于多模态与区块链的图像可信鉴真系统
   描述：包含数据源管理、核心资产存储、语义事实分离及区块链锚定结构
*/

-- 1. 初始化环境
DROP DATABASE IF EXISTS image_auth_system;
CREATE DATABASE image_auth_system DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE image_auth_system;

-- 2. 数据来源表 (Data Source)
-- 符合3NF范式，独立管理来源，便于系统扩展
CREATE TABLE data_source (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(50) NOT NULL COMMENT '来源名称',
    source_type VARCHAR(20) DEFAULT 'crawler',
    trust_level TINYINT DEFAULT 1 COMMENT '可信度 1-5'
);

-- 预埋数据
INSERT INTO data_source (source_name, source_type, trust_level) VALUES ('新华社_多媒体库', 'crawler', 5);

-- 3. 原始数据缓冲表 (Raw Buffer)
-- 外键关联数据源，确保数据可溯源
CREATE TABLE raw_crawl_buffer (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_id INT DEFAULT 1,
    raw_json JSON COMMENT '原始元数据副本',
    status VARCHAR(20) DEFAULT 'processed',
    ingest_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES data_source(id)
);

-- 4. 核心影像资产表 (Core Assets)
-- 物理存储与哈希指纹，实现存证分离
CREATE TABLE image_core (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid VARCHAR(64) UNIQUE NOT NULL COMMENT '系统内部流转ID',
    file_sha256 VARCHAR(64) UNIQUE NOT NULL COMMENT 'SHA256物理指纹(防篡改)',
    p_hash VARCHAR(64) NOT NULL COMMENT 'pHash感知指纹(防伪造)',
    storage_path VARCHAR(255),
    source_url VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 语义事实表 (Semantic Facts)
-- 存取分离，支持全文检索
CREATE TABLE semantic_facts (
    image_id INT PRIMARY KEY,
    fact_time VARCHAR(50) COMMENT '历史时间',
    fact_location VARCHAR(100),
    description TEXT,
    FOREIGN KEY (image_id) REFERENCES image_core(id) ON DELETE CASCADE,
    FULLTEXT KEY ft_desc (description)
);

-- 6. 区块链锚定表 (Chain Anchor)
-- 模拟区块链交互接口
CREATE TABLE chain_anchor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    tx_hash VARCHAR(100) DEFAULT NULL COMMENT '链上交易哈希',
    block_height INT DEFAULT 0 COMMENT '区块高度',
    status VARCHAR(20) DEFAULT 'unsigned',
    FOREIGN KEY (image_id) REFERENCES image_core(id)
);

-- 7. 鉴真视图 (View)
-- 简化应用层查询逻辑，聚合核心信息
CREATE VIEW v_auth_report AS
SELECT 
    c.uuid, 
    c.file_sha256, 
    s.fact_time, 
    s.description, 
    a.status AS chain_status
FROM image_core c
JOIN semantic_facts s ON c.id = s.image_id
LEFT JOIN chain_anchor a ON c.id = a.image_id;