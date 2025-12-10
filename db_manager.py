import pymysql
import hashlib
import imagehash
import json
import uuid
from PIL import Image

class ImageDB:
    def __init__(self, config):
        self.config = config
    
    def get_conn(self):
        return pymysql.connect(**self.config)

    # 1. 计算双重指纹：SHA256 (物理层) + pHash (感知层)
    def compute_fingerprints(self, path):
        try:
            # 计算 SHA256
            sha = hashlib.sha256()
            with open(path, 'rb') as f:
                while chunk := f.read(8192): sha.update(chunk)
            
            # 计算 pHash
            img = Image.open(path)
            phash = str(imagehash.phash(img))
            return sha.hexdigest(), phash
        except Exception as e:
            print(f"文件读取错误: {path} - {e}")
            return None, None

    # 2. 核心业务流程：ETL入库
    def process_data(self, file_path, meta):
        conn = self.get_conn()
        conn.autocommit = False # 开启事务
        try:
            cursor = conn.cursor()
            
            # Step A: 原始数据留痕 (存入 raw_crawl_buffer)
            cursor.execute("INSERT INTO raw_crawl_buffer (source_id, raw_json) VALUES (1, %s)", 
                           (json.dumps(meta, ensure_ascii=False),))
            
            # Step B: 计算指纹
            sha, phash = self.compute_fingerprints(file_path)
            if not sha: return "FILE_ERROR"

            # Step C: 查重 (利用 SHA256 唯一索引)
            cursor.execute("SELECT id FROM image_core WHERE file_sha256=%s", (sha,))
            if cursor.fetchone():
                conn.rollback() # 发现重复，回滚 Step A
                return "DUPLICATE_SKIPPED"

            # Step D: 存入核心库 (image_core)
            sys_uuid = str(uuid.uuid4())
            cursor.execute("INSERT INTO image_core (uuid, file_sha256, p_hash, storage_path, source_url) VALUES (%s,%s,%s,%s,%s)",
                           (sys_uuid, sha, phash, file_path, meta['url']))
            img_id = cursor.lastrowid

            # Step E: 存入语义事实表 (semantic_facts)
            cursor.execute("INSERT INTO semantic_facts (image_id, fact_time, fact_location, description) VALUES (%s,%s,%s,%s)",
                           (img_id, meta['date'], meta['loc'], meta['desc']))

            # Step F: 初始化区块链锚定 (chain_anchor)
            cursor.execute("INSERT INTO chain_anchor (image_id, status) VALUES (%s, 'unsigned')", (img_id,))

            conn.commit() # 提交事务
            return "SUCCESS"
        except Exception as e:
            conn.rollback()
            print(f"Database Error: {e}")
            return "FAILED"
        finally:
            conn.close()

    # 3. 搜索功能：利用数据库视图 (View) 进行查询
    def search(self, keyword):
        conn = self.get_conn()
        cursor = conn.cursor()
        # 直接查询 v_auth_report 视图
        sql = """
        SELECT uuid, fact_time, description, chain_status 
        FROM v_auth_report 
        WHERE description LIKE %s
        """
        cursor.execute(sql, (f"%{keyword}%",))
        return cursor.fetchall()