import streamlit as st
import pymysql
import os
from PIL import Image

# ================= 数据库配置 =================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '668157',  
    'database': 'image_auth_system'
}
# ============================================

def get_db_connection():
    """获取数据库连接"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None

def main():
    # 页面基础设置
    st.set_page_config(
        page_title="图像可信鉴真系统",
        layout="wide"
    )

    # 侧边栏：系统状态面板
    with st.sidebar:
        st.header("系统状态面板")
        st.text("连接状态: 在线")
        st.text("当前节点: Localhost")
        st.markdown("---")
        st.text("功能模块:")
        st.text("1. 语义检索 (Active)")
        st.text("2. 区块链存证 (Standby)")

    # 主界面：标题区
    st.header("基于多模态与区块链的图像可信鉴真系统")
    st.markdown("Image Trust & Authentication System based on Blockchain")
    st.markdown("---")

    # 输入区
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        search_keyword = st.text_input("请输入查询关键词", placeholder="支持时间、地点、事件描述模糊匹配")
    with col_btn:
        st.write("") # 占位排版
        st.write("") 
        is_search = st.button("执行检索", use_container_width=True)

    # 业务逻辑区
    if is_search or search_keyword:
        conn = get_db_connection()
        if not conn:
            return

        cursor = conn.cursor()
        
        # SQL逻辑：关联 image_core (获取图片路径/UUID), semantic_facts (获取描述), chain_anchor (获取状态)
        # 注意：这里直接关联表查询，以获取 storage_path 字段，保证图片可显示
        sql_query = """
        SELECT 
            c.storage_path, 
            c.uuid, 
            s.fact_time, 
            s.description, 
            IFNULL(a.status, 'unsigned') as chain_status
        FROM image_core c
        JOIN semantic_facts s ON c.id = s.image_id
        LEFT JOIN chain_anchor a ON c.id = a.image_id
        WHERE s.description LIKE %s OR s.fact_location LIKE %s
        """
        
        # 执行查询
        param = f"%{search_keyword}%"
        cursor.execute(sql_query, (param, param))
        results = cursor.fetchall()
        conn.close()

        # 结果渲染
        if results:
            st.success(f"检索完成，共匹配到 {len(results)} 条记录。")
            
            for index, row in enumerate(results):
                img_path, uuid, fact_time, description, chain_status = row
                
                with st.container():
                    st.markdown("---")
                    col_img, col_info = st.columns([2, 3])
                    
                    # 左侧：显示影像证据
                    with col_img:
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            st.image(image, caption="原始影像副本", use_container_width=True)
                        else:
                            st.warning(f"文件未找到: {img_path}")

                    # 右侧：显示元数据与存证状态
                    with col_info:
                        st.subheader("影像元数据档案")
                        st.text(f"系统唯一标识 (UUID): {uuid}")
                        st.text(f"历史时间事实: {fact_time}")
                        
                        st.markdown("**语义描述:**")
                        st.info(description)
                        
                        st.markdown("**区块链锚定状态:**")
                        if chain_status == 'unsigned':
                            st.markdown("状态: **待签名 (Unsigned)**")
                            # 模拟上链交互
                            if st.button(f"发起上链请求", key=f"btn_{index}"):
                                st.info("正在调用智能合约接口... (模拟)")
                                st.success(f"交易广播成功! TX_HASH: 0x{uuid[:8]}...")
                        else:
                            st.markdown(f"状态: **已确认 ({chain_status})**")
        else:
            if search_keyword:
                st.warning("未检索到匹配的存证记录。")

if __name__ == "__main__":
    main()