import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from db_manager import ImageDB

# ================= 配置区域 =================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '668157', 
    'database': 'image_auth_system'
}
# ===========================================

console = Console()
db = ImageDB(DB_CONFIG)

# 模拟数据
MOCK_DATA = [
    {
        "filename": "test1.jpg", 
        "url": "http://xinhua.org/history/1972/nixon_visit.jpg",
        "desc": "1972年2月21日，周恩来总理在首都机场迎接美国总统尼克松。中美两国领导人的手握在一起，标志着中美关系一个新时代的开始。",
        "date": "1972-02-21",
        "loc": "北京首都机场"
    },
    {
        "filename": "test2.jpg", 
        "url": "http://photoarchive.cn/1896/shen_family.jpg",
        "desc": "1896年沈钧儒家庭照。这是中国照片档案馆现存最早的玻璃底片，记录了晚清时期沈钧儒与家人的珍贵影像。",
        "date": "1896",
        "loc": "浙江嘉兴"
    }
]

def main():
    console.print(Panel.fit("[bold cyan]基于多模态与区块链的图像可信鉴真系统[/bold cyan]", subtitle="Created by Blockchain Engineer"))
    
    # --- 阶段一：ETL 数据入库 ---
    console.print("\n[yellow]>>> 正在初始化数据源 [新华社_多媒体库]...[/yellow]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("文件名", style="dim")
    table.add_column("时间事实")
    table.add_column("入库状态")
    table.add_column("哈希指纹")

    for item in MOCK_DATA:
        path = os.path.join("data", "download", item['filename'])
        
        # 检查文件是否存在
        if not os.path.exists(path):
            table.add_row(item['filename'], "---", "[文件缺失]", "---")
            continue

        # 执行核心逻辑
        status = db.process_data(path, item)
        
        status_str = f"[green]入库成功[/green]" if status == "SUCCESS" else \
                     f"[yellow]重复跳过[/yellow]" if status == "DUPLICATE_SKIPPED" else \
                     f"[red]失败[/red]"
        
        table.add_row(item['filename'], item['date'], status_str, "SHA256+pHash")

    console.print(table)

    # --- 阶段二：系统演示 ---
    console.print("\n[yellow]>>> 系统就绪，进入检索模式[/yellow]")
    console.print("[dim]提示：输入 '尼克松' 或 '沈钧儒' 进行搜索 (输入 exit 退出)[/dim]")

    while True:
        keyword = console.input("\n[bold green]请输入查询关键词 > [/bold green]")
        if keyword.strip().lower() == 'exit':
            console.print("[bold cyan]系统已关闭。[/bold cyan]")
            break
        
        results = db.search(keyword)
        if results:
            res_table = Table(title=f"检索结果: '{keyword}'")
            res_table.add_column("UUID (系统ID)", style="dim", width=12)
            res_table.add_column("拍摄时间", style="cyan")
            res_table.add_column("语义描述", width=40)
            res_table.add_column("链上状态", justify="center")

            for row in results:
                # row[0]=uuid, row[1]=time, row[2]=desc, row[3]=status
                chain_status = f"[red]{row[3]}[/red]" if row[3] == "unsigned" else f"[green]{row[3]}[/green]"
                res_table.add_row(
                    row[0][:8]+"...", 
                    str(row[1]), 
                    row[2][:30]+"...", 
                    chain_status
                )
            console.print(res_table)
        else:
            console.print("[red]未找到匹配的存证记录[/red]")

if __name__ == "__main__":
    main()



# 相关命令 后续执行可简化指令
#  D:/python3.10/python.exe main.py
#  D:/python3.10/python.exe -m streamlit run web_app.py