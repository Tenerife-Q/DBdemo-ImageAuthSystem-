"""
🎓 我的第一个爬虫 - Wikipedia图文爬取
这是最简单的版本，用来理解基础原理
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import time

# ==================== 第一部分：配置 ====================
class Config:
    """爬虫配置 - 改这里就能改爬虫行为"""
    
    # 目标网址（Wikipedia词条）
    TARGET_URL = "https://en.wikipedia.org/wiki/Misinformation"
    
    # 保存文件夹
    OUTPUT_DIR = "downloaded_data"
    
    # 图片文件夹
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    
    # 要下载的图片数量
    MAX_IMAGES = 10
    
    # 请求头（告诉服务器你是浏览器）
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

# ==================== 第二部分：爬虫主逻辑 ====================
class WikipediaScraper: 
    """Wikipedia爬虫 - 从维基百科爬取图文数据"""
    
    def __init__(self, config=Config):
        self.config = config
        # 创建输出目录（同时创建 images 子目录）
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.IMAGES_DIR, exist_ok=True)
        print(f"✓ 初始化完成，输出目录: {self.config.OUTPUT_DIR}")
    
    def step1_download_html(self, url):
        """
        第一步：下载网页
        作用：获取网页的HTML代码
        """
        print("\n" + "="*60)
        print("📖 第一步：下载网页")
        print("="*60)
        
        try: 
            print(f"正在访问: {url}")
            
            # 关键代码：发送HTTP请求
            response = requests.get(url, headers=self.config.HEADERS, timeout=10)
            
            # 检查是否成功
            if response. status_code == 200:
                print(f"✅ 成功！状态码: {response.status_code}")
                print(f"   网页大小: {len(response.text)/1024:.1f} KB")
                return response
            else:
                print(f"❌ 失败！状态码: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 错误:  {e}")
            return None
    
    def step2_parse_html(self, response):
        """
        第二步：解析HTML
        作用：从HTML中提取我们需要的信息
        """
        print("\n" + "="*60)
        print("🔍 第二步：解析HTML")
        print("="*60)
        
        try:
            # 关键代码：使用BeautifulSoup解析
            soup = BeautifulSoup(response.text, 'html.parser')
            print("✅ HTML解析成功")
            
            # 提取页面标题
            title = soup.find('h1', class_='firstHeading')
            if title:
                print(f"   页面标题: {title.get_text()}")
            
            # 提取页面摘要
            summary = soup. find('p')
            if summary:
                summary_text = summary.get_text()[:100]
                print(f"   页面摘要: {summary_text}...")
            
            return soup
            
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            return None
    
    def step3_extract_images(self, soup):
        """
        第三步：提取图片
        作用：从HTML中找出所有图片的链接
        """
        print("\n" + "="*60)
        print("🖼️  第三步：提取图片链接")
        print("="*60)
        
        try:  
            # 方法1：找所有img标签
            all_images = soup.find_all('img')
            print(f"✅ 找到 {len(all_images)} 个图片标签")
            
            # 方法2：过滤出有用的图片（过滤掉logo、icon等）
            useful_images = []
            for idx, img in enumerate(all_images):
                src = img.get('src')
                alt = img.get('alt', '')
                
                # 只要有src属性的图片
                if src:  
                    useful_images.append({
                        'src': src,
                        'alt': alt,
                        'idx': idx
                    })
            
            print(f"   有效图片:  {len(useful_images)} 个")
            
            # 显示前5个
            print("\n   前5个图片:")
            for i, img in enumerate(useful_images[: 5]):
                print(f"      {i+1}. {img['alt'][: 50]}")
            
            return useful_images
            
        except Exception as e:
            print(f"❌ 提取失败: {e}")
            return []
    
    def step4_download_images(self, response, image_list):
        """
        第四步：下载图片
        作用：将图片URL转换成本地文件
        """
        print("\n" + "="*60)
        print("⬇️  第四步：下载图片")
        print("="*60)
        
        downloaded_count = 0
        failed_count = 0
        
        for idx, img_info in enumerate(image_list[: self.config.MAX_IMAGES]):
            src = img_info['src'] or ''
            
            # 跳过 data URI
            if src.startswith('data:'):
                print("   ⏭️  跳过 data URL")
                continue
            
            # 使用 urljoin 规范化 URL（处理 //、/ 和相对路径）
            src = urljoin(response.url, src)
            
            try:
                print(f"\n   [{idx+1}/{min(self.config.MAX_IMAGES, len(image_list))}] 下载中...")
                print(f"   URL: {src[: 120]}...")
                
                # 下载图片
                img_response = requests.get(src, timeout=10, headers=self.config.HEADERS)
                
                if img_response.status_code == 200:
                    # 尝试用 PIL 打开图片，失败则跳过
                    try:
                        img = Image.open(BytesIO(img_response.content))
                    except Exception as e:
                        print(f"   ❌ 无法识别图片文件: {e}")
                        failed_count += 1
                        continue
                    
                    # 获取图片信息
                    width, height = img.size
                    file_format = img.format
                    
                    # 只保存较大的图片（过滤掉 icon）
                    if width > 100 and height > 100:
                        # 修正文件名空格问题，统一为 .jpg
                        filename = f"image_{idx:03d}.jpg"
                        filepath = os.path.join(self.config.IMAGES_DIR, filename)
                        
                        # 转换为RGB（有些图片是PNG或其他格式）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 保存（以 JPEG 形式）
                        try:
                            img.save(filepath, format='JPEG', quality=85)
                        except Exception as e:
                            print(f"   ❌ 保存失败: {e}")
                            failed_count += 1
                            continue
                        
                        downloaded_count += 1
                        print(f"   ✅ 成功!  大小: {width}x{height} 格式: {file_format}")
                    else:
                        print(f"   ⏭️  跳过（太小:  {width}x{height}）")
                else:
                    print(f"   ❌ 图片请求失败，状态码: {img_response.status_code}")
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f"   ❌ 失败: {str(e)[:200]}")
        
        print(f"\n   总结:  {downloaded_count} 成功, {failed_count} 失败")
        return downloaded_count
    
    def step5_save_metadata(self, soup):
        """
        第五步：保存元数据（文本）
        作用：保存网页文本内容和图片对应关系
        """
        print("\n" + "="*60)
        print("📝 第五步：保存元数据")
        print("="*60)
        
        try:
            # 提取所有文本
            paragraphs = soup.find_all('p')
            text_content = []
            
            for p in paragraphs[: 10]:  # 只取前10段
                text = p.get_text().strip()
                if text and len(text) > 20:  # 过滤掉太短的
                    text_content.append(text)
            
            # 保存为文本文件
            metadata_path = os.path.join(self.config.OUTPUT_DIR, "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("爬虫数据汇总\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"数据源: {self.config.TARGET_URL}\n")
                f.write(f"爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("提取的文本内容:\n")
                f.write("-" * 60 + "\n")
                for i, text in enumerate(text_content, 1):
                    f.write(f"\n{i}. {text[: 200]}...\n")
            
            print(f"✅ 元数据已保存:  {metadata_path}")
            print(f"   包含 {len(text_content)} 段文本")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")

# ==================== 第三部分：主程序 ====================
def main():
    """主函数 - 执行完整的爬虫流程"""
    
    print("\n" + "🎓 "*20)
    print("欢迎使用：Wikipedia 图文爬虫")
    print("🎓 "*20 + "\n")
    
    # 1. 创建爬虫实例
    scraper = WikipediaScraper()
    
    # 2. 执行第一步：下载HTML
    response = scraper.step1_download_html(Config. TARGET_URL)
    if not response:
        print("❌ 程序中止")
        return
    
    # 3. 执行第二步：解析HTML
    soup = scraper.step2_parse_html(response)
    if not soup:
        print("❌ 程序中止")
        return
    
    # 4. 执行第三步：提取图片
    images = scraper.step3_extract_images(soup)
    if not images:
        print("❌ 没找到图片")
        return
    
    # 5. 执行第四步：下载图片
    downloaded = scraper.step4_download_images(response, images)
    
    # 6. 执行第五步：保存元数据
    scraper.step5_save_metadata(soup)
    
    # 7. 最终总结
    print("\n" + "="*60)
    print("✅ 爬虫完成!")
    print("="*60)
    print(f"📁 输出目录: {Config.OUTPUT_DIR}/")
    print(f"   ├── images/ ({downloaded} 张图片)")
    print(f"   └── metadata.txt (文本数据)")
    print("\n下一步：检查 downloaded_data 文件夹查看结果 👉")

if __name__ == "__main__":
    main()