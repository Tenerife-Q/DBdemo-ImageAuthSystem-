"""
改进版爬虫 - 解决Wikipedia图片下载问题
使用更智能的图片识别和处理
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm

class ImprovedScraper:
    """改进的爬虫"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def __init__(self, output_dir='downloaded_data_v2'):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        print(f"✓ 初始化完成:  {output_dir}/")
    
    def scrape(self, url, max_images=15):
        """
        改进的爬虫逻辑
        """
        print("\n" + "="*70)
        print(f"🔍 开始爬取: {url}")
        print("="*70)
        
        try:
            # 1. 下载网页
            print("\n[1/5] 下载网页...")
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            response. encoding = 'utf-8'
            print(f"✅ 成功 (大小: {len(response.text)/1024:.1f} KB)")
            
            # 2. 解析HTML
            print("\n[2/5] 解析HTML...")
            soup = BeautifulSoup(response.text, 'html.parser')
            print("✅ 解析完成")
            
            # 3. 提取所有图片
            print("\n[3/5] 提取图片链接...")
            all_img_tags = soup.find_all('img')
            print(f"✅ 找到 {len(all_img_tags)} 个图片标签")
            
            # 4. 过滤有用的图片
            print("\n[4/5] 过滤图片...")
            useful_images = []
            
            for img in all_img_tags:
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                # 转换相对URL为绝对URL
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    src = urljoin(url, src)
                
                # 过滤掉太小的图片（icon等）
                width = img.get('width', '0')
                height = img. get('height', '0')
                
                try:
                    if int(width or 0) > 150 and int(height or 0) > 150:
                        useful_images.append(src)
                except: 
                    # 如果没有width/height属性，也加入（稍后判断）
                    useful_images.append(src)
            
            print(f"✅ 过滤后:  {len(useful_images)} 个有用的图片")
            
            # 5. 下载图片
            print(f"\n[5/5] 下载图片 (最多 {max_images} 张)...")
            
            downloaded = 0
            failed = 0
            skipped = 0
            
            # 用进度条
            for idx, img_url in enumerate(useful_images[: max_images]):
                try:
                    # 下载
                    img_response = requests.get(img_url, timeout=5, headers=self.HEADERS)
                    
                    if img_response.status_code != 200:
                        failed += 1
                        continue
                    
                    # 打开图片
                    try:
                        img = Image. open(BytesIO(img_response.content))
                    except: 
                        # 如果PIL无法识别，尝试用其他方式
                        failed += 1
                        continue
                    
                    # 检查图片大小（过滤太小的）
                    width, height = img.size
                    if width < 150 or height < 150:
                        skipped += 1
                        continue
                    
                    # 转换为RGB
                    if img.mode != 'RGB': 
                        img = img.convert('RGB')
                    
                    # 保存
                    filename = f"image_{downloaded: 03d}. jpg"
                    filepath = os.path.join(self.output_dir, 'images', filename)
                    img.save(filepath, quality=85, optimize=True)
                    
                    downloaded += 1
                    print(f"   ✓ [{downloaded}] {width}x{height} - {filename}")
                    
                except Exception as e:
                    failed += 1
            
            # 提取文本
            print(f"\n[提取文本]...")
            paragraphs = soup.find_all('p')
            texts = []
            for p in paragraphs[: 15]: 
                text = p.get_text().strip()
                if len(text) > 50:  # 只要>50字的段落
                    texts.append(text)
            
            # 保存元数据
            metadata_path = os.path.join(self.output_dir, 'metadata.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("爬虫数据汇总\n")
                f.write("="*70 + "\n\n")
                f.write(f"数据源: {url}\n")
                f.write(f"爬取时间: {time. strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"图片统计:\n")
                f.write(f"  成功: {downloaded} 张\n")
                f.write(f"  失败: {failed} 张\n")
                f.write(f"  跳过: {skipped} 张\n\n")
                f.write("提取的文本:\n")
                f.write("-"*70 + "\n")
                for i, text in enumerate(texts, 1):
                    f.write(f"\n{i}. {text[: 150]}...\n")
            
            # 总结
            print("\n" + "="*70)
            print("✅ 爬虫完成!")
            print("="*70)
            print(f"📊 统计:")
            print(f"   ✓ 成功下载: {downloaded} 张图片")
            print(f"   ✗ 下载失败: {failed} 张")
            print(f"   ⊘ 跳过: {skipped} 张")
            print(f"   📝 文本段落:  {len(texts)} 段")
            print(f"\n📁 输出目录: {self.output_dir}/")
            print(f"   ├── images/ ({downloaded} 张图片)")
            print(f"   └── metadata.txt")
            print("="*70)
            
            return {
                'downloaded': downloaded,
                'failed': failed,
                'skipped': skipped,
                'texts': len(texts)
            }
            
        except Exception as e: 
            print(f"\n❌ 错误: {e}")
            return None

# ==================== 运行 ====================
if __name__ == "__main__":
    scraper = ImprovedScraper('downloaded_data_v2')
    
    # 试试爬虚假新闻相关的Wikipedia页面
    scraper.scrape(
        url='https://en.wikipedia.org/wiki/Fake_news',
        max_images=15
    )