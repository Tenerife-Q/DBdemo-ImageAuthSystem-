"""
Reddit爬虫 - 爬取Reddit上的新闻和图片
适合虚假信息检测项目
"""

import requests
import json
import os
from datetime import datetime
import time

class RedditScraper: 
    """Reddit爬虫 - 爬取子版块数据"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def __init__(self, output_dir='reddit_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 初始化完成: {output_dir}/")
    
    def scrape_subreddit(self, subreddit_name, post_count=30):
        """
        爬取Reddit子版块
        
        参数:
        - subreddit_name: 版块名称 (如 'news', 'worldnews')
        - post_count: 要爬取的帖子数
        """
        print("\n" + "="*70)
        print(f"🔍 爬取 Reddit - r/{subreddit_name}")
        print("="*70)
        
        try:
            # Reddit官方JSON API
            url = f"https://www.reddit.com/r/{subreddit_name}/new.json"
            
            print(f"\n[1/3] 连接到:  {url}")
            
            # 发送请求
            response = requests.get(
                url,
                headers=self.HEADERS,
                params={'limit': post_count},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ 失败!  状态码: {response. status_code}")
                return None
            
            print("✅ 连接成功")
            
            # 解析JSON数据
            print(f"\n[2/3] 解析数据...")
            data = response. json()
            posts = []
            
            # 提取帖子信息
            for post_data in data['data']['children']:
                post = post_data['data']
                
                # 获取关键信息
                post_info = {
                    'id': post. get('id', ''),
                    'title': post.get('title', ''),
                    'text': post.get('selftext', '')[:  300],  # 前300字
                    'author':  post.get('author', ''),
                    'subreddit': post.get('subreddit', ''),
                    'score': post.get('score', 0),  # 赞数
                    'comments': post.get('num_comments', 0),  # 评论数
                    'url': post.get('url', ''),
                    'image_url': post.get('preview', {}).get('images', [{}])[0].get('source', {}).get('url', ''),
                    'created_at': datetime.fromtimestamp(post.get('created_utc', 0)).isoformat()
                }
                
                posts.append(post_info)
            
            print(f"✅ 解析完成，找到 {len(posts)} 个帖子")
            
            # 保存为JSON
            print(f"\n[3/3] 保存数据...")
            filename = os.path.join(
                self.output_dir,
                f"r_{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 数据已保存:  {filename}")
            
            # 统计信息
            print("\n" + "="*70)
            print(f"📊 统计信息 (r/{subreddit_name}):")
            print("="*70)
            
            total_score = sum(p['score'] for p in posts)
            total_comments = sum(p['comments'] for p in posts)
            posts_with_images = len([p for p in posts if p['image_url']])
            posts_with_text = len([p for p in posts if p['text']])
            
            print(f"  📝 总帖子数: {len(posts)}")
            print(f"  👍 总赞数: {total_score}")
            print(f"  💬 总评论数: {total_comments}")
            print(f"  🖼️  包含图片: {posts_with_images} 个")
            print(f"  📄 包含文本: {posts_with_text} 个")
            print(f"  ⭐ 平均评分: {total_score / len(posts) if posts else 0:.1f}")
            
            # 显示前3个帖子
            print(f"\n前3个热门帖子:")
            print("-"*70)
            for i, post in enumerate(sorted(posts, key=lambda x:  x['score'], reverse=True)[:  3], 1):
                print(f"\n{i}. [{post['score']} 赞] {post['title'][:  60]}")
                print(f"   作者: {post['author']} | 评论: {post['comments']}")
            
            print("\n" + "="*70)
            
            return posts
            
        except Exception as e:
            print(f"\n❌ 错误:  {e}")
            return None

# ==================== 运行 ====================
if __name__ == "__main__":
    scraper = RedditScraper('reddit_data')
    
    # 爬取多个子版块
    subreddits = [
        'news',          # 新闻
        'worldnews',     # 世界新闻  
        'nottheonion',   # 不是洋葱新闻（真实但荒谬的新闻）
    ]
    
    print("\n🚀 开始爬取Reddit数据...")
    print(f"将爬取 {len(subreddits)} 个子版块\n")
    
    for subreddit in subreddits: 
        scraper.scrape_subreddit(subreddit, post_count=20)
        time.sleep(2)  # 礼貌延迟（避免被Ban）
    
    print("\n✅ 所有爬虫任务完成!")
    print("📁 查看 reddit_data/ 文件夹查看JSON数据")