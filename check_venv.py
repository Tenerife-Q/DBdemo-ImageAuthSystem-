"""
检查虚拟环境是否正确激活
"""
import sys
import os

print("=" * 60)
print("🔍 虚拟环境检查")
print("=" * 60)

print(f"\n✓ Python路径: {sys.executable}")
print(f"✓ 虚拟环境前缀: {sys.prefix}")
print(f"✓ 项目文件夹: {os.getcwd()}")

# 检查是否在虚拟环境中
if 'venv' in sys.prefix:
    print("\n✅ 虚拟环境已激活！")
    print(f"   激活位置: {sys.prefix}")
else:
    print("\n❌ 虚拟环境未激活！")
    print(f"   使用的是系统Python: {sys.prefix}")

print("\n已安装的库:")
try:
    import requests
    print(f"  ✓ requests: {requests.__version__}")
except: 
    print(f"  ✗ requests 未安装")

try:
    import bs4
    print(f"  ✓ beautifulsoup4: 已安装")
except:
    print(f"  ✗ beautifulsoup4 未安装")

print("\n" + "=" * 60)