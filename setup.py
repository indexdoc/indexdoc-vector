from setuptools import setup, find_packages
import pathlib

# 读取README作为项目描述（可选）
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

# 读取requirements.txt中的依赖
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="indexdoc_vector",          # 你的工具名称（PyPI上唯一）
    version="0.0.1",                # 版本号（遵循语义化版本）
    description="轻量级、线程安全的内存映射向量存储库，支持高效的余弦相似度搜索和向量管理。",
    long_description=README,
    long_description_content_type="text/markdown",
    author="杭州智予数信息技术有限公司",
    author_email="indexdoc@qq.com",
    url="https://github.com/indexdoc/indexdoc-vector.git",  # 可选
    packages=find_packages(),       # 自动发现所有包
    install_requires=requirements,  # 关键：指定安装时需要自动下载的依赖
    python_requires=">=3.10",        # 指定兼容的Python版本
    # entry_points={                  # 可选：配置命令行入口（如直接用your_tool运行）
    #     "console_scripts": [
    #         "your_tool = your_tool_name.main:main",
    #     ],
    # },
)