<div align="center">
  <div style="font-size: 15px; line-height: 2; padding: 4px 0; letter-spacing: 0.5px;">
    <strong style="color: #24292f;">简体中文</strong> 
    | <a href="README_EN.md" style="color: #0969da; text-decoration: none;">English</a>
    <!-- | <a href="https://demo.aituple.com/pc/free/index.html?path=batchDoc" target="_blank" style="color: #165DFF; font-weight: 600; text-decoration: none;">✨ 在线Demo</a> -->
  </div>
</div>
  <div style="font-size: 14px; color: #57606a; padding: 2px 0; text-align: left;">
    <span style="background: #f6f8fa; padding: 2px 8px; border-radius: 4px; font-size: 13px;">核心仓库</span><br/>
    <a href="https://github.com/indexdoc/indexdoc-model-to-code" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-model-to-code（代码生成器 / CodeAsst）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-ai-offline" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-ai-offline（本地文档AI助手）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-converter" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-converter（文档转换器）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-editor" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-editor（Markdown编辑器）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-batch-generator" target="_blank" style="color: #0969da; text-decoration: none; display: block; margin: 4px 0;">indexdoc-batch-generator（批量文档助手）</a><br/>
  </div>


---
# MemMapVector: 轻量级内存映射向量存储库
轻量级、线程安全的内存映射向量存储库，支持高效的余弦相似度搜索和向量管理。

[![Python Version](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)  [![GitHub Stars](https://img.shields.io/github/stars/indexdoc/indexdoc-vector?style=social)](https://github.com/indexdoc/indexdoc-vector.git)   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## ✨ 核心特性
- **内存映射IO**：基于 `numpy.memmap` 实现，支持TB级向量存储，内存占用极低
- **线程安全**：内置公平可重入读写锁(RWLock)，支持高并发读写
- **高效搜索**：支持全库余弦相似度搜索和指定ID范围搜索，批量处理优化
- **向量管理**：完整的添加、删除、压缩、统计功能
- **零依赖核心**：仅依赖NumPy，无需复杂的数据库部署
- **跨平台兼容**：支持Windows/Linux/macOS，处理Windows文件锁定问题
- **空间优化**：支持压缩删除的零向量，回收磁盘空间

## 🚀 快速开始

### 基础使用示例
```python

# 初始化向量存储（维度默认512）
vec_db = MemMapVector("vectors.vec", dimension=512)

# 生成测试向量
vectors = np.random.rand(1000, 512).astype(np.float32)

# 添加向量
vector_ids = vec_db.add_vectors(vectors)
print(f"添加了 {len(vector_ids)} 个向量，ID范围: {vector_ids[0]} - {vector_ids[-1]}")

# 余弦相似度搜索
query = np.random.rand(512).astype(np.float32)
results = vec_db.cosine_search(query, top_k=10, min_score=0.5)
print("Top 10 搜索结果:", results)

# 标记向量为删除（置零）
delete_ids = [0, 5, 10]
deleted_count = vec_db.mark_deleted(delete_ids)
print(f"标记 {deleted_count} 个向量为删除")

# 压缩存储（移除零向量）
id_mapping = vec_db.compact()
print(f"压缩完成，ID映射关系示例: {id_mapping[:5]}")

# 获取统计信息
stats = vec_db.stat()
print("存储统计信息:", stats)
```

## 📚 API 文档

### 核心类
#### `MemMapVector(vec_file_path: str, dimension: int = 512)`
初始化向量存储实例
- `vec_file_path`: 向量文件存储路径
- `dimension`: 向量维度，默认512

### 写操作
#### `add_vectors(vectors: np.ndarray) -> List[int]`
添加批量向量，返回分配的vector_id列表
- `vectors`: 二维float32数组，形状为(N, dimension)

#### `add_vector_list(vector_list: List[np.ndarray]) -> List[int]`
添加向量列表，内部会转换为数组后调用`add_vectors`

#### `mark_deleted(vector_ids: List[int]) -> int`
标记指定ID的向量为删除（置零向量），返回实际标记的数量

#### `compact() -> List[Tuple[int, int]]`
压缩向量文件，移除所有零向量，返回新旧ID映射关系

### 读操作
#### `cosine_search(query: np.ndarray, top_k: int = 10, min_score: float = 0.0, batch_size: int = 500_000) -> List[Tuple[int, float]]`
全库余弦相似度搜索
- `query`: 查询向量（一维float32数组）
- `top_k`: 返回Top-K结果
- `min_score`: 最小相似度阈值
- `batch_size`: 批量处理大小，控制内存使用

#### `cosine_search_in_vector_ids(query: np.ndarray, vector_ids: List[int], top_k: int = 10, min_score: float = 0.0, batch_size: int = 500_000) -> List[Tuple[int, float]]`
在指定ID范围内进行余弦相似度搜索

#### `get_vector_by_vector_id(vector_id: int) -> np.ndarray | None`
根据ID获取单个向量，无效ID返回None

#### `get_total_vectors() -> int`
获取向量总数（包括已标记删除的向量）

#### `stat() -> Dict`
获取详细的存储统计信息，包括：
- 向量总数/有效数/删除数
- 文件大小和空间利用率
- 可回收空间大小
- 删除的向量ID列表等

## ⚡ 性能优化
1. **批量处理**：搜索和统计功能均采用批量处理，避免一次性加载所有数据
2. **顺序IO**：删除标记和压缩操作采用顺序写入，大幅提升磁盘IO效率
3. **内存管理**：显式关闭memmap句柄，避免文件锁定和内存泄漏
4. **读写分离**：读写锁机制保证读操作并发，写操作独占，最大化并发性能

## 🎯 适用场景
- 中小规模向量检索（百万至千万级）
- 嵌入式系统或资源受限环境
- 需要持久化存储但不想部署复杂数据库的场景
- 作为大型向量数据库的轻量级替代方案

## 🚨 注意事项
1. 向量维度在初始化时确定，无法动态修改
2. 删除操作只是置零向量，需要调用`compact()`才能回收磁盘空间
3. `compact()`操作会改变向量ID，需要维护ID映射关系
4. Windows系统下注意文件锁定问题，确保正确关闭memmap句柄

## 📊 性能参考
| 向量数量 | 搜索时间 | 内存占用 | 磁盘占用 |
|---------|---------|---------|---------|
| 100万   | ~0.5s   | ~100MB  | ~2GB    |
| 1000万  | ~5s     | ~100MB  | ~20GB   |
| 5000万  | ~25s    | ~100MB  | ~100GB  |

*测试环境：Intel i7-12700H, 32GB RAM, NVMe SSD, 512维向量*

---
