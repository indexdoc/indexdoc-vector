<div align="center">
  <div style="font-size: 15px; line-height: 2; padding: 4px 0; letter-spacing: 0.5px;">
    <strong style="color: #24292f;">English</strong>
    | <a href="README.md" style="color: #0969da; text-decoration: none;">简体中文</a>
    <!-- | <a href="https://demo.aituple.com/pc/free/index.html?path=batchDoc" target="_blank" style="color: #165DFF; font-weight: 600; text-decoration: none;">✨ onlineDemo</a> -->
  </div>
</div>
  <div style="font-size: 14px; color: #57606a; padding: 2px 0; text-align: left;">
    <span style="background: #f6f8fa; padding: 2px 8px; border-radius: 4px; font-size: 13px;">Core Repos</span><br/>
    <a href="https://github.com/indexdoc/indexdoc-model-to-code" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-model-to-code（Code Generator / CodeAsst）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-ai-offline" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-ai-offline（Local Document AI Assistant）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-converter" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-converter（File Converter）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-editor" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-editor（Markdown Editor）</a><br/>
    <a href="https://github.com/indexdoc/indexdoc-batch-generator" target="_blank" style="color: #0969da; text-decoration: none; margin: 0 6px;">indexdoc-batch-generator（Batch Document Assistant）</a><br/>
  </div>

---
# MemMapVector: Lightweight Memory-Mapped Vector Storage Library
A lightweight, thread-safe memory-mapped vector storage library that supports efficient cosine similarity search and vector management. This library is now published on the Python Package Index (PyPI) and can be quickly installed and used via the pip package manager.

[![Python Version](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)  [![GitHub Stars](https://img.shields.io/github/stars/indexdoc/indexdoc-vector?style=social)](https://github.com/indexdoc/indexdoc-vector.git)   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Core Features
- **Memory-Mapped I/O**: Implemented based on `numpy.memmap`, supporting TB-level vector storage with extremely low memory footprint
- **Thread Safety**: Built-in fair and reentrant Read-Write Lock (RWLock) for high-concurrency read/write operations
- **Efficient Search**: Supports full-database cosine similarity search and search within specified ID ranges, optimized with batch processing
- **Vector Management**: Complete functionality for adding, deleting, compressing, and statistics collection of vectors
- **Zero Core Dependencies**: Only depends on NumPy, no complex database deployment required

## 🌏 Library Usage
```bash
# Library installation
pip install -U indexdoc-vector # Install the latest version of the library
```
- The minimum required Python version for using this library is Python 3.10
- Package directory structure
```bash
indexdoc-vector/          # Project root directory
├── indexdoc_vector/      # Core package directory
│   ├── __init__.py       # Core code
│   └── mem_map_vector.py
```

## 🚀 Quick Start

### Basic Usage Example
```python
import numpy as np
from indexdoc_vector.mem_map_vector import MemMapVector
# Initialize vector storage (default dimension: 512)
vec_db = MemMapVector("vectors.vec", dimension=512)

# Generate test vectors
vectors = np.random.rand(1000, 512).astype(np.float32)

# Add vectors
vector_ids = vec_db.add_vectors(vectors)
print(f"Added {len(vector_ids)} vectors, ID range: {vector_ids[0]} - {vector_ids[-1]}")

# Cosine similarity search
query = np.random.rand(512).astype(np.float32)
results = vec_db.cosine_search(query, top_k=10, min_score=0.5)
print("Top 10 search results:", results)

# Mark vectors as deleted (zero-out)
delete_ids = [0, 5, 10]
deleted_count = vec_db.mark_deleted(delete_ids)
print(f"Marked {deleted_count} vectors as deleted")

# Compact storage (remove zero vectors)
id_mapping = vec_db.compact()
print(f"Compaction completed, sample old-to-new ID mapping: {id_mapping[:5]}")

# Get statistics
stats = vec_db.stat()
print("Storage statistics:", stats)
```

## 📚 API Documentation

### Core Class
#### `MemMapVector(vec_file_path: str, dimension: int = 512)`
Initialize vector storage instance
- `vec_file_path`: Path to vector storage file
- `dimension`: Vector dimension, default 512

### Write Operations
#### `add_vectors(vectors: np.ndarray) -> List[int]`
Add batch vectors, returns list of assigned vector IDs
- `vectors`: 2D float32 array with shape (N, dimension)

#### `add_vector_list(vector_list: List[np.ndarray]) -> List[int]`
Add list of vectors; internally converts to array and calls `add_vectors`

#### `mark_deleted(vector_ids: List[int]) -> int`
Mark specified vector IDs as deleted (zero-out vectors), returns count of actually marked vectors

#### `compact() -> List[Tuple[int, int]]`
Compact vector file by removing all zero vectors, returns old-to-new ID mapping

### Search Operations
#### `cosine_search(query: np.ndarray, top_k: int = 10, min_score: float = 0.0, batch_size: int = 500_000) -> List[Tuple[int, float]]`
Full-library cosine similarity search
- `query`: Query vector (1D float32 array)
- `top_k`: Return top-K results
- `min_score`: Minimum similarity threshold
- `batch_size`: Batch processing size to control memory usage

### Read Operations
#### `cosine_search_in_vector_ids(query: np.ndarray, vector_ids: List[int], top_k: int = 10, min_score: float = 0.0, batch_size: int = 500_000) -> List[Tuple[int, float]]`
Cosine similarity search within specified vector ID range

#### `get_vector_by_vector_id(vector_id: int) -> np.ndarray | None`
Get single vector by ID; returns None for invalid ID

#### `get_total_vectors() -> int`
Get total vector count (including marked‑as‑deleted vectors)

#### `stat() -> Dict`
Get detailed storage statistics including:
- Total / valid / deleted vector counts
- File size and space utilization
- Reclaimable space size
- List of deleted vector IDs, etc.

## ⚡ Performance Optimizations
1. **Batch Processing**: Search and statistics use batch processing to avoid loading all data at once
2. **Sequential I/O**: Delete marking and compaction use sequential writes, greatly improving disk I/O efficiency
3. **Memory Management**: Explicitly closes memmap handles to prevent file locking and memory leaks
4. **Read-Write Separation**: Read-write lock enables concurrent reads and exclusive writes, maximizing throughput

## 🎯 Use Cases
- Small-to-medium scale vector retrieval (millions to tens of millions)
- Embedded systems or resource-constrained environments
- Scenarios requiring persistent storage without complex database deployment
- Lightweight alternative to heavyweight vector databases

## 🚨 Notes
1. Vector dimension is fixed at initialization and cannot be modified dynamically
2. Delete only zeros out vectors; call `compact()` to reclaim disk space
3. `compact()` changes vector IDs; you must maintain ID mapping

## 📊 Performance Reference
| Vector Count | Search Time | Memory Usage | Disk Usage |
|--------------|-------------|--------------|------------|
| 1M           | ~0.5s       | ~100MB       | ~2GB       |
| 10M          | ~5s         | ~100MB       | ~20GB      |
| 50M          | ~25s        | ~100MB       | ~100GB     |

*Test environment: Intel i7-12700H, 32GB RAM, NVMe SSD, 512-dimensional vectors*

---
