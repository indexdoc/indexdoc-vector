import shutil

import numpy as np
import os
from typing import List, Tuple, Dict
import logging
import threading
from pathlib import Path
import settings
import time
MIN_SCORE = settings.MIN_SCORE * 0.5

# -----------------------------
# 轻量级读写锁（公平、可重入）
# -----------------------------
class RWLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0

    def gen_rlock(self):
        class _ReadLock:
            def __init__(self, rwlock):
                self.rwlock = rwlock

            def __enter__(self):
                with self.rwlock._read_ready:
                    self.rwlock._readers += 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                with self.rwlock._read_ready:
                    self.rwlock._readers -= 1
                    if self.rwlock._readers == 0:
                        self.rwlock._read_ready.notify_all()
        return _ReadLock(self)

    def gen_wlock(self):
        class _WriteLock:
            def __init__(self, rwlock):
                self.rwlock = rwlock

            def __enter__(self):
                self.rwlock._read_ready.acquire()
                while self.rwlock._readers > 0:
                    self.rwlock._read_ready.wait()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.rwlock._read_ready.release()
        return _WriteLock(self)


class MemMapVector:
    _path_locks = {}
    _path_locks_mutex = threading.Lock()

    def __init__(self, vec_file_path: str, dimension: int = 512):
        self.base_dir = Path(vec_file_path).parent
        self._dimension = dimension

        os.makedirs(self.base_dir, exist_ok=True)
        self.vec_file_path = os.path.normpath(vec_file_path)

        # 创建空文件（如果不存在）
        if not os.path.exists(self.vec_file_path):
            open(self.vec_file_path, 'wb').close()

        # 获取路径专属锁
        with MemMapVector._path_locks_mutex:
            if self.vec_file_path not in MemMapVector._path_locks:
                MemMapVector._path_locks[self.vec_file_path] = RWLock()
            self._rwlock = MemMapVector._path_locks[self.vec_file_path]

    @staticmethod
    def _close(vec_mmap):
        """显式关闭 memmap 句柄，防止 Windows 文件锁定"""
        if vec_mmap is not None:
            # 必须使用 is not None 判断，防止 Numpy 歧义报错
            if hasattr(vec_mmap, '_mmap') and vec_mmap._mmap is not None:
                try:
                    vec_mmap._mmap.close()
                except Exception:
                    pass
            del vec_mmap

    def _open_vector_mmap(self, mode='r'):
        n_vec = os.path.getsize(self.vec_file_path) // (self._dimension * 4)
        if n_vec <= 0:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.memmap(self.vec_file_path, dtype=np.float32, mode=mode, shape=(n_vec, self._dimension))

    # ──────────────── 写操作 ────────────────

    def add_vector_list(self, vector_list: List[np.ndarray]) -> List[int]:
        """添加向量列表，返回分配的 vector_id 列表"""
        vectors = np.stack(vector_list)
        return self.add_vectors(vectors)

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(f"Expected shape (N, {self._dimension})")

        n_new = vectors.shape[0]

        with self._rwlock.gen_wlock():
            start_id = os.path.getsize(self.vec_file_path) // (self._dimension * 4)
            with open(self.vec_file_path, "ab") as f:
                vectors.tofile(f)
            return list(range(start_id, start_id + n_new))

    # ──────────────── 读操作 ────────────────

    def cosine_search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        min_score: float = MIN_SCORE,
        batch_size: int = 500_000
    ) -> List[Tuple[int, float]]:
        """
        全库搜索，返回 [(vector_id, score), ...]
        """
        if min_score > 1.0:
            return []

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        query = np.ascontiguousarray(query)

        with self._rwlock.gen_rlock():
            vec_mmap = self._open_vector_mmap()
            try:
                n_total = len(vec_mmap)
                if n_total == 0:
                    return []

                all_scores = []
                for start in range(0, n_total, batch_size):
                    end = min(start + batch_size, n_total)
                    chunk = vec_mmap[start:end]
                    scores = chunk @ query
                    all_scores.append(scores)

                all_scores = np.concatenate(all_scores)
                above_threshold_mask = all_scores >= min_score

                filtered_scores = all_scores[above_threshold_mask]
                filtered_vector_ids = np.where(above_threshold_mask)[0].astype(int)

                n_filtered = len(filtered_scores)
                if n_filtered == 0:
                    return []

                k = min(top_k, n_filtered)
                top_indices = np.argpartition(-filtered_scores, k - 1)[:k]
                top_indices = top_indices[np.argsort(-filtered_scores[top_indices])]
            except Exception as e:
                raise e
            finally:
                MemMapVector._close(vec_mmap)

            return [
                (int(filtered_vector_ids[i]), float(filtered_scores[i]))
                for i in top_indices
            ]

    def cosine_search_in_vector_ids(
        self,
        query: np.ndarray,
        vector_ids: List[int],
        top_k: int = 10,
        min_score: float = MIN_SCORE,
        batch_size: int = 500_000
    ) -> List[Tuple[int, float]]:
        """
        在指定 vector_id 列表中搜索
        """
        if not vector_ids or min_score > 1.0:
            return []

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        query = np.ascontiguousarray(query)

        vector_ids = np.array(vector_ids, dtype=np.int64)
        if len(vector_ids) == 0:
            return []

        with self._rwlock.gen_rlock():
            vec_mmap = self._open_vector_mmap()
            try:
                n_total = len(vec_mmap)
                if n_total == 0:
                    return []

                # 过滤有效 vector_id（防止越界或负数）
                valid_mask = (vector_ids >= 0) & (vector_ids < n_total)
                valid_vector_ids = vector_ids[valid_mask]
                if len(valid_vector_ids) == 0:
                    return []

                # 去重并排序（提升 mmap 局部性，非必需但推荐）
                # valid_vector_ids = np.unique(valid_vector_ids)

                all_scores = []
                all_vids_batch = []

                for start in range(0, len(valid_vector_ids), batch_size):
                    end = min(start + batch_size, len(valid_vector_ids))
                    batch_vids = valid_vector_ids[start:end]
                    batch_vectors = vec_mmap[batch_vids]
                    scores = batch_vectors @ query
                    all_scores.append(scores)
                    all_vids_batch.append(batch_vids)

                combined_scores = np.concatenate(all_scores)
                combined_vector_ids = np.concatenate(all_vids_batch)

                above_threshold_mask = combined_scores >= min_score
                filtered_scores = combined_scores[above_threshold_mask]
                filtered_vector_ids = combined_vector_ids[above_threshold_mask]

                n_filtered = len(filtered_scores)
                if n_filtered == 0:
                    return []

                k = min(top_k, n_filtered)
                top_indices = np.argpartition(-filtered_scores, k - 1)[:k]
                top_indices = top_indices[np.argsort(-filtered_scores[top_indices])]
            except Exception as e:
                raise e
            finally:
                MemMapVector._close(vec_mmap)
            return [
                (int(filtered_vector_ids[i]), float(filtered_scores[i]))
                for i in top_indices
            ]

    def get_total_vectors(self) -> int:
        if not os.path.exists(self.vec_file_path):
            return 0
        return os.path.getsize(self.vec_file_path) // (self._dimension * 4)

    def get_vector_by_vector_id(self, vector_id: int) -> np.ndarray | None:
        if vector_id is None:
            return None
        with self._rwlock.gen_rlock():
            n_total = self.get_total_vectors()
            if vector_id < 0 or vector_id >= n_total:
                return None
            vec_mmap = self._open_vector_mmap()
            _vectors = vec_mmap[vector_id].copy()
            MemMapVector._close(vec_mmap)
            return _vectors

    def mark_deleted(self, vector_ids: List[int]) -> int:
        if not vector_ids:
            return 0

        # 1. 排序并去重：核心优化点
        # 排序保证了磁盘顺序访问 (Sequential I/O)，大幅提升 mmap 刷新效率
        unique_sorted_ids = sorted(list(set(vector_ids)))

        with self._rwlock.gen_wlock():
            n_total = self.get_total_vectors()

            # 2. 过滤有效范围内的 ID
            valid_ids = [vid for vid in unique_sorted_ids if 0 <= vid < n_total]
            if not valid_ids:
                return 0

            # 3. 以读写模式打开
            vec_mmap = self._open_vector_mmap(mode='r+')
            zero_vector = np.zeros(self._dimension, dtype=np.float32)

            # 4. 顺序写入零向量
            for vid in valid_ids:
                vec_mmap[vid] = zero_vector

            # 5. 强制写回磁盘（顺序写比随机写快得多）
            vec_mmap.flush()
            MemMapVector._close(vec_mmap)
            logging.debug(f"Successfully marked {len(valid_ids)} vectors as deleted.")
            return len(valid_ids)

    def compact(self) -> List[Tuple[int, int]]:
        """
        压缩向量存储：使用顺序 I/O 移除零向量。
        返回映射列表：[(old_vector_id, new_vector_id), ...]
        """
        import tempfile
        with self._rwlock.gen_wlock():
            n_total = self.get_total_vectors()
            if n_total == 0:
                return []

            # 1. 以只读方式加载当前向量 (注意：在 finally 块外定义以确保能 del)
            vec_mmap = np.memmap(
                self.vec_file_path,
                dtype=np.float32,
                mode='r',
                shape=(n_total, self._dimension)
            )

            # 2. 创建临时文件
            temp_dir = os.path.dirname(self.vec_file_path)
            fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix='.tmp')
            os.close(fd)

            # 用于存储旧 ID 到新 ID 的映射
            old_to_new: List[Tuple[int, int]] = []
            new_vector_id_counter = 0

            # 批量处理大小，控制内存峰值
            batch_size = 500_000

            try:
                with open(temp_path, 'wb') as f:
                    for old_id_start in range(0, n_total, batch_size):
                        old_id_end = min(old_id_start + batch_size, n_total)

                        # 顺序读取当前批次的向量 (I/O 性能高)
                        vectors_chunk = vec_mmap[old_id_start:old_id_end]

                        # 找到当前批次中的有效（非零）向量
                        is_nonzero_in_chunk = np.any(vectors_chunk != 0.0, axis=1)

                        valid_vectors_in_chunk = vectors_chunk[is_nonzero_in_chunk]

                        # 3. 追加写入有效向量到新文件
                        valid_vectors_in_chunk.tofile(f)

                        # 4. 更新映射表
                        old_ids_in_chunk = np.arange(old_id_start, old_id_end)
                        valid_old_ids_in_chunk = old_ids_in_chunk[is_nonzero_in_chunk]

                        for old_id in valid_old_ids_in_chunk:
                            old_to_new.append((int(old_id), new_vector_id_counter))
                            new_vector_id_counter += 1

                if new_vector_id_counter == n_total:
                    # 没有删除任何向量，避免替换，返回恒等映射
                    os.unlink(temp_path)  # 清理临时文件
                    logging.info("Compact: No deleted vectors found. Skipping replacement.")
                    return [(i, i) for i in range(n_total)]

                # 5. 在文件替换前，明确关闭 memmap 句柄，释放文件锁！
                MemMapVector._close(vec_mmap)

                # 6. 原子替换原文件
                bak_path = f"{self.vec_file_path}.{int(time.time())}.bak"
                if os.path.exists(bak_path):
                    os.remove(bak_path)
                shutil.move(self.vec_file_path, bak_path)
                os.replace(temp_path, self.vec_file_path)
                logging.info(
                    f"Compact successful. Removed {n_total - new_vector_id_counter} vectors. New total: {new_vector_id_counter}.")
                return old_to_new

            except Exception as e:
                logging.error(f"Compact failed: {e}")
                raise
            finally:
                # 确保清理临时文件（如果存在且替换失败）
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
                MemMapVector._close(vec_mmap)


    def stat(self) -> Dict:
        """
        获取向量存储的统计信息

        Returns:
            Dict: 包含各种统计信息的字典
        """
        with self._rwlock.gen_rlock():
            # 获取文件信息
            total_size = os.path.getsize(self.vec_file_path)
            n_total = total_size // (self._dimension * 4)

            # 统计零向量数量
            zero_count = 0
            deleted_vector_ids = []

            if n_total > 0:
                vec_mmap = self._open_vector_mmap()
                try:
                    # 分批次统计以提高内存效率
                    batch_size = 100000
                    for start in range(0, n_total, batch_size):
                        end = min(start + batch_size, n_total)
                        chunk = vec_mmap[start:end]

                        # 检测零向量
                        is_zero = np.all(chunk == 0.0, axis=1)
                        zero_indices = np.where(is_zero)[0] + start

                        zero_count += len(zero_indices)
                        deleted_vector_ids.extend(zero_indices.tolist())
                except Exception as e:
                    raise e
                finally:
                    MemMapVector._close(vec_mmap)
            # 计算有效向量数量
            valid_count = n_total - zero_count

            # 计算空间利用率
            if total_size > 0:
                theoretical_min_size = valid_count * self._dimension * 4
                space_efficiency = (theoretical_min_size / total_size) * 100
            else:
                space_efficiency = 0.0

            # 计算文件大小（人类可读格式）
            def format_size(size_bytes):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024.0 or unit == 'GB':
                        return f"{size_bytes:.2f} {unit}"
                    size_bytes /= 1024.0

            return {
                'file_path': self.vec_file_path,
                'dimension': self._dimension,
                'total_vectors': n_total,
                'valid_vectors': valid_count,
                'deleted_vectors': zero_count,
                'deleted_vector_ids': deleted_vector_ids[:100],  # 只显示前100个，避免输出过大
                'deleted_count_full': zero_count,
                'file_size': total_size,
                'file_size_human': format_size(total_size),
                'space_efficiency_percent': round(space_efficiency, 2),
                'dtype': 'float32',
                'bytes_per_vector': self._dimension * 4,
                'theoretical_min_size': format_size(valid_count * self._dimension * 4) if valid_count > 0 else '0 B',
                'can_reclaim_space': zero_count > 0,
                'reclaimable_space': format_size(zero_count * self._dimension * 4) if zero_count > 0 else '0 B',
                'lock_status': 'Read lock acquired'
            }

if __name__ == '__main__':
    mmap_vector = MemMapVector(r"xxx\content.vec")
    print(mmap_vector.stat())
    mmap_vector.compact()
    print(mmap_vector.stat())
