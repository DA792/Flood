import argparse
import random
import pandas as pd
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index.bottomClusters import stratified_sampling, read_query, get_sorted_dimensions
from utils import *
import numpy as np
import math
from bisect import bisect_left
from collections import namedtuple

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument('-d', help='dataset', default='foursquare2')
parser.add_argument('-ls', type=int, choices=[100, 500, 1000, 5000, 10000], help='lower bound selectivity', default=100)
parser.add_argument('-hs', type=int, choices=[200, 1000, 2000, 10000, 20000], help='higher bound selectivity',
                    default=200)
parser.add_argument('-r', type=float, choices=[0.005, 0.01, 0.05, 0.1, 0.5, 1], help='query range ratio', default=0.05)
parser.add_argument('-isk', type=int, choices=[0, 1], help='skew distribution', default=0)
parser.add_argument('-ig', type=int, choices=[0, 1], help='gaussian distribution', default=0)
parser.add_argument('-im', type=int, choices=[0, 1], help='mixed distribution', default=0)
parser.add_argument('-sr', type=int, help='sample rate', default=100)
parser.add_argument('-et', type=float, help='PLM error threshold', default=0.01)

# ... 其他函数保持不变 ...

# 定义分段线性模型的片段结构
Segment = namedtuple('Segment', ['start_val', 'start_idx', 'slope', 'intercept'])


class PiecewiseLinearModel:
    """分段线性模型（PLM）实现，用于CDF建模和快速搜索"""

    def __init__(self, values, indices, error_threshold=0.01):
        """
        初始化PLM模型

        参数:
            values: 排序后的值列表
            indices: 对应的索引
            error_threshold: 平均误差阈值 δ
        """
        self.segments = []
        self.values = values
        self.indices = indices
        self.error_threshold = error_threshold
        self._build_model()

    def _build_model(self):
        """使用贪心算法构建分段线性模型"""
        if not self.values:
            return

        n = len(self.values)
        current_slice_start = 0

        while current_slice_start < n:
            # 开始一个新片段
            segment_start_val = self.values[current_slice_start]
            segment_start_idx = self.indices[current_slice_start]

            # 初始化为包含单个点的线段
            best_slope = 0
            best_intercept = segment_start_idx

            # 尝试扩展当前片段
            current_slice_end = current_slice_start + 1
            while current_slice_end < n:
                # 计算线性回归参数（保证下界性质）
                x_points = self.values[current_slice_start:current_slice_end + 1]
                y_points = self.indices[current_slice_start:current_slice_end + 1]

                # 为保证下界特性，我们找到最小斜率使所有点在线段上方
                if len(x_points) > 1:
                    slopes = []
                    for i in range(1, len(x_points)):
                        if x_points[i] != x_points[0]:  # 避免除以零
                            slopes.append((y_points[i] - y_points[0]) / (x_points[i] - x_points[0]))

                    if slopes:
                        slope = min(slopes)  # 确保下界
                        intercept = y_points[0] - slope * x_points[0]
                    else:
                        slope = 0
                        intercept = y_points[0]
                else:
                    slope = 0
                    intercept = y_points[0]

                # 计算预测值
                predictions = [max(0, int(slope * x + intercept)) for x in x_points]

                # 计算误差
                errors = [y - p for y, p in zip(y_points, predictions)]
                avg_error = sum(errors) / len(errors) if errors else 0

                # 如果平均误差超过阈值，结束当前片段
                if avg_error > self.error_threshold:
                    break

                best_slope = slope
                best_intercept = intercept
                current_slice_end += 1

            # 保存当前片段
            self.segments.append(Segment(
                start_val=segment_start_val,
                start_idx=segment_start_idx,
                slope=best_slope,
                intercept=best_intercept
            ))

            # 移动到下一个片段的起点
            current_slice_start = current_slice_end

    def predict(self, value):
        """预测给定值在排序列表中的位置"""
        if not self.segments:
            return 0

        # 找到合适的片段
        segment_idx = 0
        for i, segment in enumerate(self.segments):
            if i == len(self.segments) - 1 or value < self.segments[i + 1].start_val:
                segment_idx = i
                break

        segment = self.segments[segment_idx]
        predicted_idx = max(0, int(segment.slope * value + segment.intercept))

        return predicted_idx

    def search_with_refinement(self, value, data, key_func=lambda x: x):
        """使用模型预测位置，然后进行细化搜索"""
        if not self.segments or not data:
            return 0, 0

        # 1. 模型预测初始位置
        predicted_idx = self.predict(value)
        predicted_idx = min(len(data) - 1, max(0, predicted_idx))

        # 2. 向左搜索找到第一个小于或等于目标值的位置
        left_idx = predicted_idx
        while left_idx > 0 and key_func(data[left_idx]) > value:
            left_idx -= 1

        # 如果最左侧元素仍大于目标值，则目标值小于所有元素
        if left_idx == 0 and key_func(data[left_idx]) > value:
            return 0, 0  # 范围为空

        # 3. 继续向左搜索，找到第一个小于目标值的位置
        while left_idx > 0 and key_func(data[left_idx]) == value:
            left_idx -= 1

        if key_func(data[left_idx]) < value:
            left_idx += 1  # 调整到第一个等于目标值的位置

        # 4. 向右搜索找到第一个大于目标值的位置
        right_idx = predicted_idx
        while right_idx < len(data) - 1 and key_func(data[right_idx]) <= value:
            right_idx += 1

        # 5. 继续向右搜索，找到最后一个等于目标值的位置
        while right_idx < len(data) - 1 and key_func(data[right_idx]) == value:
            right_idx += 1

        if key_func(data[right_idx]) > value:
            right_idx -= 1  # 调整到最后一个等于目标值的位置

        # 确保返回有效的范围
        if left_idx > right_idx:
            return 0, 0

        return left_idx, right_idx + 1  # 返回闭开区间 [left, right)


class BottomCluster:
    def __init__(self, pl, qs, sr=None, num_partitions=10, error_threshold=0.01):
        self.points = pl  # 原始数据点集合
        self.queries = qs
        self.num_partitions = num_partitions
        self.error_threshold = error_threshold
        self.partitions, self.mbrs, self.sort_dim = self._flat_partition_with_mbr()  # 执行扁平划分并计算MBR
        self.models = self._build_partition_models()  # 为每个分区构建PLM模型

    def _flat_partition_with_mbr(self):
        """扁平化划分并计算每个分区的MBR"""
        if not self.points:
            return [[] for _ in range(self.num_partitions)], [None] * self.num_partitions

        # 计算X和Y维度的选择性
        x_values = [p.x for p in self.points]
        y_values = [p.y for p in self.points]
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        
        # 计算每个维度的选择性（范围越大，选择性越低）
        x_selectivity = 1.0 / x_range if x_range > 0 else float('inf')
        y_selectivity = 1.0 / y_range if y_range > 0 else float('inf')
        
        # 选择选择性高的维度作为排序维度
        sort_dim = 'x' if x_selectivity > y_selectivity else 'y'
        print(f"\n维度选择性分析:")
        print(f"X维度选择性: {x_selectivity:.6f}")
        print(f"Y维度选择性: {y_selectivity:.6f}")
        print(f"选择 {sort_dim.upper()} 维度作为排序维度")

        # 按选择的维度排序
        sorted_points = sorted(self.points, key=lambda p: getattr(p, sort_dim))
        partition_size = max(1, len(sorted_points) // self.num_partitions)
        partitions = []
        mbrs = []  # 存储每个分区的MBR
        split_points = []  # 存储分割点

        for i in range(self.num_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < self.num_partitions - 1 else len(sorted_points)
            partition = sorted_points[start:end]
            partitions.append(partition)

            # 记录分割点
            if i < self.num_partitions - 1:
                split_points.append(getattr(sorted_points[end-1], sort_dim))

            # 计算分区的MBR
            if partition:
                x_values = [p.x for p in partition]
                y_values = [p.y for p in partition]
                mbr = {
                    'x_min': min(x_values),
                    'x_max': max(x_values),
                    'y_min': min(y_values),
                    'y_max': max(y_values)
                }
                mbrs.append(mbr)
            else:
                mbrs.append(None)

        return partitions, mbrs, sort_dim

    def _build_partition_models(self):
        """为每个分区的排序维度建立PLM模型"""
        models = []

        for partition in self.partitions:
            if not partition:
                models.append({'x': None, 'y': None})
                continue

            # 提取X维度的值和索引
            sorted_x = sorted((p.x, i) for i, p in enumerate(partition))
            x_values = [x for x, _ in sorted_x]
            x_indices = [i for _, i in sorted_x]
            x_model = PiecewiseLinearModel(x_values, x_indices, self.error_threshold)

            # 提取Y维度的值和索引
            sorted_y = sorted((p.y, i) for i, p in enumerate(partition))
            y_values = [y for y, _ in sorted_y]
            y_indices = [i for _, i in sorted_y]
            y_model = PiecewiseLinearModel(y_values, y_indices, self.error_threshold)

            models.append({'x': x_model, 'y': y_model})

        return models

    def intersects_with_query(self, mbr, query):
        """检查MBR是否与查询相交"""
        return (query.min_x <= mbr['x_max'] and query.max_x >= mbr['x_min'] and
                query.min_y <= mbr['y_max'] and query.max_y >= mbr['y_min'])

    def query_range_with_detail(self, query):
        """使用PLM模型执行范围查询，返回详细的分区和细化信息"""
        result = []
        intersecting_partitions = []
        refined_results_per_partition = []

        print(f"\n执行查询: ({query.min_x}, {query.min_y}) - ({query.max_x}, {query.max_y})")

        # 分析相交的分区
        for i, partition in enumerate(self.partitions):
            if not partition or not self.mbrs[i]:
                continue

            # 检查分区是否与查询相交
            mbr = self.mbrs[i]
            if self.intersects_with_query(mbr, query):
                print(f"分区 {i} 与查询相交")
                print(f"MBR: X[{mbr['x_min']:.2f}, {mbr['x_max']:.2f}], Y[{mbr['y_min']:.2f}, {mbr['y_max']:.2f}]")
                
                intersecting_partitions.append({
                    'partition_id': i,
                    'mbr': mbr,
                    'point_count': len(partition)
                })

                # 使用排序维度模型找到范围起点和终点
                sort_model = self.models[i][self.sort_dim]
                if sort_model:
                    # 使用带有细化的搜索，分别搜索最小值和最大值
                    min_val = getattr(query, f'min_{self.sort_dim}')
                    max_val = getattr(query, f'max_{self.sort_dim}')
                    min_idx, min_end = sort_model.search_with_refinement(min_val, partition,
                                                                       key_func=lambda p: getattr(p, self.sort_dim))
                    max_idx, max_end = sort_model.search_with_refinement(max_val, partition,
                                                                       key_func=lambda p: getattr(p, self.sort_dim))
                    
                    # 合并范围，确保范围有效
                    start_idx = min_idx
                    end_idx = max_end
                    if start_idx > end_idx:
                        start_idx, end_idx = 0, 0
                    print(f"{self.sort_dim.upper()}维度搜索范围: [{start_idx}, {end_idx}]")

                    # 候选点（排序维度过滤后）
                    candidates = partition[start_idx:end_idx]
                    print(f"{self.sort_dim.upper()}维度过滤后的候选点数量: {len(candidates)}")

                    # 进一步用另一个维度过滤
                    other_dim = 'y' if self.sort_dim == 'x' else 'x'
                    matching_points = []
                    for point in candidates:
                        min_val = getattr(query, f'min_{other_dim}')
                        max_val = getattr(query, f'max_{other_dim}')
                        if min_val <= getattr(point, other_dim) <= max_val:
                            matching_points.append(point)
                            result.append(point)
                    print(f"{other_dim.upper()}维度过滤后的匹配点数量: {len(matching_points)}")

                    # 记录细化后的结果
                    if matching_points:
                        refined_range = {
                            'x_min': query.min_x if self.sort_dim == 'y' else min(p.x for p in matching_points),
                            'x_max': query.max_x if self.sort_dim == 'y' else max(p.x for p in matching_points),
                            'y_min': query.min_y if self.sort_dim == 'x' else min(p.y for p in matching_points),
                            'y_max': query.max_y if self.sort_dim == 'x' else max(p.y for p in matching_points)
                        }
                    else:
                        refined_range = None

                    refined_results_per_partition.append({
                        'partition_id': i,
                        'candidates_after_sort_filter': len(candidates),
                        'final_matching_points': len(matching_points),
                        'sample_points': matching_points[:min(3, len(matching_points))],
                        'refined_range': refined_range
                    })

        print(f"查询结果总数: {len(result)}")
        return result, intersecting_partitions, refined_results_per_partition

    def traditional_query(self, query):
        """传统方法执行范围查询（用于对比）"""
        result = []
        for point in self.points:
            if (query.min_x <= point.x <= query.max_x and
                    query.min_y <= point.y <= query.max_y):
                result.append(point)
        return result

    def print_partition_info(self):
        """打印分区信息，包括分割点和MBR"""
        print("\n分区信息:")
        print(f"总分区数: {self.num_partitions}")
        print(f"每个分区的平均点数: {len(self.points) / self.num_partitions:.2f}")
        
        # 打印分割点
        sorted_points = sorted(self.points, key=lambda p: p.x)
        partition_size = max(1, len(sorted_points) // self.num_partitions)
        print("\n分割点位置:")
        for i in range(self.num_partitions - 1):
            split_idx = (i + 1) * partition_size - 1
            if split_idx < len(sorted_points):
                print(f"分区 {i} 和 {i+1} 之间的分割点: X = {sorted_points[split_idx].x:.2f}")

        # 打印每个分区的详细信息
        print("\n各分区详细信息:")
        for i, (partition, mbr) in enumerate(zip(self.partitions, self.mbrs)):
            print(f"\n分区 {i}:")
            print(f"  包含点数: {len(partition)}")
            if mbr:
                print(f"  MBR范围: X[{mbr['x_min']:.2f}, {mbr['x_max']:.2f}], Y[{mbr['y_min']:.2f}, {mbr['y_max']:.2f}]")
                if partition:
                    print("  示例点:")
                    for j, point in enumerate(partition[:3]):
                        print(f"    点 {j}: ({point.x}, {point.y})")
                    if len(partition) > 3:
                        print(f"    ... 还有 {len(partition) - 3} 个点未显示")

    def analyze_query_intersection(self, query_set):
        """分析查询集与分区的相交情况"""
        if not query_set:
            print("查询集为空，无法分析相交情况")
            return

        print("\n查询集与分区相交分析:")
        total_matching_points = 0

        for q_idx, query in enumerate(query_set):
            print(f"\n查询 {q_idx}: ({query.min_x}, {query.min_y}) - ({query.max_x}, {query.max_y})")
            query_matching_points = 0

            # 执行带详细信息的查询
            results, intersecting_parts, refined_results = self.query_range_with_detail(query)
            query_matching_points = len(results)
            total_matching_points += query_matching_points

            # 输出查询结果总数
            print(f"  查询结果总数: {query_matching_points} 个数据点")

            # 显示相交的分区
            print(f"  与查询相交的分区数量: {len(intersecting_parts)}")
            for part in intersecting_parts:
                print(f"    分区 {part['partition_id']}: 包含 {part['point_count']} 个点")
                mbr = part['mbr']
                print(f"      MBR: X[{mbr['x_min']:.2f}, {mbr['x_max']:.2f}], Y[{mbr['y_min']:.2f}, {mbr['y_max']:.2f}]")

            # 显示细化后的结果
            print("  细化后各分区的匹配情况:")
            for ref in refined_results:
                print(f"    分区 {ref['partition_id']}:")
                print(f"      {self.sort_dim.upper()}维度过滤后的候选点数量: {ref['candidates_after_sort_filter']}")
                print(f"      最终匹配的点数量: {ref['final_matching_points']}")
                
                # 显示细化后的范围
                if ref['refined_range']:
                    print(f"      细化后的范围: X[{ref['refined_range']['x_min']:.2f}, {ref['refined_range']['x_max']:.2f}], "
                          f"Y[{ref['refined_range']['y_min']:.2f}, {ref['refined_range']['y_max']:.2f}]")
                    print("      示例匹配点:")
                    for pt in ref['sample_points']:
                        print(f"        ({pt.x}, {pt.y})")

        print(f"\n所有查询的总匹配点数: {total_matching_points}")


if __name__ == '__main__':
    args = parser.parse_args()
    is_skew = True if args.isk == 1 else False
    is_gau = True if args.ig == 1 else False
    is_mixed = True if args.im == 1 else False
    sample_rate = args.sr
    error_threshold = args.et
    print('使用简化数据集进行演示')

    # 读取简化数据集
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'data', 'data', 'simple_foursquare.csv')
    try:
        # 使用空格作为分隔符，并处理多个空格
        data = pd.read_csv(data_file, header=None, delim_whitespace=True)
        data_set = [Point(row[0], row[1]) for _, row in data.iterrows()]
        print('数据集大小:', len(data_set))
        print('数据集范围:')
        x_values = [p.x for p in data_set]
        y_values = [p.y for p in data_set]
        print(f'X: [{min(x_values)}, {max(x_values)}]')
        print(f'Y: [{min(y_values)}, {max(y_values)}]')
        print('\n数据集点:')
        for i, point in enumerate(data_set):
            print(f'点 {i}: ({point.x}, {point.y})')
    except Exception as e:
        print(f"读取数据集时出错: {e}")
        data_set = []

    # 读取简化查询集
    query_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'query', 'training', 'simple_queries.csv')
    try:
        query = pd.read_csv(query_file, header=None)
        query_set = []
        for _, row in query.iterrows():
            q = Query(row[0], row[1], row[2], row[3])
            query_set.append(q)
        print('\n查询集大小:', len(query_set))
        print('查询范围:')
        for i, q in enumerate(query_set):
            print(f'查询 {i}: ({q.min_x}, {q.min_y}) - ({q.max_x}, {q.max_y})')
    except Exception as e:
        print(f"读取查询文件时出错: {e}")
        query_set = []

    if data_set and query_set:
        # 创建 BottomCluster 对象，使用4个分区
        root = BottomCluster(data_set, query_set, num_partitions=4, error_threshold=error_threshold)

        # 打印分区信息
        root.print_partition_info()

        # 分析查询集与分区的相交情况
        root.analyze_query_intersection(query_set)
    else:
        print("数据集或查询集为空，无法继续...")
