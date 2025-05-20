import numpy as np
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import math
from bisect import bisect_right
import time
from cost_model import FloodCostModel, QueryStats, CostWeights


@dataclass
class Point:
    """表示d维空间中的一个点"""
    coordinates: List[float]


@dataclass
class Query:
    """表示一个范围查询"""
    min_bounds: List[float]  # 每个维度的最小值
    max_bounds: List[float]  # 每个维度的最大值


class PLMSegment:
    """PLM模型中的线性段"""

    def __init__(self, start_value: float, start_index: int, end_value: float, end_index: int):
        self.start_value = start_value
        self.start_index = start_index
        self.end_value = end_value
        self.end_index = end_index
        # 避免除零错误
        if end_value == start_value:
            self.slope = 0
        else:
            self.slope = (end_index - start_index) / (end_value - start_value)
        self.intercept = start_index - self.slope * start_value

    def predict(self, value: float) -> float:
        """预测给定值的索引位置"""
        return self.slope * value + self.intercept


class PLMModel:
    """分段线性模型(PLM)实现"""

    def __init__(self, error_threshold: float = 0.1):
        self.error_threshold = error_threshold
        self.cell_segments: Dict[Tuple[int, ...], List[PLMSegment]] = {}  # 每个单元格的段
        self.cell_boundaries: Dict[Tuple[int, ...], List[float]] = {}  # 每个单元格的边界值

    def fit(self, values: List[float], indices: List[int], cell_id: Tuple[int, ...]):
        """训练PLM模型
        使用贪心算法将数据分成多个段，每个段的平均误差不超过阈值
        参数:
            values: 该单元格内的值
            indices: 对应的索引
            cell_id: 单元格标识
        """
        if not values:
            return

        # 对值进行排序
        sorted_pairs = sorted(zip(values, indices))
        values, indices = zip(*sorted_pairs)
        values = list(values)
        indices = list(indices)

        current_segment_start = 0
        current_segment_values = []
        current_segment_indices = []
        segments = []
        boundaries = []

        for i in range(len(values)):
            current_segment_values.append(values[i])
            current_segment_indices.append(indices[i])

            # 计算当前段的平均误差
            if len(current_segment_values) > 1:
                segment = PLMSegment(
                    current_segment_values[0],
                    current_segment_indices[0],
                    current_segment_values[-1],
                    current_segment_indices[-1]
                )

                # 计算平均误差
                total_error = 0
                for v, d in zip(current_segment_values, current_segment_indices):
                    p = segment.predict(v)
                    total_error += d - p  # 使用下界性质

                avg_error = total_error / len(current_segment_values)

                # 如果平均误差超过阈值，开始新的段
                if avg_error > self.error_threshold:
                    # 保存当前段
                    segments.append(segment)
                    boundaries.append(values[current_segment_start])

                    # 开始新的段
                    current_segment_start = i
                    current_segment_values = [values[i]]
                    current_segment_indices = [indices[i]]

        # 添加最后一个段
        if current_segment_values:
            segments.append(PLMSegment(
                current_segment_values[0],
                current_segment_indices[0],
                current_segment_values[-1],
                current_segment_indices[-1]
            ))
            boundaries.append(values[current_segment_start])

        # 保存该单元格的段和边界
        self.cell_segments[cell_id] = segments
        self.cell_boundaries[cell_id] = boundaries

    def predict(self, value: float, cell_id: Tuple[int, ...]) -> float:
        """使用PLM模型预测索引位置"""
        if cell_id not in self.cell_segments:
            return 0.0

        segments = self.cell_segments[cell_id]
        boundaries = self.cell_boundaries[cell_id]

        # 使用二分查找找到合适的段
        segment_idx = bisect_right(boundaries, value) - 1
        if segment_idx < 0:
            segment_idx = 0
        elif segment_idx >= len(segments):
            segment_idx = len(segments) - 1

        # 使用选定的段进行预测
        prediction = segments[segment_idx].predict(value)
        return max(0, min(1, prediction))  # 归一化到[0,1]范围


class RMIModel:
    """递归模型索引(RMI)实现，用于建模属性的CDF"""

    def __init__(self, num_models: int = 2):
        self.num_models = num_models
        self.models = []  # 存储每个阶段的模型
        self.stage_boundaries = []  # 存储每个阶段的边界值
        self.min_value = 0.0
        self.max_value = 0.0
        self.total_points = 0
        self.cdf_boundaries = []  # 存储CDF边界值

    def predict(self, value: float) -> float:
        """使用RMI模型预测CDF值"""
        if not self.models:
            return 0.0

        # 找到合适的阶段
        stage_idx = bisect_right(self.stage_boundaries, value) - 1
        if stage_idx < 0:
            stage_idx = 0
        elif stage_idx >= len(self.models):
            stage_idx = len(self.models) - 1

        # 使用该阶段的模型进行预测
        slope, intercept = self.models[stage_idx]
        prediction = slope * value + intercept
        return max(0, min(1, prediction))  # 归一化到[0,1]范围

    def fit(self, values: List[float], indices: List[int]):
        """训练RMI模型来建模CDF
        参数:
            values: 输入值
            indices: 对应的索引
        """
        if not values:
            return

        # 保存数据范围信息
        self.min_value = min(values)
        self.max_value = max(values)
        self.total_points = len(values)

        # 对值进行排序
        sorted_pairs = sorted(zip(values, indices))
        values, indices = zip(*sorted_pairs)
        values = list(values)
        indices = list(indices)

        # 计算每个阶段的数据范围
        total_range = self.max_value - self.min_value
        stage_size = total_range / self.num_models

        # 为每个阶段训练一个线性模型
        for i in range(self.num_models):
            stage_min = self.min_value + i * stage_size
            stage_max = stage_min + stage_size

            # 获取该阶段的数据点
            stage_values = []
            stage_indices = []
            for v, idx in zip(values, indices):
                if stage_min <= v <= stage_max:
                    stage_values.append(v)
                    stage_indices.append(idx)

            if stage_values:
                # 计算线性模型的参数
                if len(stage_values) > 1:
                    # 计算CDF值（归一化到[0,1]范围）
                    cdf_values = [idx / self.total_points for idx in stage_indices]
                    slope = (cdf_values[-1] - cdf_values[0]) / (stage_values[-1] - stage_values[0])
                    intercept = cdf_values[0] - slope * stage_values[0]
                else:
                    slope = 0
                    intercept = stage_indices[0] / self.total_points

                self.models.append((slope, intercept))
                self.stage_boundaries.append(stage_min)
            else:
                # 如果没有数据点，使用前一个阶段的模型
                if self.models:
                    self.models.append(self.models[-1])
                else:
                    self.models.append((0, 0))
                self.stage_boundaries.append(stage_min)

    def compute_cdf_boundaries(self, num_columns: int):
        """计算等深度划分的CDF边界值"""
        self.cdf_boundaries = []
        for i in range(num_columns + 1):
            self.cdf_boundaries.append(i / num_columns)

    def get_column_index(self, value: float, num_columns: int) -> int:
        """根据CDF值确定点应该被分配到哪个列"""
        cdf_value = self.predict(value)
        return min(int(cdf_value * num_columns), num_columns - 1)

    def get_column_boundaries(self, num_columns: int) -> List[float]:
        """获取等深度划分的列边界值"""
        if not self.cdf_boundaries or len(self.cdf_boundaries) != num_columns + 1:
            self.compute_cdf_boundaries(num_columns)
        return self.cdf_boundaries


class LayoutOptimizer:
    """布局优化算法实现"""

    def __init__(self, dataset: List[Point], queries: List[Query],
                 sample_rate: float = 0.1, num_partitions: int = 3,
                 error_threshold: float = 0.1):
        self.dataset = dataset
        self.queries = queries
        self.sample_rate = sample_rate
        self.num_partitions = num_partitions
        self.dimensions = len(dataset[0].coordinates) if dataset else 0
        self.plm_models = [PLMModel(error_threshold) for _ in range(self.dimensions)]
        self.rmi_models = [RMIModel() for _ in range(self.dimensions)]

    def sample_data(self) -> Tuple[List[Point], List[Query]]:
        """采样数据和查询"""
        # 采样数据点
        sample_size = int(len(self.dataset) * self.sample_rate)
        sampled_data = random.sample(self.dataset, sample_size)

        # 采样查询
        sample_size = int(len(self.queries) * self.sample_rate)
        sampled_queries = random.sample(self.queries, sample_size)

        return sampled_data, sampled_queries

    def get_cell_id(self, point: Point, grid_sizes: List[float], dim_order: List[int]) -> Tuple[int, ...]:
        """获取点所在的单元格ID，使用CDF进行列划分"""
        cell_id = []
        for i, dim in enumerate(dim_order[:-1]):
            # 使用RMI模型预测CDF值并确定列索引
            col_idx = self.rmi_models[dim].get_column_index(point.coordinates[dim], self.num_partitions)
            cell_id.append(col_idx)
        return tuple(cell_id)

    def flatten_data(self, data: List[Point], layout: Tuple[List[int], List[int]]) -> List[Point]:
        """使用CDF进行等深度划分
        参数:
            data: 原始数据点
            layout: 布局信息 (维度顺序, 列数)
        返回:
            展平后的数据点
        """
        dim_order, col_counts = layout

        # 使用训练好的RMI模型进行等深度划分
        flattened_data = []
        for point in data:
            flattened_coords = []
            for dim in range(self.dimensions):
                if dim == dim_order[-1]:  # 如果是排序维度，保持原值
                    flattened_coords.append(point.coordinates[dim])
                else:
                    # 使用CDF值进行等深度划分
                    dim_idx = dim_order.index(dim)
                    num_cols = col_counts[dim_idx]
                    col_idx = self.rmi_models[dim].get_column_index(point.coordinates[dim], num_cols)
                    # 将列索引映射回[0,1]范围
                    flattened_coords.append(col_idx / num_cols)
            flattened_data.append(Point(flattened_coords))

        return flattened_data

    def calculate_selectivity(self, queries: List[Query], dim: int) -> float:
        """计算给定维度的平均选择性"""
        total_selectivity = 0
        for query in queries:
            range_size = query.max_bounds[dim] - query.min_bounds[dim]
            total_selectivity += range_size
        return total_selectivity / len(queries) if queries else 0

    def get_dimension_order(self, sort_dim: int) -> List[int]:
        """根据选择性生成维度顺序"""
        # 计算每个维度的选择性
        selectivities = [(dim, self.calculate_selectivity(self.queries, dim))
                         for dim in range(self.dimensions) if dim != sort_dim]

        # 按选择性从高到低排序
        sorted_dims = [dim for dim, _ in sorted(selectivities, key=lambda x: x[1], reverse=True)]

        # 将排序维度放在最后
        return sorted_dims + [sort_dim]

    def refine_candidates(self, data: List[Point], query: Query,
                          candidates: List[Point]) -> List[Point]:
        """细化过程：对候选点进行精确过滤
        参数:
            data: 原始数据集
            query: 查询范围
            candidates: 候选点列表
        返回:
            细化后的点列表
        """
        refined_points = []
        for point in candidates:
            # 检查点是否在查询范围内
            in_range = True
            for dim in range(self.dimensions):
                if not (query.min_bounds[dim] <= point.coordinates[dim] <= query.max_bounds[dim]):
                    in_range = False
                    break
            if in_range:
                refined_points.append(point)
        return refined_points

    def estimate_query_time(self, data: List[Point], query: Query,
                            layout: Tuple[List[int], List[int]]) -> Tuple[float, Dict[str, float]]:
        """估计给定布局下的查询时间
        成本模型: Time(D,q,L) = wp*Nc + wr*Nc + ws*Ns
        其中:
        - wp: 投影时间常数，取决于数据集和查询特征
        - wr: 细化时间常数，取决于数据集和查询特征
        - ws: 扫描时间常数，取决于数据集和查询特征
        - Nc: 查询矩形内的单元格数量
        - Ns: 需要扫描的点数
        """
        dim_order, col_counts = layout
        
        # 1. 计算查询范围内的单元格数量 (Nc)
        cells_to_scan = 1
        for i, dim in enumerate(dim_order[:-1]):
            # 使用RMI模型预测查询范围的CDF值
            min_cdf = max(0.0, min(1.0, self.rmi_models[dim].predict(query.min_bounds[dim])))
            max_cdf = max(0.0, min(1.0, self.rmi_models[dim].predict(query.max_bounds[dim])))
            
            # 计算查询范围内的列数
            min_col = int(min_cdf * col_counts[i])
            max_col = int(max_cdf * col_counts[i])
            cells_in_dim = max(1, max_col - min_col + 1)  # 确保至少有一个单元格
            cells_to_scan *= cells_in_dim
        
        # 2. 估计需要扫描的点数 (Ns)
        # 使用网格密度来估计
        total_cells = max(1, np.prod(col_counts))  # 确保至少有一个单元格
        avg_points_per_cell = len(data) / total_cells
        estimated_points = max(1, cells_to_scan * avg_points_per_cell)  # 确保至少有一个点
        
        # 3. 计算成本模型参数
        # 这些参数应该基于数据集和查询的特征来动态调整
        query_size = max(0.1, np.prod([max(0.1, query.max_bounds[i] - query.min_bounds[i]) 
                                     for i in range(self.dimensions)]))
        data_density = max(0.1, len(data) / (self.dimensions * 100))  # 假设数据范围是[0,100]
        
        # 动态调整权重，确保最小成本
        wp = max(0.1, 0.1 * (1 + query_size))  # 查询范围越大，投影开销越大
        wr = max(0.1, 0.2 * (1 + data_density))  # 数据密度越大，细化开销越大
        ws = max(0.1, 0.05 * (1 + 1/query_size))  # 查询范围越小，扫描开销越大
        
        # 4. 计算总成本
        projection_cost = wp * cells_to_scan  # 投影成本
        refinement_cost = wr * cells_to_scan  # 细化成本
        scan_cost = ws * estimated_points     # 扫描成本
        
        total_cost = projection_cost + refinement_cost + scan_cost
        
        # 返回总成本和详细的成本组成
        cost_details = {
            'total_cost': total_cost,
            'projection_cost': projection_cost,
            'refinement_cost': refinement_cost,
            'scan_cost': scan_cost,
            'cells_to_scan': cells_to_scan,
            'estimated_points': estimated_points,
            'wp': wp,
            'wr': wr,
            'ws': ws,
            'query_size': query_size,
            'data_density': data_density
        }
        
        return total_cost, cost_details

    def execute_query(self, query: Query, layout: Tuple[List[int], List[int]]) -> Tuple[List[Point], int, int]:
        """执行查询，包括投影、细化和扫描过程
        返回:
            (结果点列表, 扫描单元格数, 扫描点数)
        """
        dim_order, col_counts = layout
        sort_dim = dim_order[-1]  # 获取排序维度

        # 1. 投影过程：找到查询范围内的单元格
        projection_start = time.time()
        cells_to_scan = []
        for i, dim in enumerate(dim_order[:-1]):
            # 使用RMI模型预测查询范围的CDF值
            min_cdf = self.rmi_models[dim].predict(query.min_bounds[dim])
            max_cdf = self.rmi_models[dim].predict(query.max_bounds[dim])
            
            # 将CDF值转换为列索引
            min_col = int(min_cdf * col_counts[i])
            max_col = int(max_cdf * col_counts[i])
            cells_to_scan.append((min_col, max_col))

        # 计算查询范围内的单元格数量 (Nc)
        Nc = 1
        for min_col, max_col in cells_to_scan:
            Nc *= (max_col - min_col + 1)
        
        # 模拟投影阶段的延迟
        time.sleep(0.001 * Nc)  # 每个单元格0.001秒的延迟
        projection_time = time.time() - projection_start

        # 2. 获取候选点
        refinement_start = time.time()
        candidates = []
        for point in self.dataset:
            in_candidate_cells = True
            for i, (dim, (min_col, max_col)) in enumerate(zip(dim_order[:-1], cells_to_scan)):
                col_idx = self.rmi_models[dim].get_column_index(point.coordinates[dim], col_counts[i])
                if not (min_col <= col_idx <= max_col):
                    in_candidate_cells = False
                    break
            if in_candidate_cells:
                candidates.append(point)
        
        # 模拟细化阶段的延迟
        time.sleep(0.0005 * len(candidates))  # 每个候选点0.0005秒的延迟
        refinement_time = time.time() - refinement_start

        # 3. 使用PLM模型进行细化
        scan_start = time.time()
        refined_points = []
        # 创建单元格字典来存储每个单元格中的点
        cells = {}
        for point in candidates:
            # 获取点所在的单元格坐标
            cell_coords = []
            for i, (dim, (min_col, max_col)) in enumerate(zip(dim_order[:-1], cells_to_scan)):
                col_idx = self.rmi_models[dim].get_column_index(point.coordinates[dim], col_counts[i])
                cell_coords.append(col_idx)
            
            # 将点添加到对应的单元格中
            cell_key = tuple(cell_coords)
            if cell_key not in cells:
                cells[cell_key] = []
            cells[cell_key].append(point)

        # 对每个单元格中的点进行细化
        for cell_key, points in cells.items():
            # 首先对非排序维度进行范围检查
            refined_cell_points = points.copy()
            for dim in range(self.dimensions):
                if dim != sort_dim:  # 非排序维度使用直接范围检查
                    refined_cell_points = [p for p in refined_cell_points 
                                         if query.min_bounds[dim] <= p.coordinates[dim] <= query.max_bounds[dim]]
            
            # 对排序维度使用PLM模型进行细化
            if refined_cell_points:
                # 获取排序维度的值
                sort_values = [p.coordinates[sort_dim] for p in refined_cell_points]
                # 使用PLM模型预测
                predictions = [self.plm_models[sort_dim].predict(v, cell_key) for v in sort_values]
                # 根据预测结果进行细化
                refined_cell_points = [p for p, pred in zip(refined_cell_points, predictions)
                                    if query.min_bounds[sort_dim] <= p.coordinates[sort_dim] <= query.max_bounds[sort_dim]]
            
            refined_points.extend(refined_cell_points)
        
        # 模拟扫描阶段的延迟
        time.sleep(0.0001 * len(refined_points))  # 每个结果点0.0001秒的延迟
        scan_time = time.time() - scan_start

        # 计算实际的权重
        wp = projection_time / Nc if Nc > 0 else 0
        wr = refinement_time / Nc if Nc > 0 else 0
        ws = scan_time / len(refined_points) if len(refined_points) > 0 else 0

        # 输出查询统计信息
        Ns = len(refined_points)
        print(f"\n查询统计信息:")
        print(f"查询范围: ({query.min_bounds[0]:.1f}, {query.min_bounds[1]:.1f}) - ({query.max_bounds[0]:.1f}, {query.max_bounds[1]:.1f})")
        print(f"扫描单元格数 (Nc): {Nc}")
        print(f"扫描点数 (Ns): {Ns}")
        print(f"投影时间: {projection_time:.6f}秒")
        print(f"细化时间: {refinement_time:.6f}秒")
        print(f"扫描时间: {scan_time:.6f}秒")
        print(f"实际权重:")
        print(f"  wp (投影时间常数): {wp:.6f}")
        print(f"  wr (细化时间常数): {wr:.6f}")
        print(f"  ws (扫描时间常数): {ws:.6f}")

        return refined_points, Nc, Ns

    def gradient_descent(self, data: List[Point], queries: List[Query],
                        dim_order: List[int], learning_rate: float = 0.1,
                        max_iterations: int = 100) -> Tuple[List[int], float, Dict[str, float]]:
        """使用梯度下降搜索最优列数
        参数:
            data: 数据样本
            queries: 查询样本
            dim_order: 维度顺序
            learning_rate: 学习率
            max_iterations: 最大迭代次数
        返回:
            最优列数, 最小成本, 成本详情
        """
        # 初始化列数，使用数据分布来设置初始值
        col_counts = []
        for i in range(self.dimensions - 1):
            dim = dim_order[i]
            values = [point.coordinates[dim] for point in data]
            # 使用数据分布的标准差来设置初始列数
            std_dev = np.std(values)
            range_size = max(values) - min(values)
            # 根据数据分布特征设置初始列数，限制在2-5之间
            initial_cols = max(2, min(5, int(range_size / (3 * std_dev))))
            col_counts.append(initial_cols)
        
        print(f"初始列数: {col_counts}")
        
        best_cost = float('inf')
        best_col_counts = col_counts.copy()
        best_cost_details = None
        
        # 动态学习率
        current_lr = learning_rate
        
        for iteration in range(max_iterations):
            # 计算当前成本
            current_costs = [self.estimate_query_time(data, query, (dim_order, col_counts)) 
                            for query in queries]
            current_cost = sum(cost for cost, _ in current_costs) / len(queries)
            current_details = current_costs[0][1]
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_col_counts = col_counts.copy()
                best_cost_details = current_details
                # 如果找到更好的解，增加学习率
                current_lr = min(learning_rate * 1.1, 0.5)
            else:
                # 如果没有改善，减小学习率
                current_lr = max(learning_rate * 0.5, 0.01)
            
            # 计算梯度
            gradients = []
            for i in range(len(col_counts)):
                # 正向扰动
                perturbed_counts_plus = col_counts.copy()
                perturbed_counts_plus[i] += 1
                plus_costs = [self.estimate_query_time(data, query, (dim_order, perturbed_counts_plus)) 
                             for query in queries]
                plus_cost = sum(cost for cost, _ in plus_costs) / len(queries)
                
                # 负向扰动
                perturbed_counts_minus = col_counts.copy()
                perturbed_counts_minus[i] = max(2, perturbed_counts_minus[i] - 1)
                minus_costs = [self.estimate_query_time(data, query, (dim_order, perturbed_counts_minus)) 
                              for query in queries]
                minus_cost = sum(cost for cost, _ in minus_costs) / len(queries)
                
                # 计算中心差分梯度
                gradient = (plus_cost - minus_cost) / 2
                gradients.append(gradient)
            
            # 更新列数
            for i in range(len(col_counts)):
                # 使用动态学习率更新
                new_col = max(2, int(col_counts[i] - current_lr * gradients[i]))
                # 限制最大列数
                col_counts[i] = min(new_col, 5)
            
            # 打印优化进度
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 当前成本 = {current_cost:.2f}")
                print(f"  当前列数: {col_counts}")
                print(f"  学习率: {current_lr:.4f}")
                print(f"  投影成本: {current_details['projection_cost']:.2f}")
                print(f"  细化成本: {current_details['refinement_cost']:.2f}")
                print(f"  扫描成本: {current_details['scan_cost']:.2f}")
        
        return best_col_counts, best_cost, best_cost_details

    def optimize_layout(self) -> Tuple[List[int], List[int], Dict[str, float]]:
        """执行布局优化算法"""
        print("\n开始布局优化...")
        sampled_data, sampled_queries = self.sample_data()
        print(f"采样数据点数量: {len(sampled_data)}")
        print(f"采样查询数量: {len(sampled_queries)}")

        # 为每个维度训练RMI模型
        for dim in range(self.dimensions):
            values = [point.coordinates[dim] for point in sampled_data]
            indices = list(range(len(sampled_data)))
            self.rmi_models[dim].fit(values, indices)

        best_cost = float('inf')
        best_layout = None
        best_cost_details = None

        print("\n开始维度排序优化...")
        for sort_dim in range(self.dimensions):
            print(f"\n尝试维度 {sort_dim} 作为排序维度")
            dim_order = self.get_dimension_order(sort_dim)
            print(f"维度顺序: {dim_order}")

            # 使用梯度下降优化列数
            col_counts, cost, cost_details = self.gradient_descent(sampled_data, sampled_queries, dim_order)
            print(f"当前布局成本: {cost:.2f}")

            if cost < best_cost:
                best_cost = cost
                best_layout = (dim_order, col_counts)
                best_cost_details = cost_details
                print(f"找到更好的布局! 成本: {best_cost:.2f}")

        # 使用优化后的列数进行等深度划分
        print("\n执行等深度划分:")
        for i, dim in enumerate(best_layout[0][:-1]):  # 不包含排序维度
            num_cols = best_layout[1][i]
            print(f"\n维度 {dim} 的等深度划分:")
            print(f"列数: {num_cols}")
            boundaries = self.rmi_models[dim].get_column_boundaries(num_cols)
            print("CDF边界值:")
            for j, boundary in enumerate(boundaries):
                print(f"  边界 {j}: {boundary:.3f}")

        return best_layout, best_cost_details


def main():
    # 创建更大的示例数据集
    dataset = []
    # 生成1000个随机点
    for _ in range(1000):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        dataset.append(Point([x, y]))

    # 创建更多的示例查询
    queries = []
    # 生成20个不同大小的查询
    for _ in range(20):
        # 随机选择查询中心点
        center_x = random.uniform(0, 100)
        center_y = random.uniform(0, 100)
        # 随机选择查询范围大小
        range_size = random.uniform(5, 30)
        # 创建查询范围
        min_x = max(0, center_x - range_size/2)
        max_x = min(100, center_x + range_size/2)
        min_y = max(0, center_y - range_size/2)
        max_y = min(100, center_y + range_size/2)
        queries.append(Query([min_x, min_y], [max_x, max_y]))

    # 创建布局优化器
    optimizer = LayoutOptimizer(dataset, queries, sample_rate=0.5, num_partitions=3, error_threshold=0.1)

    # 执行优化
    best_layout, cost_details = optimizer.optimize_layout()
    print("\n最终优化结果:")
    print("维度顺序:", best_layout[0])
    print("列数:", best_layout[1])

    # 输出每个维度的列数
    print("\n各维度列数:")
    for dim in range(optimizer.dimensions):
        if dim == best_layout[0][-1]:  # 如果是排序维度
            print(f"维度 {dim} (排序维度): 不划分")
        else:
            # 找到非排序维度在dim_order中的位置
            dim_idx = best_layout[0].index(dim)
            print(f"维度 {dim}: {best_layout[1][dim_idx]}")

    # 执行示例查询并收集实际的执行时间
    print("\n执行示例查询:")
    total_Nc = 0
    total_Ns = 0
    total_projection_time = 0
    total_refinement_time = 0
    total_scan_time = 0
    
    for i, query in enumerate(queries):
        results, Nc, Ns = optimizer.execute_query(query, best_layout)
        total_Nc += Nc
        total_Ns += Ns
        print(f"\n查询 {i} 结果:")
        print(f"找到 {len(results)} 个结果点")
        if len(results) > 0:
            print("前5个结果点:")
            for point in results[:5]:
                print(f"  ({point.coordinates[0]:.2f}, {point.coordinates[1]:.2f})")
    
    # 输出平均统计信息
    print("\n查询性能统计:")
    print(f"平均扫描单元格数 (Nc): {total_Nc/len(queries):.2f}")
    print(f"平均扫描点数 (Ns): {total_Ns/len(queries):.2f}")
    if total_Nc > 0:
        print(f"平均每单元格点数: {total_Ns/total_Nc:.2f}")
    else:
        print("平均每单元格点数: 0.00")


if __name__ == "__main__":
    main()