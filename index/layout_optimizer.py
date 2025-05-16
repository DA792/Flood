import numpy as np
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import math

@dataclass
class Point:
    """表示d维空间中的一个点"""
    coordinates: List[float]

@dataclass
class Query:
    """表示一个范围查询"""
    min_bounds: List[float]  # 每个维度的最小值
    max_bounds: List[float]  # 每个维度的最大值

class RMIModel:
    """递归模型索引(RMI)实现"""
    def __init__(self, num_models: int = 2):
        self.num_models = num_models
        self.models = [LinearRegression() for _ in range(num_models)]
        self.boundaries = []
        
    def fit(self, values: List[float], indices: List[int]):
        """训练RMI模型"""
        n = len(values)
        if n == 0:
            return
            
        # 计算每个模型的训练数据范围
        chunk_size = n // self.num_models
        self.boundaries = [values[i * chunk_size] for i in range(self.num_models)]
        self.boundaries.append(float('inf'))
        
        # 训练每个模型
        for i in range(self.num_models):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_models - 1 else n
            
            X = np.array(values[start_idx:end_idx]).reshape(-1, 1)
            y = np.array(indices[start_idx:end_idx])
            
            if len(X) > 0:
                self.models[i].fit(X, y)
    
    def predict(self, value: float) -> float:
        """使用RMI模型预测索引位置"""
        # 找到合适的模型
        model_idx = 0
        for i, boundary in enumerate(self.boundaries[:-1]):
            if value < self.boundaries[i + 1]:
                model_idx = i
                break
                
        # 使用选定的模型进行预测
        prediction = self.models[model_idx].predict([[value]])[0]
        return max(0, min(1, prediction))  # 归一化到[0,1]范围

class LayoutOptimizer:
    """布局优化算法实现"""
    def __init__(self, dataset: List[Point], queries: List[Query], 
                 sample_rate: float = 0.1, num_partitions: int = 10):
        self.dataset = dataset
        self.queries = queries
        self.sample_rate = sample_rate
        self.num_partitions = num_partitions
        self.dimensions = len(dataset[0].coordinates) if dataset else 0
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
    
    def flatten_data(self, data: List[Point]) -> List[Point]:
        """使用RMI模型展平数据"""
        flattened_data = []
        
        # 为每个维度训练RMI模型
        for dim in range(self.dimensions):
            values = [point.coordinates[dim] for point in data]
            indices = list(range(len(values)))
            sorted_pairs = sorted(zip(values, indices))
            values, indices = zip(*sorted_pairs)
            self.rmi_models[dim].fit(list(values), list(indices))
        
        # 使用训练好的模型展平数据
        for point in data:
            flattened_coords = []
            for dim in range(self.dimensions):
                cdf_value = self.rmi_models[dim].predict(point.coordinates[dim])
                flattened_coords.append(cdf_value)
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
        - wp: 在单个单元格上执行投影的平均时间
        - wr: 在单元格上执行细化的平均时间
        - ws: 每次扫描点的平均时间
        - Nc: 查询矩形内的单元格数量
        - Ns: 需要扫描的点数
        """
        dim_order, col_counts = layout
        grid_sizes = []
        
        # 计算每个维度的网格大小
        for i, dim in enumerate(dim_order[:-1]):
            values = [point.coordinates[dim] for point in data]
            min_val, max_val = min(values), max(values)
            grid_size = (max_val - min_val) / col_counts[i]
            grid_sizes.append(grid_size)
        
        # 计算查询范围内的单元格数量 (Nc)
        cells_to_scan = 1
        for i, (dim, grid_size) in enumerate(zip(dim_order[:-1], grid_sizes)):
            query_range = query.max_bounds[dim] - query.min_bounds[dim]
            cells_in_dim = math.ceil(query_range / grid_size)
            cells_to_scan *= cells_in_dim
        
        # 估计需要扫描的点数 (Ns)
        # 使用网格密度来估计
        total_cells = np.prod(col_counts)
        avg_points_per_cell = len(data) / total_cells
        estimated_points = cells_to_scan * avg_points_per_cell
        
        # 成本模型参数
        wp = 0.1  # 投影时间常数
        wr = 0.2  # 细化时间常数
        ws = 0.05  # 扫描时间常数
        
        # 计算总成本
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
            'ws': ws
        }
        
        return total_cost, cost_details

    def execute_query(self, query: Query, layout: Tuple[List[int], List[int]]) -> List[Point]:
        """执行查询，包括投影、细化和扫描过程
        参数:
            query: 查询范围
            layout: 布局信息 (维度顺序, 列数)
        返回:
            查询结果点列表
        """
        dim_order, col_counts = layout
        grid_sizes = []
        
        # 1. 计算网格大小
        for i, dim in enumerate(dim_order[:-1]):
            values = [point.coordinates[dim] for point in self.dataset]
            min_val, max_val = min(values), max(values)
            grid_size = (max_val - min_val) / col_counts[i]
            grid_sizes.append(grid_size)
        
        # 2. 投影过程：找到查询范围内的单元格
        cells_to_scan = []
        for i, (dim, grid_size) in enumerate(zip(dim_order[:-1], grid_sizes)):
            min_cell = math.floor((query.min_bounds[dim] - min_val) / grid_size)
            max_cell = math.ceil((query.max_bounds[dim] - min_val) / grid_size)
            cells_to_scan.append((min_cell, max_cell))
        
        # 3. 获取候选点
        candidates = []
        for point in self.dataset:
            in_candidate_cells = True
            for i, (dim, (min_cell, max_cell)) in enumerate(zip(dim_order[:-1], cells_to_scan)):
                cell_idx = math.floor((point.coordinates[dim] - min_val) / grid_sizes[i])
                if not (min_cell <= cell_idx <= max_cell):
                    in_candidate_cells = False
                    break
            if in_candidate_cells:
                candidates.append(point)
        
        # 4. 细化过程：对候选点进行精确过滤
        refined_points = self.refine_candidates(self.dataset, query, candidates)
        
        return refined_points

    def gradient_descent(self, data: List[Point], queries: List[Query], 
                        dim_order: List[int], learning_rate: float = 0.1, 
                        max_iterations: int = 100) -> Tuple[List[int], float, Dict[str, float]]:
        """使用梯度下降搜索最优列数"""
        # 只为前 d-1 个维度分配列数
        col_counts = [self.num_partitions] * (self.dimensions - 1)
        best_cost = float('inf')
        best_col_counts = col_counts.copy()
        best_cost_details = None
        
        for iteration in range(max_iterations):
            # 计算当前成本
            current_costs = [self.estimate_query_time(data, query, (dim_order, col_counts)) 
                           for query in queries]
            current_cost = sum(cost for cost, _ in current_costs) / len(queries)
            current_details = current_costs[0][1]  # 使用第一个查询的详细信息
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_col_counts = col_counts.copy()
                best_cost_details = current_details
            
            # 计算梯度
            gradients = []
            for i in range(len(col_counts)):
                # 扰动当前列数
                perturbed_counts = col_counts.copy()
                perturbed_counts[i] += 1
                
                # 计算扰动后的成本
                perturbed_costs = [self.estimate_query_time(data, query, (dim_order, perturbed_counts)) 
                                 for query in queries]
                perturbed_cost = sum(cost for cost, _ in perturbed_costs) / len(queries)
                
                # 计算梯度
                gradient = perturbed_cost - current_cost
                gradients.append(gradient)
            
            # 更新列数
            for i in range(len(col_counts)):
                col_counts[i] = max(2, int(col_counts[i] - learning_rate * gradients[i]))
            
            # 打印优化进度
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 当前成本 = {current_cost:.2f}")
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
        
        flattened_data = self.flatten_data(sampled_data)
        print("数据展平完成")
        
        best_cost = float('inf')
        best_layout = None
        best_cost_details = None
        
        print("\n开始维度排序优化...")
        for sort_dim in range(self.dimensions):
            print(f"\n尝试维度 {sort_dim} 作为排序维度")
            dim_order = self.get_dimension_order(sort_dim)
            print(f"维度顺序: {dim_order}")
            
            col_counts, cost, cost_details = self.gradient_descent(flattened_data, sampled_queries, dim_order)
            print(f"当前布局成本: {cost:.2f}")
            
            if cost < best_cost:
                best_cost = cost
                best_layout = (dim_order, col_counts)
                best_cost_details = cost_details
                print(f"找到更好的布局! 成本: {best_cost:.2f}")
        
        return best_layout, best_cost_details

def main():
    # 创建更大的示例数据集
    dataset = [
        Point([1.0, 2.0]),
        Point([2.0, 3.0]),
        Point([3.0, 1.0]),
        Point([1.5, 2.5]),
        Point([2.5, 1.5]),
        Point([3.5, 2.5]),
        Point([1.2, 3.2]),
        Point([2.8, 1.8]),
        Point([3.2, 2.8]),
        Point([1.8, 2.2])
    ]
    
    # 创建更多的示例查询
    queries = [
        Query([1.0, 1.0], [2.0, 2.0]),
        Query([2.0, 2.0], [3.0, 3.0]),
        Query([1.5, 1.5], [2.5, 2.5]),
        Query([2.5, 1.5], [3.5, 2.5]),
        Query([1.2, 2.2], [2.2, 3.2])
    ]
    
    # 创建布局优化器
    optimizer = LayoutOptimizer(dataset, queries, sample_rate=0.5)  # 增加采样率
    
    # 执行优化
    best_layout, cost_details = optimizer.optimize_layout()
    print("\n最终优化结果:")
    print("维度顺序:", best_layout[0])
    print("列数:", best_layout[1])
    print("\n成本模型参数:")
    print(f"投影时间常数 (wp): {cost_details['wp']}")
    print(f"细化时间常数 (wr): {cost_details['wr']}")
    print(f"扫描时间常数 (ws): {cost_details['ws']}")
    print("\n成本组成:")
    print(f"总成本: {cost_details['total_cost']:.2f}")
    print(f"投影成本: {cost_details['projection_cost']:.2f}")
    print(f"细化成本: {cost_details['refinement_cost']:.2f}")
    print(f"扫描成本: {cost_details['scan_cost']:.2f}")
    print(f"扫描单元格数: {cost_details['cells_to_scan']}")
    print(f"估计扫描点数: {cost_details['estimated_points']:.2f}")
    
    # 执行示例查询
    print("\n执行示例查询:")
    for i, query in enumerate(queries):
        results = optimizer.execute_query(query, best_layout)
        print(f"\n查询 {i}:")
        print(f"查询范围: ({query.min_bounds[0]}, {query.min_bounds[1]}) - ({query.max_bounds[0]}, {query.max_bounds[1]})")
        print(f"找到 {len(results)} 个结果点:")
        for point in results:
            print(f"  ({point.coordinates[0]}, {point.coordinates[1]})")

if __name__ == "__main__":
    main() 