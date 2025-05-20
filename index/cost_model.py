import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import time


@dataclass
class Point:
    """表示多维空间中的一个点"""
    coordinates: List[float]


@dataclass
class Query:
    """表示一个范围查询"""
    min_bounds: List[float]  # 查询范围的最小边界
    max_bounds: List[float]  # 查询范围的最大边界


@dataclass
class QueryStats:
    """查询统计信息"""
    num_cells: int  # 与查询矩形相交的单元格数量 (Nc)
    num_points: int  # 扫描的点数 (Ns)
    total_cells: int  # 总单元格数
    avg_cell_size: float  # 可过滤单元格大小的平均值
    median_cell_size: float  # 可过滤单元格大小的中位数
    cell_size_quantiles: List[float]  # 可过滤单元格大小的分位数
    num_filtered_dims: int  # 查询过滤的维度数量
    avg_points_per_cell: float  # 每个访问单元格中访问的平均点数
    points_in_exact_range: int  # 在精确子范围中访问的点数


@dataclass
class CostWeights:
    """成本权重"""
    wp: float  # 投影时间常数
    wr: float  # 细化时间常数
    ws: float  # 扫描时间常数


class FloodCostModel:
    """Flood成本模型实现"""

    def __init__(self):
        self.wp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.wr_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ws_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def extract_features(self, stats: QueryStats) -> np.ndarray:
        """从查询统计信息中提取特征"""
        features = [
            stats.num_cells,
            stats.num_points,
            stats.total_cells,
            stats.avg_cell_size,
            stats.median_cell_size,
            stats.num_filtered_dims,
            stats.avg_points_per_cell,
            stats.points_in_exact_range
        ]
        # 添加分位数特征
        features.extend(stats.cell_size_quantiles)
        return np.array(features)

    def train(self, training_data: List[Tuple[QueryStats, CostWeights]]):
        """训练成本模型
        参数:
            training_data: 训练数据列表，每个元素是(查询统计信息, 成本权重)的元组
        """
        X = np.array([self.extract_features(stats) for stats, _ in training_data])
        y_wp = np.array([weights.wp for _, weights in training_data])
        y_wr = np.array([weights.wr for _, weights in training_data])
        y_ws = np.array([weights.ws for _, weights in training_data])

        # 训练三个模型
        self.wp_model.fit(X, y_wp)
        self.wr_model.fit(X, y_wr)
        self.ws_model.fit(X, y_ws)
        self.is_trained = True

    def predict_weights(self, stats: QueryStats) -> CostWeights:
        """预测成本权重
        参数:
            stats: 查询统计信息
        返回:
            预测的成本权重
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        features = self.extract_features(stats).reshape(1, -1)
        wp = self.wp_model.predict(features)[0]
        wr = self.wr_model.predict(features)[0]
        ws = self.ws_model.predict(features)[0]

        return CostWeights(wp=wp, wr=wr, ws=ws)

    def estimate_query_time(self, stats: QueryStats) -> float:
        """估计查询时间
        参数:
            stats: 查询统计信息
        返回:
            估计的查询时间
        """
        weights = self.predict_weights(stats)
        return (weights.wp * stats.num_cells + 
                weights.wr * stats.num_cells + 
                weights.ws * stats.num_points)


def generate_training_data(num_samples: int, 
                         dimensions: int,
                         num_queries: int,
                         dataset: List[Point],
                         queries: List[Query]) -> List[Tuple[QueryStats, CostWeights]]:
    """生成训练数据
    参数:
        num_samples: 训练样本数量
        dimensions: 数据维度
        num_queries: 每个布局的查询数量
        dataset: 数据集
        queries: 查询列表
    返回:
        训练数据列表
    """
    training_data = []
    
    for _ in range(num_samples):
        # 生成随机布局
        dim_order = list(range(dimensions))
        random.shuffle(dim_order)
        col_counts = [random.randint(2, 5) for _ in range(dimensions-1)]
        layout = (dim_order, col_counts)
        
        # 对每个查询测量实际执行时间
        for query in queries:
            # 开始计时
            start_time = time.time()
            
            # 1. 投影阶段
            projection_start = time.time()
            # 计算查询范围内的单元格
            cells_to_scan = []
            for i, dim in enumerate(dim_order[:-1]):
                min_cdf = 0.0  # 这里应该使用RMI模型预测
                max_cdf = 1.0  # 这里应该使用RMI模型预测
                min_col = int(min_cdf * col_counts[i])
                max_col = int(max_cdf * col_counts[i])
                cells_to_scan.append((min_col, max_col))
            projection_time = time.time() - projection_start
            
            # 2. 细化阶段
            refinement_start = time.time()
            # 获取候选点
            candidates = []
            for point in dataset:
                in_candidate_cells = True
                for i, (dim, (min_col, max_col)) in enumerate(zip(dim_order[:-1], cells_to_scan)):
                    col_idx = 0  # 这里应该使用RMI模型预测
                    if not (min_col <= col_idx <= max_col):
                        in_candidate_cells = False
                        break
                if in_candidate_cells:
                    candidates.append(point)
            refinement_time = time.time() - refinement_start
            
            # 3. 扫描阶段
            scan_start = time.time()
            # 对候选点进行细化
            refined_points = []
            for point in candidates:
                if all(query.min_bounds[dim] <= point.coordinates[dim] <= query.max_bounds[dim] 
                      for dim in range(dimensions)):
                    refined_points.append(point)
            scan_time = time.time() - scan_start
            
            # 计算总时间
            total_time = time.time() - start_time
            
            # 收集统计信息
            num_cells = 1
            for min_col, max_col in cells_to_scan:
                num_cells *= (max_col - min_col + 1)
            
            # 计算单元格大小统计信息
            cell_sizes = []
            for i in range(dimensions-1):
                cell_width = 1.0 / col_counts[i]
                cell_sizes.append(cell_width)
            
            stats = QueryStats(
                num_cells=num_cells,
                num_points=len(refined_points),
                total_cells=np.prod(col_counts),
                avg_cell_size=np.mean(cell_sizes),
                median_cell_size=np.median(cell_sizes),
                cell_size_quantiles=np.percentile(cell_sizes, [25, 50, 75, 90]),
                num_filtered_dims=dimensions,
                avg_points_per_cell=len(refined_points) / num_cells if num_cells > 0 else 0,
                points_in_exact_range=len(refined_points)
            )
            
            # 计算权重
            weights = CostWeights(
                wp=projection_time / num_cells if num_cells > 0 else 0,
                wr=refinement_time / num_cells if num_cells > 0 else 0,
                ws=scan_time / len(refined_points) if len(refined_points) > 0 else 0
            )
            
            training_data.append((stats, weights))
            
            print(f"查询执行时间: {total_time:.4f}秒")
            print(f"投影时间: {projection_time:.4f}秒")
            print(f"细化时间: {refinement_time:.4f}秒")
            print(f"扫描时间: {scan_time:.4f}秒")
            print(f"权重: wp={weights.wp:.6f}, wr={weights.wr:.6f}, ws={weights.ws:.6f}")
            print("---")
    
    return training_data


def main():
    # 创建示例数据集
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

    # 创建示例查询
    queries = [
        Query([1.0, 1.0], [2.0, 2.0]),
        Query([2.0, 2.0], [3.0, 3.0]),
        Query([1.5, 1.5], [2.5, 2.5]),
        Query([2.5, 1.5], [3.5, 2.5]),
        Query([1.2, 2.2], [2.2, 3.2])
    ]

    # 创建成本模型
    cost_model = FloodCostModel()
    
    # 生成训练数据
    print("生成训练数据...")
    training_data = generate_training_data(
        num_samples=10,  # 生成10个训练样本
        dimensions=2,    # 2维数据
        num_queries=5,   # 每个布局5个查询
        dataset=dataset,
        queries=queries
    )
    
    # 训练模型
    print("\n训练模型...")
    cost_model.train(training_data)
    
    # 测试模型
    print("\n测试模型:")
    test_stats = QueryStats(
        num_cells=5,
        num_points=50,
        total_cells=500,
        avg_cell_size=0.5,
        median_cell_size=0.4,
        cell_size_quantiles=[0.2, 0.4, 0.6, 0.8],
        num_filtered_dims=2,
        avg_points_per_cell=10,
        points_in_exact_range=25
    )
    
    # 预测权重
    weights = cost_model.predict_weights(test_stats)
    print(f"预测的权重:")
    print(f"wp (投影时间常数): {weights.wp:.6f}")
    print(f"wr (细化时间常数): {weights.wr:.6f}")
    print(f"ws (扫描时间常数): {weights.ws:.6f}")
    
    # 估计查询时间
    estimated_time = cost_model.estimate_query_time(test_stats)
    print(f"\n估计的查询时间: {estimated_time:.6f} 秒")


if __name__ == "__main__":
    main() 