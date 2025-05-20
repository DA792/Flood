import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import time


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
                         num_queries: int) -> List[Tuple[QueryStats, CostWeights]]:
    """生成训练数据
    参数:
        num_samples: 训练样本数量
        dimensions: 数据维度
        num_queries: 每个布局的查询数量
    返回:
        训练数据列表
    """
    training_data = []
    
    for _ in range(num_samples):
        # 生成随机布局
        dim_order = list(range(dimensions))
        random.shuffle(dim_order)
        col_counts = [random.randint(2, 5) for _ in range(dimensions-1)]
        
        # 生成随机查询
        queries = []
        for _ in range(num_queries):
            min_bounds = [random.random() for _ in range(dimensions)]
            max_bounds = [min_bounds[i] + random.random() * 0.5 
                         for i in range(dimensions)]
            queries.append((min_bounds, max_bounds))
        
        # 对每个查询测量实际执行时间
        for query in queries:
            # 模拟查询执行并收集统计信息
            start_time = time.time()
            
            # 这里应该调用实际的查询执行代码
            # 为了演示，我们生成一些随机统计信息
            stats = QueryStats(
                num_cells=random.randint(1, 10),
                num_points=random.randint(10, 100),
                total_cells=random.randint(100, 1000),
                avg_cell_size=random.random(),
                median_cell_size=random.random(),
                cell_size_quantiles=[random.random() for _ in range(4)],
                num_filtered_dims=random.randint(1, dimensions),
                avg_points_per_cell=random.random() * 10,
                points_in_exact_range=random.randint(5, 50)
            )
            
            # 测量各个阶段的执行时间
            projection_time = random.random() * 0.1
            refinement_time = random.random() * 0.2
            scan_time = random.random() * 0.3
            
            # 计算权重
            weights = CostWeights(
                wp=projection_time / stats.num_cells if stats.num_cells > 0 else 0,
                wr=refinement_time / stats.num_cells if stats.num_cells > 0 else 0,
                ws=scan_time / stats.num_points if stats.num_points > 0 else 0
            )
            
            training_data.append((stats, weights))
    
    return training_data


def main():
    # 创建成本模型
    cost_model = FloodCostModel()
    
    # 生成训练数据
    print("生成训练数据...")
    training_data = generate_training_data(
        num_samples=100,  # 生成100个训练样本
        dimensions=3,     # 3维数据
        num_queries=10    # 每个布局10个查询
    )
    
    # 训练模型
    print("训练模型...")
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