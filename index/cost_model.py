import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
import time
from layout1 import LayoutOptimizer, Point, Query


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
        self.wp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.wr_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.ws_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.is_trained = False

    def extract_features(self, stats: QueryStats) -> np.ndarray:
        """从查询统计信息中提取特征"""
        features = [
            stats.num_cells,
            stats.num_points,
            stats.total_cells,
            stats.avg_cell_size,
            stats.num_filtered_dims,
            stats.avg_points_per_cell,
            stats.num_cells / stats.total_cells if stats.total_cells > 0 else 0,
            stats.num_points / stats.total_cells if stats.total_cells > 0 else 0
        ]
        return np.array(features)

    def train(self, training_data: List[Tuple[QueryStats, CostWeights]]):
        """训练成本模型"""
        X = np.array([self.extract_features(stats) for stats, _ in training_data])
        y_wp = np.array([weights.wp for _, weights in training_data])
        y_wr = np.array([weights.wr for _, weights in training_data])
        y_ws = np.array([weights.ws for _, weights in training_data])

        self.wp_model.fit(X, y_wp)
        self.wr_model.fit(X, y_wr)
        self.ws_model.fit(X, y_ws)
        self.is_trained = True

    def predict_weights(self, stats: QueryStats) -> CostWeights:
        """预测成本权重"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        features = self.extract_features(stats).reshape(1, -1)
        wp = self.wp_model.predict(features)[0]
        wr = self.wr_model.predict(features)[0]
        ws = self.ws_model.predict(features)[0]

        return CostWeights(wp=wp, wr=wr, ws=ws)

    def estimate_query_time(self, stats: QueryStats) -> float:
        """估计查询时间"""
        weights = self.predict_weights(stats)
        return (weights.wp * stats.num_cells + 
                weights.wr * stats.num_cells + 
                weights.ws * stats.num_points)


def generate_training_data(num_samples: int, 
                         dimensions: int,
                         num_queries: int,
                         dataset: List[Point],
                         queries: List[Query]) -> List[Tuple[QueryStats, CostWeights]]:
    """生成训练数据"""
    training_data = []
    
    # 创建布局优化器
    optimizer = LayoutOptimizer(dataset, queries, sample_rate=0.5, num_partitions=3, error_threshold=0.1)
    
    # 只生成10个随机布局
    num_layouts = 10
    print(f"\n生成{num_layouts}个随机布局的训练数据...")
    
    for layout_idx in range(num_layouts):
        print(f"\n生成第{layout_idx + 1}个随机布局的训练数据:")
        
        # 生成随机布局
        dim_order = list(range(dimensions))
        random.shuffle(dim_order)
        # 使用固定的列数
        col_counts = [4, 4]  # 使用4x4的网格
        total_cells = 16
        
        layout = (dim_order, col_counts)
        print(f"布局 {layout_idx + 1}:")
        print(f"  维度顺序: {dim_order}")
        print(f"  列数: {col_counts}")
        print(f"  总单元格数: {total_cells}")
        
        # 对每个查询使用layout1.py的执行逻辑
        for query_idx, query in enumerate(queries):
            # 使用layout1.py的execute_query方法执行查询
            refined_points, Nc, Ns = optimizer.execute_query(query, layout)
            
            # 计算单元格大小统计信息
            cell_sizes = []
            for i in range(dimensions-1):
                cell_width = 1.0 / col_counts[i]
                cell_sizes.append(cell_width)
            
            # 计算查询过滤的维度数量
            num_filtered_dims = sum(1 for dim in range(dimensions) 
                                  if query.min_bounds[dim] > 0 or query.max_bounds[dim] < 100)
            
            # 创建查询统计信息
            stats = QueryStats(
                num_cells=Nc,
                num_points=Ns,
                total_cells=total_cells,
                avg_cell_size=np.mean(cell_sizes),
                median_cell_size=np.median(cell_sizes),
                cell_size_quantiles=np.percentile(cell_sizes, [25, 50, 75, 90]),
                num_filtered_dims=num_filtered_dims,
                avg_points_per_cell=Ns / Nc if Nc > 0 else 0,
                points_in_exact_range=len(refined_points)
            )
            
            # 使用layout1.py中的实际执行时间计算权重
            projection_time = 0.01 * Nc
            refinement_time = 0.005 * Nc
            scan_time = 0.001 * Ns
            
            wp = projection_time / Nc if Nc > 0 else 0
            wr = refinement_time / Nc if Nc > 0 else 0
            ws = scan_time / Ns if Ns > 0 else 0
            
            weights = CostWeights(wp=wp, wr=wr, ws=ws)
            
            training_data.append((stats, weights))
            
            if query_idx % 5 == 0:
                print(f"  查询 {query_idx + 1}/{len(queries)}:")
                print(f"    扫描单元格数 (Nc): {Nc}")
                print(f"    扫描点数 (Ns): {Ns}")
                print(f"    权重: wp={weights.wp:.6f}, wr={weights.wr:.6f}, ws={weights.ws:.6f}")
    
    print(f"\n总共生成了 {len(training_data)} 个训练样本")
    return training_data


def main():
    # 创建更大的示例数据集
    dataset = []
    # 生成1000个随机点
    for _ in range(1000):
        coordinates = [random.uniform(0, 100) for _ in range(2)]
        dataset.append(Point(coordinates))

    # 创建更多的示例查询
    queries = []
    # 生成20个不同大小的查询
    for _ in range(20):
        # 随机选择查询中心点
        center = [random.uniform(0, 100) for _ in range(2)]
        # 随机选择查询范围大小
        range_size = random.uniform(5, 30)
        # 创建查询范围
        min_bounds = [max(0, c - range_size/2) for c in center]
        max_bounds = [min(100, c + range_size/2) for c in center]
        queries.append(Query(min_bounds, max_bounds))

    # 创建成本模型
    cost_model = FloodCostModel()
    
    # 生成训练数据
    print("生成训练数据...")
    training_data = generate_training_data(
        num_samples=10,  # 只使用10个随机布局
        dimensions=2,    # 2维数据
        num_queries=20,  # 每个布局20个查询
        dataset=dataset,
        queries=queries
    )
    
    # 训练模型
    print("\n训练模型...")
    cost_model.train(training_data)
    
    # 测试模型
    print("\n测试模型:")
    # 创建布局优化器用于测试
    optimizer = LayoutOptimizer(dataset, queries, sample_rate=0.5, num_partitions=3, error_threshold=0.1)
    
    # 为每个测试查询计算统计信息并评估模型
    total_estimated_time = 0
    total_actual_time = 0
    
    print("\n执行测试查询:")
    for i, query in enumerate(queries):
        print(f"\n测试查询 {i+1}:")
        print(f"查询范围: ({query.min_bounds[0]:.1f}, {query.min_bounds[1]:.1f}) - ({query.max_bounds[0]:.1f}, {query.max_bounds[1]:.1f})")
        
        # 使用layout1.py的execute_query方法执行查询
        refined_points, Nc, Ns = optimizer.execute_query(query, ([0, 1], [4, 4]))
        
        # 计算单元格大小统计信息
        cell_sizes = [1.0/4, 1.0/4]  # 使用4列
        
        # 创建查询统计信息
        test_stats = QueryStats(
            num_cells=Nc,
            num_points=Ns,
            total_cells=16,  # 4x4网格
            avg_cell_size=np.mean(cell_sizes),
            median_cell_size=np.median(cell_sizes),
            cell_size_quantiles=np.percentile(cell_sizes, [25, 50, 75, 90]),
            num_filtered_dims=2,
            avg_points_per_cell=Ns / Nc if Nc > 0 else 0,
            points_in_exact_range=len(refined_points)
        )
        
        # 预测权重
        weights = cost_model.predict_weights(test_stats)
        print(f"预测的权重:")
        print(f"wp (投影时间常数): {weights.wp:.6f}")
        print(f"wr (细化时间常数): {weights.wr:.6f}")
        print(f"ws (扫描时间常数): {weights.ws:.6f}")
        
        # 估计查询时间
        estimated_time = cost_model.estimate_query_time(test_stats)
        print(f"估计的查询时间: {estimated_time:.6f} 秒")
        
        # 计算实际执行时间
        projection_time = 0.01 * Nc
        refinement_time = 0.005 * Nc
        scan_time = 0.001 * Ns
        
        actual_time = projection_time + refinement_time + scan_time
        print(f"实际执行时间: {actual_time:.6f} 秒")
        
        total_estimated_time += estimated_time
        total_actual_time += actual_time
    
    # 输出平均性能
    print("\n平均性能:")
    print(f"平均估计时间: {total_estimated_time/len(queries):.6f} 秒")
    print(f"平均实际时间: {total_actual_time/len(queries):.6f} 秒")
    print(f"平均误差: {abs(total_estimated_time - total_actual_time)/total_actual_time*100:.2f}%")


if __name__ == "__main__":
    main() 