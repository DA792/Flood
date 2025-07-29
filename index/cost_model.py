import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import random
import time

try:
    from .data_structures import Point, Query, QueryStats, CostWeights
except ImportError:
    from data_structures import Point, Query, QueryStats, CostWeights


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
        """训练成本模型 - 为每个权重训练独立的随机森林模型"""
        print(f"\n开始训练3个独立的随机森林模型...")
        print(f"训练数据样本数: {len(training_data)}")
        
        X = np.array([self.extract_features(stats) for stats, _ in training_data])
        y_wp = np.array([weights.wp for _, weights in training_data])
        y_wr = np.array([weights.wr for _, weights in training_data])
        y_ws = np.array([weights.ws for _, weights in training_data])

        print(f"\n特征矩阵形状: {X.shape}")
        print(f"wp目标值范围: [{np.min(y_wp):.6f}, {np.max(y_wp):.6f}]")
        print(f"wr目标值范围: [{np.min(y_wr):.6f}, {np.max(y_wr):.6f}]")
        print(f"ws目标值范围: [{np.min(y_ws):.6f}, {np.max(y_ws):.6f}]")

        # 训练wp模型（投影权重预测）
        print(f"\n训练wp随机森林模型（投影权重预测）...")
        self.wp_model.fit(X, y_wp)
        wp_score = self.wp_model.score(X, y_wp)
        print(f"wp模型训练完成，R²得分: {wp_score:.4f}")
        print(f"wp模型特征重要性: {self.wp_model.feature_importances_}")

        # 训练wr模型（细化权重预测）
        print(f"\n训练wr随机森林模型（细化权重预测）...")
        self.wr_model.fit(X, y_wr)
        wr_score = self.wr_model.score(X, y_wr)
        print(f"wr模型训练完成，R²得分: {wr_score:.4f}")
        print(f"wr模型特征重要性: {self.wr_model.feature_importances_}")

        # 训练ws模型（扫描权重预测）
        print(f"\n训练ws随机森林模型（扫描权重预测）...")
        self.ws_model.fit(X, y_ws)
        ws_score = self.ws_model.score(X, y_ws)
        print(f"ws模型训练完成，R²得分: {ws_score:.4f}")
        print(f"ws模型特征重要性: {self.ws_model.feature_importances_}")

        self.is_trained = True
        
        print(f"\n所有随机森林模型训练完成!")
        print(f"模型性能总结:")
        print(f"  wp模型 (投影): R² = {wp_score:.4f}")
        print(f"  wr模型 (细化): R² = {wr_score:.4f}")
        print(f"  ws模型 (扫描): R² = {ws_score:.4f}")

    def predict_weights(self, stats: QueryStats) -> CostWeights:
        """使用3个独立的随机森林模型预测成本权重"""
        if not self.is_trained:
            raise RuntimeError("随机森林模型尚未训练")

        features = self.extract_features(stats).reshape(1, -1)
        
        # 使用3个独立的随机森林模型分别预测每个权重
        wp = self.wp_model.predict(features)[0]  # 投影权重
        wr = self.wr_model.predict(features)[0]  # 细化权重
        ws = self.ws_model.predict(features)[0]  # 扫描权重

        return CostWeights(wp=wp, wr=wr, ws=ws)

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        if not self.is_trained:
            return {"status": "未训练"}
        
        return {
            "status": "已训练",
            "wp_model": {
                "n_estimators": self.wp_model.n_estimators,
                "max_depth": self.wp_model.max_depth,
                "feature_importances": self.wp_model.feature_importances_.tolist()
            },
            "wr_model": {
                "n_estimators": self.wr_model.n_estimators,
                "max_depth": self.wr_model.max_depth,
                "feature_importances": self.wr_model.feature_importances_.tolist()
            },
            "ws_model": {
                "n_estimators": self.ws_model.n_estimators,
                "max_depth": self.ws_model.max_depth,
                "feature_importances": self.ws_model.feature_importances_.tolist()
            }
        }

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
    
    # 只生成10个随机布局用于训练
    num_layouts = 10
    print(f"\n生成{num_layouts}个随机布局的训练数据...")
    
    for layout_idx in range(num_layouts):
        print(f"\n生成第{layout_idx + 1}个随机布局的训练数据:")
        
        # 生成随机布局
        dim_order = list(range(dimensions))
        random.shuffle(dim_order)
        
        # 为每个维度随机生成列数，确保总单元格数在合理范围内
        col_counts = []
        remaining_cells = random.randint(64, 256)  # 增加总单元格数范围以适应1兆数据
        for i in range(dimensions-1):
            if i == dimensions-2:  # 最后一个维度
                col_counts.append(remaining_cells)
            else:
                # 随机分配列数，但确保不会导致总单元格数过大
                max_cols = int(remaining_cells ** (1/(dimensions-i-1)))
                cols = random.randint(4, max(5, max_cols))  # 增加最小列数
                col_counts.append(cols)
                remaining_cells //= cols
        
        total_cells = np.prod(col_counts)
        
        layout = (dim_order, col_counts)
        print(f"布局 {layout_idx + 1}:")
        print(f"  维度顺序: {dim_order}")
        print(f"  列数: {col_counts}")
        print(f"  总单元格数: {total_cells}")
        
        # 对每个查询执行
        for query_idx, query in enumerate(queries):
            # 1. 投影阶段
            cells_to_scan = []
            for i, dim in enumerate(dim_order[:-1]):
                min_cdf = query.min_bounds[dim] / 100.0
                max_cdf = query.max_bounds[dim] / 100.0
                min_col = int(min_cdf * col_counts[i])
                max_col = int(max_cdf * col_counts[i])
                cells_to_scan.append((min_col, max_col))
            
            # 计算查询范围内的单元格数量 (Nc)
            Nc = 1
            for min_col, max_col in cells_to_scan:
                Nc *= (max_col - min_col + 1)
            
            # 记录投影阶段开始时间
            projection_start = time.time()
            
            # 模拟投影阶段的延迟（减少延迟以适应大数据集）
            time.sleep(0.001 * Nc)  # 每个单元格0.001秒
            
            # 2. 细化阶段（优化：直接进行范围查询而不是逐点检查）
            candidates = []
            candidate_count = 0
            for point in dataset:
                # 快速检查点是否在查询范围内
                if (query.min_bounds[0] <= point.coordinates[0] <= query.max_bounds[0] and
                    query.min_bounds[1] <= point.coordinates[1] <= query.max_bounds[1]):
                    candidates.append(point)
                    candidate_count += 1
                    # 限制候选点数量以提高训练性能
                    if candidate_count > 10000:  # 训练时最多处理1万个候选点
                        break
            
            # 记录细化阶段开始时间
            refinement_start = time.time()
            
            # 模拟细化阶段的延迟（减少延迟）
            time.sleep(0.0001 * len(candidates))  # 每个候选点0.0001秒
            
            # 3. 扫描阶段（已经在上面完成了精确匹配）
            refined_points = candidates
            
            # 记录扫描阶段结束时间
            scan_end = time.time()
            
            # 模拟扫描阶段的延迟（减少延迟）
            time.sleep(0.00001 * len(refined_points))  # 每个结果点0.00001秒
            
            # 计算各阶段的实际执行时间
            projection_time = refinement_start - projection_start
            refinement_time = scan_end - refinement_start
            scan_time = scan_end - refinement_start
            
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
                num_points=len(refined_points),
                total_cells=total_cells,
                avg_cell_size=np.mean(cell_sizes),
                median_cell_size=np.median(cell_sizes),
                cell_size_quantiles=np.percentile(cell_sizes, [25, 50, 75, 90]),
                num_filtered_dims=num_filtered_dims,
                avg_points_per_cell=len(refined_points) / Nc if Nc > 0 else 0,
                points_in_exact_range=len(refined_points)
            )
            
            # 计算实际权重
            wp = projection_time / Nc if Nc > 0 else 0
            wr = refinement_time / Nc if Nc > 0 else 0
            ws = scan_time / len(refined_points) if len(refined_points) > 0 else 0
            
            weights = CostWeights(wp=wp, wr=wr, ws=ws)
            
            training_data.append((stats, weights))
            
            if query_idx % 5 == 0:
                print(f"  查询 {query_idx + 1}/{len(queries)}:")
                print(f"    扫描单元格数 (Nc): {Nc}")
                print(f"    扫描点数 (Ns): {len(refined_points)}")
                print(f"    实际权重: wp={weights.wp:.6f}, wr={weights.wr:.6f}, ws={weights.ws:.6f}")
    
    print(f"\n总共生成了 {len(training_data)} 个训练样本")
    return training_data


def main():
    # 创建1兆数据的示例数据集
    dataset = []
    print("正在生成1,000,000个随机数据点...")
    # 生成1,000,000个随机点
    for i in range(1000000):
        coordinates = [random.uniform(0, 100) for _ in range(2)]
        dataset.append(Point(coordinates))
        # 每生成10万个点显示一次进度
        if (i + 1) % 100000 == 0:
            print(f"已生成 {i + 1:,} 个数据点")

    print(f"数据集生成完成，共 {len(dataset):,} 个数据点")

    # 创建更多的示例查询
    queries = []
    print("正在生成100个测试查询...")
    # 生成100个不同大小的查询
    for _ in range(100):
        # 随机选择查询中心点
        center = [random.uniform(0, 100) for _ in range(2)]
        # 随机选择查询范围大小
        range_size = random.uniform(5, 30)
        # 创建查询范围
        min_bounds = [max(0, c - range_size/2) for c in center]
        max_bounds = [min(100, c + range_size/2) for c in center]
        queries.append(Query(min_bounds, max_bounds))

    print(f"查询集生成完成，共 {len(queries)} 个查询")

    # 创建成本模型
    cost_model = FloodCostModel()
    
    # 生成训练数据
    print("生成训练数据...")
    training_data = generate_training_data(
        num_samples=10,  # 只使用10个随机布局
        dimensions=2,    # 2维数据
        num_queries=50,  # 每个布局50个查询（从前50个查询中选择）
        dataset=dataset,
        queries=queries[:50]  # 只使用前50个查询进行训练
    )
    
    # 训练模型
    print("\n训练模型...")
    cost_model.train(training_data)
    
    # 测试模型
    print("\n测试模型:")
    
    # 为每个测试查询计算统计信息并评估模型
    total_estimated_time = 0
    total_actual_time = 0
    
    # 使用后50个查询进行测试
    test_queries = queries[50:]
    print(f"\n执行测试查询（使用后{len(test_queries)}个查询进行测试）:")
    
    for i, query in enumerate(test_queries):
        print(f"\n测试查询 {i+1}/{len(test_queries)}:")
        print(f"查询范围: ({query.min_bounds[0]:.1f}, {query.min_bounds[1]:.1f}) - ({query.max_bounds[0]:.1f}, {query.max_bounds[1]:.1f})")
        
        # 使用最优布局进行测试（这里使用16x16网格以适应更大的数据集）
        dim_order = [0, 1]  # 最优维度顺序
        col_counts = [16, 16]  # 最优列数配置（增加到16x16以适应1兆数据）
        total_cells = 256
        
        # 1. 投影阶段
        cells_to_scan = []
        for dim in range(2):
            min_cdf = query.min_bounds[dim] / 100.0
            max_cdf = query.max_bounds[dim] / 100.0
            min_col = int(min_cdf * col_counts[dim])
            max_col = int(max_cdf * col_counts[dim])
            cells_to_scan.append((min_col, max_col))
        
        # 计算单元格数量
        Nc = 1
        for min_col, max_col in cells_to_scan:
            Nc *= (max_col - min_col + 1)
        
        # 记录投影阶段开始时间
        projection_start = time.time()
        
        # 模拟投影阶段的延迟（减少延迟以适应更大数据集）
        time.sleep(0.001 * Nc)  # 每个单元格0.001秒
        
        # 2. 计算扫描的点数（优化：只扫描相关区域的点）
        candidates = []
        candidate_count = 0
        for point in dataset:
            # 快速检查点是否在查询范围内
            if (query.min_bounds[0] <= point.coordinates[0] <= query.max_bounds[0] and
                query.min_bounds[1] <= point.coordinates[1] <= query.max_bounds[1]):
                candidates.append(point)
                candidate_count += 1
                # 限制候选点数量以提高性能
                if candidate_count > 50000:  # 最多处理5万个候选点
                    break
        
        # 记录细化阶段开始时间
        refinement_start = time.time()
        
        # 模拟细化阶段的延迟（减少延迟）
        time.sleep(0.0001 * len(candidates))  # 每个候选点0.0001秒
        
        # 3. 扫描阶段（已经在上面完成了精确匹配）
        refined_points = candidates
        
        # 记录扫描阶段结束时间
        scan_end = time.time()
        
        # 模拟扫描阶段的延迟（减少延迟）
        time.sleep(0.00001 * len(refined_points))  # 每个结果点0.00001秒
        
        # 3. 计算单元格大小统计信息
        cell_sizes = [1.0/16, 1.0/16]  # 16x16网格
        
        # 创建查询统计信息
        test_stats = QueryStats(
            num_cells=Nc,
            num_points=len(refined_points),
            total_cells=total_cells,
            avg_cell_size=np.mean(cell_sizes),
            median_cell_size=np.median(cell_sizes),
            cell_size_quantiles=np.percentile(cell_sizes, [25, 50, 75, 90]),
            num_filtered_dims=2,
            avg_points_per_cell=len(refined_points) / Nc if Nc > 0 else 0,
            points_in_exact_range=len(refined_points)
        )
        
        # 使用训练好的模型预测权重
        weights = cost_model.predict_weights(test_stats)
        print(f"预测的权重:")
        print(f"wp (投影时间常数): {weights.wp:.6f}")
        print(f"wr (细化时间常数): {weights.wr:.6f}")
        print(f"ws (扫描时间常数): {weights.ws:.6f}")
        
        # 使用预测的权重估计查询时间
        estimated_time = cost_model.estimate_query_time(test_stats)
        print(f"估计的查询时间: {estimated_time:.6f} 秒")
        
        # 计算实际执行时间
        actual_time = scan_end - projection_start
        print(f"实际执行时间: {actual_time:.6f} 秒")
        print(f"扫描单元格数 (Nc): {Nc}")
        print(f"匹配点数: {len(refined_points):,}")
        
        total_estimated_time += estimated_time
        total_actual_time += actual_time
    
    # 输出平均性能
    print("\n平均性能:")
    print(f"平均估计时间: {total_estimated_time/len(test_queries):.6f} 秒")
    print(f"平均实际时间: {total_actual_time/len(test_queries):.6f} 秒")
    print(f"平均误差: {abs(total_estimated_time - total_actual_time)/total_actual_time*100:.2f}%")


if __name__ == "__main__":
    main()
