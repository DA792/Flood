import argparse
import random
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os
import bisect

parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument('-d', help='dataset', default='foursquare2')
parser.add_argument('-ls', type=int, choices=[100, 500, 1000, 5000, 10000], help='lower bound selectivity', default=100)
parser.add_argument('-hs', type=int, choices=[200, 1000, 2000, 10000, 20000], help='higher bound selectivity', default=200)
parser.add_argument('-r', type=float, choices=[0.005, 0.01, 0.05, 0.1, 0.5, 1], help='query range ratio', default=0.05)
parser.add_argument('-isk', type=int, choices=[0, 1], help='skew distribution', default=0)
parser.add_argument('-ig', type=int, choices=[0, 1], help='gaussian distribution', default=0)
parser.add_argument('-im', type=int, choices=[0, 1], help='mixed distribution', default=0)
parser.add_argument('-sr', type=int, help='sample rate', default=100)
parser.add_argument('-et', type=float, help='error threshold', default=0.1)

def calculate_selectivity(query, dim, full_range):
    """计算单个查询 query 在维度 dim 上的选择性；full_range 是该维度总体跨度。"""
    if dim == 'x':
        sel_range = query.max_x - query.min_x
    else:
        sel_range = query.max_y - query.min_y
    return sel_range / full_range if full_range > 0 else 0.0

def get_sorted_dimensions(queries, data_points):
    """
    根据所有查询在各维度上的平均选择性，返回 ['x','y'] 或 ['y','x']。
    full_range_x/y 从 data_points 中计算。
    """
    xs = [p.x for p in data_points]
    ys = [p.y for p in data_points]
    full_x = max(xs) - min(xs)
    full_y = max(ys) - min(ys)

    avg_sel_x = sum(calculate_selectivity(q, 'x', full_x) for q in queries) / len(queries)
    avg_sel_y = sum(calculate_selectivity(q, 'y', full_y) for q in queries) / len(queries)

    # 选择性大的维度优先排在前面
    if avg_sel_x >= avg_sel_y:
        return ['x', 'y']
    else:
        return ['y', 'x']

def stratified_sampling(q, ns, sr):  # q: 查询集合， ns: 采样个数， sr: 采样比例
    x_min = min(q[0])
    y_min = min(q[1])
    x_max = max(q[0])
    y_max = max(q[1])
    x_range = (x_max - x_min) / (ns - 1)  # 计算x轴和y轴的区间长度
    y_range = (y_max - y_min) / (ns - 1)
    distribution_array = [[[] for a in range(ns)] for b in range(ns)]  # 创建一个二维列表，用于存储每个查询的索引
    for i in range(q.shape[0]):  # 遍历查询集合
        x1 = q[0][i]
        y1 = q[1][i]
        x_idx = int((x1 - x_min) // x_range)
        y_idx = int((y1 - y_min) // y_range)
        distribution_array[x_idx][y_idx].append(i)

    sampled_query_idx = []  # 用于存储最终的采样结果
    for n in range(ns):
        for m in range(ns):
            if distribution_array[n][m]:
                if int(len(distribution_array[n][m]) * sr / 100) > 0:
                    sample = random.sample(distribution_array[n][m], int(len(distribution_array[n][m]) * sr / 100))   # 随机打乱采样顺序
                    sampled_query_idx.extend(sample)
                else:
                    sampled_query_idx.extend(distribution_array[n][m])
    return sampled_query_idx

def read_query(query_file, idx=None):
    query_set = []
    if idx is None:  # 读取所有查询
        with open(query_file) as f:
            while True:
                line = f.readline()
                if line:
                    component = line.strip().split(',')  # 使用逗号分隔
                    if len(component) >= 4:  # 确保有足够的组件
                        query = Query(float(component[0]), float(component[1]), float(component[2]), float(component[3]))
                        query_set.append(query)
                else:
                    break
    else:
        i = 0
        with open(query_file) as f:
            while True:
                line = f.readline()
                if line:
                    component = line.strip().split(',')  # 使用逗号分隔
                    if len(component) >= 4:  # 确保有足够的组件
                        query = Query(float(component[0]), float(component[1]), float(component[2]), float(component[3]))
                        if i in idx:
                            query_set.append(query)
                        i += 1
                else:
                    break
    return query_set

class PLMModel:
    """Flood风格的分段线性模型（PLM）"""
    def __init__(self, values, delta=1.0):
        self.segments = []  # 每段: (start_val, start_idx, slope, intercept, end_idx)
        self.delta = delta
        self.values = values
        self._build_model()

    def _build_model(self):
        n = len(self.values)
        idx = 0
        while idx < n:
            start_idx = idx
            start_val = self.values[idx]
            xs, ys = [], []
            while idx < n:
                xs.append(self.values[idx])
                ys.append(idx)
                if len(xs) > 1:
                    # 拟合线性函数 y = a*x + b
                    a, b = self._fit_line(xs, ys)
                    preds = [a * x + b for x in xs]
                    errors = [y - p for y, p in zip(ys, preds)]
                    avg_error = sum(abs(e) for e in errors) / len(errors)
                    # 保证下界性质: 预测值不能超过真实索引
                    if any(p > y for p, y in zip(preds, ys)) or avg_error > self.delta:
                        break
                idx += 1
            # 用前一组参数
            if len(xs) > 1:
                a, b = self._fit_line(xs[:-1], ys[:-1])
                end_idx = idx - 1
            else:
                a, b = 0, ys[0]
                end_idx = idx
            self.segments.append({
                'start_val': start_val,
                'start_idx': start_idx,
                'slope': a,
                'intercept': b,
                'end_idx': end_idx
            })
        # 输出分段参数
        print("\nPLM分段参数:")
        for i, seg in enumerate(self.segments):
            print(f"  段 {i}: 起点值={seg['start_val']:.2f}, 起点索引={seg['start_idx']}, 斜率={seg['slope']:.4f}, 截距={seg['intercept']:.2f}, 终止索引={seg['end_idx']}")

    def _fit_line(self, xs, ys):
        n = len(xs)
        if n == 1:
            return 0, ys[0]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n))
        a = num / den if den != 0 else 0
        b = y_mean - a * x_mean
        return a, b

    def predict_range(self, min_val, max_val):
        # 找到min_val和max_val分别落在哪个分段
        segs = self.segments
        min_idx = 0
        for seg in segs:
            if min_val >= seg['start_val']:
                min_idx = seg['start_idx']
        max_idx = 0
        for seg in segs:
            if max_val >= seg['start_val']:
                max_idx = seg['start_idx']
        # 用对应分段的线性函数预测
        min_pred = int(segs[-1]['slope'] * min_val + segs[-1]['intercept'])
        max_pred = int(segs[-1]['slope'] * max_val + segs[-1]['intercept'])
        for seg in segs:
            if min_val >= seg['start_val']:
                min_pred = int(seg['slope'] * min_val + seg['intercept'])
            if max_val >= seg['start_val']:
                max_pred = int(seg['slope'] * max_val + seg['intercept'])
        return max(0, min_pred), min(len(self.values), max_pred+1)

class BottomCluster:
    def __init__(self, pl, qs, sr=None, num_partitions=10, error_threshold=1.0):
        self.points = pl
        self.queries = qs
        self.num_partitions = num_partitions
        self.error_threshold = error_threshold
        self.partitions, self.split_points = self._flat_partition()
        self.mbrs = self._calculate_mbrs(self.split_points)
        # 为每个分区构建PLM模型
        self.plm_models = []
        for part in self.partitions:
            if part:
                xs = [p.x for p in part]
                self.plm_models.append(PLMModel(xs, delta=error_threshold))
            else:
                self.plm_models.append(None)

    def _flat_partition(self):
        """只在X轴上进行扁平化分割，返回分区和分割点"""
        if not self.points:
            return [[] for _ in range(self.num_partitions)], []

        # 强制以X轴为排序和分割维度
        sort_dim = 'x'
        print(f"\n扁平化分割：只在X轴分割")

        sorted_points = sorted(self.points, key=lambda p: getattr(p, sort_dim))
        partition_size = len(sorted_points) // self.num_partitions
        partitions = []
        split_points = []

        for i in range(self.num_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < self.num_partitions - 1 else len(sorted_points)
            partition = sorted_points[start:end]
            partitions.append(partition)
            if i < self.num_partitions - 1:
                split_points.append(getattr(sorted_points[end-1], sort_dim))

        # 输出分割点信息
        print(f"\n分割点信息:")
        print(f"排序维度: X")
        for i, split_point in enumerate(split_points):
            print(f"分区 {i} 和 {i+1} 之间的分割点: X = {split_point:.2f}")

        return partitions, split_points

    def _calculate_mbrs(self, split_points):
        """每个分区的MBR为竖条：X区间×全局Y区间"""
        mbrs = []
        if not self.points:
            return mbrs
        y_values = [p.y for p in self.points]
        global_ymin = min(y_values)
        global_ymax = max(y_values)
        x_edges = [min(p.x for p in self.points)] + split_points + [max(p.x for p in self.points)]
        for i in range(self.num_partitions):
            mbr = {
                'x_min': x_edges[i],
                'x_max': x_edges[i+1],
                'y_min': global_ymin,
                'y_max': global_ymax
            }
            mbrs.append(mbr)
        return mbrs

    def visualize(self, show_queries=True, show_mbrs=True):
        """可视化数据点、分区竖条和查询范围"""
        plt.figure(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_partitions))
        for i, partition in enumerate(self.partitions):
            if not partition:
                continue
            x_values = [p.x for p in partition]
            y_values = [p.y for p in partition]
            plt.scatter(x_values, y_values, c=[colors[i]], alpha=0.6, label=f'Partition {i} ({len(partition)} points)')
        # 画分割线
        for split_x in self.split_points:
            plt.axvline(x=split_x, color='orange', linestyle='-', linewidth=2)
        # 画竖条MBR
        if show_mbrs:
            for mbr in self.mbrs:
                plt.plot([mbr['x_min'], mbr['x_max'], mbr['x_max'], mbr['x_min'], mbr['x_min']],
                         [mbr['y_min'], mbr['y_min'], mbr['y_max'], mbr['y_max'], mbr['y_min']],
                         c='orange', linestyle='--', alpha=0.3)
        # 画查询框
        if show_queries and self.queries:
            for i, query in enumerate(self.queries):
                plt.plot([query.min_x, query.max_x, query.max_x, query.min_x, query.min_x],
                         [query.min_y, query.min_y, query.max_y, query.max_y, query.min_y],
                         'k-', alpha=0.2, label=f'Query {i}' if i < 3 else None)
        plt.title('Data Points, X-Flat Partitions and Queries Visualization')
        plt.xlabel('Attribute 1')
        plt.ylabel('Attribute 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)

    def print_partition_info(self):
        print("\n分区信息:")
        print(f"总分区数: {self.num_partitions}")
        print(f"每个分区的平均点数: {len(self.points) / self.num_partitions:.2f}")
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
        if not query_set:
            print("查询集为空，无法分析相交情况")
            return
        print("\n查询集与分区相交分析:")
        total_matching_points = 0
        for q_idx, query in enumerate(query_set):
            print(f"\n查询 {q_idx}: ({query.min_x}, {query.min_y}) - ({query.max_x}, {query.max_y})")
            matching_points = []
            for i, (partition, mbr, plm) in enumerate(zip(self.partitions, self.mbrs, self.plm_models)):
                if not mbr or not plm:
                    continue
                if (query.min_x <= mbr['x_max'] and query.max_x >= mbr['x_min'] and
                    query.min_y <= mbr['y_max'] and query.max_y >= mbr['y_min']):
                    print(f"\n分区 {i} 与查询相交:")
                    print(f"  MBR: X[{mbr['x_min']:.2f}, {mbr['x_max']:.2f}], Y[{mbr['y_min']:.2f}, {mbr['y_max']:.2f}]")
                    xs = [p.x for p in partition]
                    # 用PLM模型预测索引区间
                    left, right = plm.predict_range(query.min_x, query.max_x)
                    print(f"  PLM预测索引区间: [{left}, {right}) (共{right-left}个候选点)")
                    candidates = partition[left:right]
                    # 进一步用Y过滤
                    partition_matches = [p for p in candidates if query.min_y <= p.y <= query.max_y]
                    print(f"  PLM过滤后实际命中点数: {len(partition_matches)}")
                    if partition_matches:
                        for j, point in enumerate(partition_matches[:3]):
                            print(f"    命中点 {j}: ({point.x}, {point.y})")
                    else:
                        print("    该分区无命中点")
                    matching_points.extend(partition_matches)
            print(f"\n查询 {q_idx} 总结: 匹配点总数: {len(matching_points)}")
            total_matching_points += len(matching_points)
        print(f"\n所有查询的总匹配点数: {total_matching_points}")

if __name__ == '__main__':
    args = parser.parse_args()
    is_skew = True if args.isk == 1 else False
    is_gau = True if args.ig == 1 else False
    is_mixed = True if args.im == 1 else False
    sample_rate = args.sr
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
        root = BottomCluster(data_set, query_set, num_partitions=4, error_threshold=args.et)

        # 打印分区信息
        root.print_partition_info()

        # 分析查询集与分区的相交情况
        root.analyze_query_intersection(query_set)

        # 显示可视化结果
        print("\n正在显示可视化结果...")
        root.visualize(show_queries=True, show_mbrs=True)
    else:
        print("数据集或查询集为空，无法继续...")
