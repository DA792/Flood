import sys
import os


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = []  # 如果完全不需要 label，可以删除这行

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.x, self.y) == (other.x, other.y)
        else:
            return False

    def update_label(self, qid):
        # 如果完全不需要 label，可以删除这行
        self.label.append(qid)

    def __repr__(self):
        # 不显示 label
        return f"Point(x={self.x}, y={self.y})"


class Query(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        # self.keys = keys


class TrainNode:  # 构造函数
    def __init__(self, pl, qs, id=-1, minx=None, miny=None, maxx=None, maxy=None):  # 构造函数
        self.point_list = pl
        self.query_set = qs
        self.id = id
        if minx is None:
            self.min_x, self.min_y, self.max_x, self.max_y = self.update_mbr(self.point_list)  # 更新 MBR
        else:
            self.min_x = minx
            self.min_y = miny
            self.max_x = maxx
            self.max_y = maxy

    def __lt__(self, other):
        # priority: > - more first; < - less first
        return len(self.query_set) > len(other.query_set)

    def update_query_set(self, qs):  # 更新 query_set
        for q in qs:
            max_min_x_coord = max(q.min_x, self.min_x)
            min_max_x_coord = min(q.max_x, self.max_x)
            max_min_y_coord = max(q.min_y, self.min_y)
            min_max_y_coord = min(q.max_y, self.max_y)
            if max_min_x_coord < min_max_x_coord and max_min_y_coord < min_max_y_coord:
                self.query_set.append(q)

    @staticmethod  # 静态方法
    def update_mbr(pl):
        min_x = sys.float_info.max
        min_y = sys.float_info.max
        max_x = -sys.float_info.max
        max_y = -sys.float_info.max
        for point in pl:
            if point.x < min_x:
                min_x = point.x
            if point.x > max_x:
                max_x = point.x
            if point.y < min_y:
                min_y = point.y
            if point.y > max_y:
                max_y = point.y
        return min_x, min_y, max_x, max_y   # 返回四个坐标组成最小矩形

class FileUtils:  # 文件操作类
    def __init__(self, dataset, low_sel, high_sel, ratio):
        self.dataset = dataset  # 数据集名称
        self.low_sel = low_sel  # 下限选择度
        self.high_sel = high_sel
        self.ratio = ratio
        # 获取项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def read_data(self, data=None):
        point_list = []
        # 构建数据文件路径
        data_file = os.path.join(self.root_dir, 'data', 'data', f'{self.dataset if data is None else data}.csv')
        
        try:
            with open(data_file) as f:
                while True:
                    line = f.readline()
                    if line:
                        component = line.strip().split(' ')
                        point = Point(float(component[0]), float(component[1]))
                        point_list.append(point)
                    else:
                        break
            print('Read Data Done')
            return point_list
        except FileNotFoundError:
            print(f"Error: Could not find data file at {data_file}")
            print("Please make sure the data file exists in the correct location.")
            return []







