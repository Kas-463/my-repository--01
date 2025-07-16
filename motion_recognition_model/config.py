import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据处理参数
WINDOW_SIZE = 100      # 2秒窗口 (50Hz * 2s)
STEP_SIZE = 50          # 滑动窗口步长
NUM_FEATURES = 8        # 8个特征 (6原始+2衍生)

# 新增配置参数
MIN_DATA_LENGTH = 100  # 最小数据长度要求
AUGMENTATION_FACTOR = 2  # 每个样本的增强倍数

# 相似动作组定义
SIMILAR_ACTION_GROUPS = [
    ('Upstairs', 'Uphills'),
    ('DownStairs', 'Uphills'),
    ('Walking', 'Running'),
    ('Upstairs', 'DownStairs'),
    ('Sitting', 'Standing')
]

# 模型参数
NUM_CLASSES = 6         # 6种运动类别
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15  # 训练集内部的验证集比例
TEST_SPLIT = 0.2         # 测试集比例

# 标签映射
LABEL_MAP = {
    'DownStairs': 0,
    'Running': 1,
    'Static': 2,
    'Uphills': 3,
    'Upstairs': 4,
    'Walking': 5
}

# 类别名称列表
CLASS_NAMES = ['DownStairs', 'Running', 'Static', 'Uphills', 'Upstairs', 'Walking']

# 路径配置
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'motion_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 在 config.py 中添加以下配置
ENABLE_DATA_AUGMENTATION = True  # 启用数据增强
AUGMENTATION_FACTOR = 5          # 每个原始文件生成5个增强版本

# 传感器原始列名
SENSOR_COLUMNS = ['时间戳', '加速度计X', '加速度计Y', '加速度计Z', '陀螺仪X', '陀螺仪Y', '陀螺仪Z']

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)