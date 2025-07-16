import os
import numpy as np
import pandas as pd
import joblib
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from config import *

def load_and_preprocess_data():
    """
    加载所有CSV文件并进行预处理 - 修改版：截取每段数据的中间部分
    """
    print("开始加载数据...")
    start_time = time.time()
    all_sequences = []
    all_labels = []
    file_count = 0
    class_distribution = {class_name: 0 for class_name in CLASS_NAMES}
    
    TRIM_RATIO = 0.15  # 截取比例，去掉前15%和后15%的数据
    
    # 添加数据质量检查
    quality_metrics = {
        "files_processed": 0,
        "files_skipped": 0,
        "columns_mismatch": 0,
        "missing_values": 0,
        "too_short": 0,
        "other_errors": 0
    }
    
    for class_name, class_id in LABEL_MAP.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 文件夹 {class_dir} 不存在，跳过")
            continue
            
        print(f"加载 {class_name} 数据...")
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.csv'):
                file_count += 1
                quality_metrics["files_processed"] += 1
                file_path = os.path.join(class_dir, filename)
                
                try:
                    # 加载CSV文件
                    df = pd.read_csv(file_path)
                    
                    # 确保列名一致
                    if len(df.columns) != len(SENSOR_COLUMNS):
                        quality_metrics["columns_mismatch"] += 1
                        print(f"文件 {filename} 的列数不符, 跳过")
                        continue
                    
                    df.columns = SENSOR_COLUMNS
                    
                    # 检查缺失值
                    if df.isnull().sum().sum() > 0:
                        quality_metrics["missing_values"] += 1
                        print(f"警告: 文件 {filename} 包含 {df.isnull().sum().sum()} 个缺失值，已填充")
                        df = df.ffill().bfill()
                    
                    # 添加衍生特征
                    df = add_derived_features(df)
                    
                    # 提取传感器数据
                    sensor_data = df[SENSOR_COLUMNS[1:] + ['acc_mag', 'gyro_mag']].values
                    
                    # 截取中间部分 - 去掉前后15%
                    total_points = len(sensor_data)
                    start_index = int(total_points * TRIM_RATIO)
                    end_index = total_points - int(total_points * TRIM_RATIO)
                    
                    # 确保有足够的数据点
                    if end_index - start_index < MIN_DATA_LENGTH:
                        quality_metrics["too_short"] += 1
                        print(f"警告: 文件 {filename} 截取后数据点不足 ({end_index - start_index} < {MIN_DATA_LENGTH})，跳过")
                        continue
                    
                    sensor_data = sensor_data[start_index:end_index]
                    
                    # 添加到数据列表
                    all_sequences.append(sensor_data)
                    
                    # 为整个文件添加标签
                    all_labels.append(np.full(len(sensor_data), class_id))
                    
                    # 更新类别统计
                    class_distribution[class_name] += int(len(sensor_data))
                    
                    # 数据增强 - 为每个文件生成多个增强版本
                    if ENABLE_DATA_AUGMENTATION:
                        for _ in range(AUGMENTATION_FACTOR):
                            augmented_data = augment_sensor_data(sensor_data, class_name)
                            all_sequences.append(augmented_data)
                            all_labels.append(np.full(len(augmented_data), class_id))
                            class_distribution[class_name] += int(len(augmented_data))
                    
                except Exception as e:
                    quality_metrics["other_errors"] += 1
                    print(f"加载文件 {file_path} 失败: {str(e)}")
    
    load_time = time.time() - start_time
    data_stats = {
        "total_files": file_count,
        "total_points": int(sum(len(seq) for seq in all_sequences)),  # 转换为int
        "class_distribution": class_distribution,
        "load_time_sec": float(load_time),  # 转换为float
        "quality_metrics": quality_metrics
    }
    
    # 保存数据统计 (确保所有数值都是JSON可序列化的)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'data_statistics.json'), 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    print(f"数据加载完成! 共 {file_count} 个文件, {data_stats['total_points']} 个数据点, 耗时 {load_time:.2f}秒")
    print(f"类别分布: {class_distribution}")
    print(f"数据质量报告: {quality_metrics}")
    
    return all_sequences, all_labels

def augment_sensor_data(data, class_name):
    """
    对传感器数据进行增强，特别关注相似动作的区分
    """
    # 1. 添加高斯噪声 - 根据动作类型调整噪声强度
    noise_level = 0.02 + np.random.uniform(0, 0.03)
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    
    # 2. 时间扭曲（加速/减速） - 特别针对动态动作
    orig_len = len(data)
    time_warp_factor = np.random.uniform(0.9, 1.1)
    new_len = int(orig_len * time_warp_factor)
    indices = np.linspace(0, orig_len-1, new_len).astype(int)
    warped_data = noisy_data[indices]
    
    # 3. 幅度缩放 - 特别针对垂直运动（上下楼梯/坡）
    scale_factors = np.random.uniform(0.8, 1.2, data.shape[1])
    scaled_data = warped_data * scale_factors
    
    # 4. 通道交换（模拟传感器方向变化） - 概率调整为50%
    if np.random.rand() > 0.5:  # 50%的概率交换轴
        # 随机选择两个轴交换
        swap_axes = np.random.choice([0, 1, 2, 3, 4, 5], size=2, replace=False)
        scaled_data[:, swap_axes] = scaled_data[:, swap_axes[::-1]]
    
    # 5. 添加针对相似动作的特定变换
    # 垂直运动增强（上/下楼梯/坡）
    if class_name in ['Upstairs', 'DownStairs', 'Uphills', 'Downhills']:
        # 添加垂直加速度变化
        vertical_scale = np.random.uniform(0.7, 1.3)
        scaled_data[:, 2] = scaled_data[:, 2] * vertical_scale
        
        # 添加周期性波动
        if np.random.rand() > 0.7:
            steps = np.linspace(0, 4 * np.pi, len(scaled_data))
            vertical_oscillation = 0.1 * np.sin(steps)
            scaled_data[:, 2] += vertical_oscillation
    
    # 行走和跑步增强
    elif class_name in ['Walking', 'Running']:
        # 添加水平方向增强
        horizontal_scale = np.random.uniform(0.8, 1.2, 2)
        scaled_data[:, 0] *= horizontal_scale[0]
        scaled_data[:, 1] *= horizontal_scale[1]
        
        # 添加步频变化
        if np.random.rand() > 0.6:
            frequency_factor = np.random.uniform(0.8, 1.2)
            time_vector = np.arange(len(scaled_data))
            step_pattern = np.sin(2 * np.pi * frequency_factor * time_vector / 50)
            scaled_data[:, 0] += 0.05 * step_pattern
            scaled_data[:, 1] += 0.05 * np.cos(2 * np.pi * frequency_factor * time_vector / 50)
    
    # 静态姿势增强（坐/站）
    elif class_name in ['Sitting', 'Standing']:
        # 减少运动噪声
        scaled_data[:, :3] *= np.random.uniform(0.9, 1.1)
        
        # 添加微小抖动
        if np.random.rand() > 0.8:
            micro_movements = np.random.normal(0, 0.02, scaled_data.shape)
            scaled_data += micro_movements
    
    return scaled_data

def add_derived_features(df):
    """添加加速度和陀螺仪的幅值特征"""
    # 计算加速度幅度
    df['acc_mag'] = np.sqrt(df['加速度计X']**2 + df['加速度计Y']**2 + df['加速度计Z']**2)
    
    # 计算陀螺仪幅度
    df['gyro_mag'] = np.sqrt(df['陀螺仪X']**2 + df['陀螺仪Y']**2 + df['陀螺仪Z']**2)
    
    # 添加垂直加速度分量（减去重力）
    df['acc_vertical'] = df['加速度计Z'] - 9.8
    
    # 添加水平加速度
    df['acc_horizontal'] = np.sqrt(df['加速度计X']**2 + df['加速度计Y']**2)
    
    return df

def create_sequences_and_labels(all_sequences, all_labels):
    """
    从连续数据创建窗口序列和标签
    返回: (X, y) 序列数据和标签
    """
    start_time = time.time()
    sequences = []
    labels = []
    class_window_counts = {class_name: 0 for class_name in CLASS_NAMES}  # 初始化为整数
    
    for data, data_labels in zip(all_sequences, all_labels):
        num_points = len(data)
        
        # 跳过太短的文件
        if num_points < WINDOW_SIZE:
            continue
            
        # 创建滑动窗口序列
        for i in range(0, num_points - WINDOW_SIZE + 1, STEP_SIZE):
            window = data[i:i + WINDOW_SIZE]
            label = data_labels[i + WINDOW_SIZE // 2]  # 使用窗口中心的标签
            
            sequences.append(window)
            labels.append(label)
            
            # 更新类别计数 (转换为Python原生int)
            class_name = CLASS_NAMES[label]
            class_window_counts[class_name] = class_window_counts.get(class_name, 0) + 1
            
    process_time = time.time() - start_time
    print(f"创建了 {len(sequences)} 个序列窗口, 耗时 {process_time:.2f}秒")
    print(f"窗口分布: {class_window_counts}")
    
    # 保存窗口分布 (确保所有值都是JSON可序列化的)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'window_distribution.json'), 'w') as f:
        json.dump(class_window_counts, f, indent=2)
    
    return np.array(sequences), np.array(labels)

def preprocess_and_split(X, y, scaler=None):
    """
    预处理数据并拆分为训练集和测试集
    返回: (X_train, X_test, y_train, y_test, scaler)
    """
    start_time = time.time()
    
    # 1. 标准化特征
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        # 保存标准化器
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print("标准化器已保存")
    else:
        X_scaled = scaler.transform(X_reshaped)
        
    X = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # 2. 随机打乱数据
    X, y = shuffle(X, y, random_state=42)
    
    # 3. 划分训练集和测试集 (分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=42
    )
    
    process_time = time.time() - start_time
    print(f"预处理完成! 训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}, 耗时 {process_time:.2f}秒")
    
    # 4. 记录训练集和测试集分布 (确保所有数值都是JSON可序列化的)
    class_distribution = {
        "train": {},
        "test": {}
    }
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        # 转换为Python原生int
        train_count = int(np.sum(y_train == class_id))
        test_count = int(np.sum(y_test == class_id))
        class_distribution["train"][class_name] = train_count
        class_distribution["test"][class_name] = test_count
        
    with open(os.path.join(RESULTS_DIR, 'split_distribution.json'), 'w') as f:
        json.dump(class_distribution, f, indent=2)
    
    return X_train, X_test, y_train, y_test, scaler