import os
import numpy as np
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import *
from utils.data_utils import add_derived_features

def load_and_preprocess_new_data():
    """
    加载专门放在new_data文件夹中的新数据进行评估
    """
    print("开始加载新测试数据...")
    start_time = time.time()
    all_sequences = []
    all_labels = []
    class_distribution = {class_name: 0 for class_name in CLASS_NAMES}
    new_data_dir = 'new_data'  # 专门的新数据目录
    
    # 添加数据质量检查计数器
    quality_metrics = {
        "files_processed": 0,
        "files_skipped": 0,
        "columns_mismatch": 0,
        "too_short": 0,
        "other_errors": 0
    }
    
    for class_name, class_id in LABEL_MAP.items():
        class_dir = os.path.join(new_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 新数据文件夹 {class_dir} 不存在，跳过")
            continue
            
        print(f"加载新测试数据: {class_name}")
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.csv'):
                quality_metrics["files_processed"] += 1
                file_path = os.path.join(class_dir, filename)
                
                try:
                    # 加载CSV文件
                    df = pd.read_csv(file_path)
                    
                    # 确保列名一致
                    if len(df.columns) != len(SENSOR_COLUMNS):
                        quality_metrics["columns_mismatch"] += 1
                        print(f"错误: 文件 {filename} 的列数不符 ({len(df.columns)}列), 跳过")
                        continue
                    
                    # 添加数据质量检查
                    if df.isnull().sum().sum() > 0:
                        print(f"警告: 文件 {filename} 包含 {df.isnull().sum().sum()} 个缺失值，已填充")
                        df = df.ffill().bfill()
                    
                    # 添加衍生特征
                    df = add_derived_features(df)
                    
                    # 提取传感器数据
                    sensor_data = df[SENSOR_COLUMNS[1:] + ['acc_mag', 'gyro_mag']].values
                    
                    # 检查数据长度
                    if len(sensor_data) < MIN_DATA_LENGTH:
                        quality_metrics["too_short"] += 1
                        print(f"警告: 文件 {filename} 数据点不足 ({len(sensor_data)} < {MIN_DATA_LENGTH})，跳过")
                        continue
                    
                    # 添加到数据列表
                    all_sequences.append(sensor_data)
                    
                    # 为整个文件添加标签
                    all_labels.append(np.full(len(sensor_data), class_id))
                    
                    # 更新类别统计 (转换为Python原生int)
                    class_distribution[class_name] += int(len(sensor_data))
                    
                except Exception as e:
                    quality_metrics["other_errors"] += 1
                    print(f"加载文件 {file_path} 失败: {str(e)}")
    
    load_time = time.time() - start_time
    data_stats = {
        "total_points": int(sum(len(seq) for seq in all_sequences)),
        "class_distribution": class_distribution,
        "load_time_sec": float(load_time),
        "quality_metrics": quality_metrics
    }
    
    print(f"新测试数据加载完成! 共 {data_stats['total_points']} 个数据点, 耗时 {load_time:.2f}秒")
    print(f"新数据类别分布: {class_distribution}")
    print(f"数据质量报告: {quality_metrics}")
    
    return all_sequences, all_labels, data_stats

def create_sequences(sequences, labels, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """从连续数据创建窗口序列和标签"""
    windowed_sequences = []
    windowed_labels = []
    
    # 添加序列长度检查
    sequence_lengths = []
    
    for data, data_labels in zip(sequences, labels):
        num_points = len(data)
        sequence_lengths.append(num_points)
        
        # 跳过太短的文件
        if num_points < window_size:
            continue
            
        # 创建滑动窗口序列
        for i in range(0, num_points - window_size + 1, step_size):
            window = data[i:i + window_size]
            label = data_labels[i + window_size // 2]  # 使用窗口中心的标签
            
            windowed_sequences.append(window)
            windowed_labels.append(label)
    
    # 添加序列长度统计
    sequence_stats = {
        "total_sequences": len(windowed_sequences),
        "min_length": min(sequence_lengths) if sequence_lengths else 0,
        "max_length": max(sequence_lengths) if sequence_lengths else 0,
        "avg_length": np.mean(sequence_lengths) if sequence_lengths else 0
    }
    
    print(f"创建了 {len(windowed_sequences)} 个序列窗口")
    print(f"序列长度统计: 最小={sequence_stats['min_length']}, 最大={sequence_stats['max_length']}, 平均={sequence_stats['avg_length']:.2f}")
    
    return np.array(windowed_sequences), np.array(windowed_labels), sequence_stats

def visualize_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", filename="confusion_matrix.png"):
    """可视化混淆矩阵并保存"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存到结果目录
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    
    print(f"混淆矩阵已保存至: {save_path}")
    return cm

def evaluate_similar_actions(model, X_test, y_test):
    """评估相似动作之间的混淆情况"""
    # 定义相似动作组
    similar_groups = [
        ('Upstairs', 'Uphills'),
        ('DownStairs', 'Uphills'),
        ('Walking', 'Running'),
        ('Upstairs', 'DownStairs'),
        ('Sitting', 'Standing')
    ]
    
    results = {}
    
    # 对每个相似组进行评估
    for group in similar_groups:
        group_name = " vs ".join(group)
        print(f"\n评估相似动作: {group_name}")
        
        # 创建掩码选择该组中的样本
        mask = np.zeros(len(y_test), dtype=bool)
        group_classes = []
        for action in group:
            if action in CLASS_NAMES:
                class_id = CLASS_NAMES.index(action)
                mask = mask | (y_test == class_id)
                group_classes.append(class_id)
        
        X_group = X_test[mask]
        y_group = y_test[mask]
        
        if len(X_group) == 0:
            print(f"没有找到{group_name}的测试样本")
            continue
            
        # 评估
        results_group = model.evaluate(X_group, y_group, verbose=0)
        y_pred = np.argmax(model.predict(X_group), axis=1)
        
        # 动态生成 target_names
        group_class_names = [CLASS_NAMES[class_id] for class_id in group_classes]
        
        # 保存结果
        group_results = {
            "test_loss": float(results_group[0]),
            "test_accuracy": float(results_group[1]),
            "classification_report": classification_report(y_group, y_pred, target_names=group_class_names, zero_division=0)
        }
        
        # 可视化混淆矩阵
        cm_filename = f"confusion_matrix_similar_{'_'.join(group)}.png"
        cm = visualize_confusion_matrix(y_group, y_pred, group_class_names, 
                                  f"相似动作混淆矩阵: {group_name}", 
                                  cm_filename)
        
        # 计算混淆百分比
        confusion_percentages = {}
        for i, true_class in enumerate(group):
            true_idx = group_classes.index(CLASS_NAMES.index(true_class))
            for j, pred_class in enumerate(group):
                if i != j:
                    pred_idx = group_classes.index(CLASS_NAMES.index(pred_class))
                    count = cm[true_idx, pred_idx]
                    total = np.sum(cm[true_idx, :])
                    percentage = count / total * 100 if total > 0 else 0
                    key = f"{true_class}->{pred_class}"
                    confusion_percentages[key] = float(percentage)
        
        group_results["confusion_percentages"] = confusion_percentages
        results[group_name] = group_results
    
    return results

def evaluate_model():
    print("开始模型评估...")
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    # 2. 评估原始测试数据
    if os.path.exists('test_data.npy') and os.path.exists('test_labels.npy'):
        print("\n评估原始测试集...")
        X_test = np.load('test_data.npy')
        y_test = np.load('test_labels.npy')
        
        # 如果有scaler，标准化数据
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            original_shape = X_test.shape
            X_reshaped = X_test.reshape(-1, original_shape[-1])
            X_scaled = scaler.transform(X_reshaped)
            X_test = X_scaled.reshape(original_shape)
        
        # 评估
        results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = results[0]
        test_acc = results[1]
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        print(f"原始测试集损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
        
        # 保存原始测试结果
        orig_results = {
            "test_loss": float(test_loss),
            "测试准确率": float(test_acc),
            "分类报告": classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
        }
        
        with open(os.path.join(RESULTS_DIR, 'original_test_results.json'), 'w') as f:
            json.dump(orig_results, f, indent=2)
            
        visualize_confusion_matrix(y_test, y_pred, CLASS_NAMES, "原始测试集混淆矩阵", "confusion_matrix_original.png")
        
        # 评估相似动作
        similar_results = evaluate_similar_actions(model, X_test, y_test)
        with open(os.path.join(RESULTS_DIR, 'similar_actions_results.json'), 'w') as f:
            json.dump(similar_results, f, indent=2)
    
    # 3. 专门评估新数据
    if any(os.path.exists(os.path.join('new_data', class_name)) for class_name in CLASS_NAMES):
        print("\n评估新数据...")
        
        # 加载新数据
        sequences, labels, data_stats = load_and_preprocess_new_data()
        
        # 创建序列窗口
        X_new, y_new, seq_stats = create_sequences(sequences, labels, WINDOW_SIZE, STEP_SIZE)
        data_stats.update(seq_stats)
        
        if len(X_new) == 0:
            print("错误: 没有足够的新数据创建序列窗口")
            return
            
        print(f"新数据创建了 {len(X_new)} 个序列窗口")
        
        # 使用相同的标准化器
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            original_shape = X_new.shape
            X_reshaped = X_new.reshape(-1, original_shape[-1])
            X_scaled = scaler.transform(X_reshaped)
            X_new = X_scaled.reshape(original_shape)
        
        # 评估新数据
        results = model.evaluate(X_new, y_new, verbose=0)
        new_loss = results[0]
        new_acc = results[1]
        y_pred_new = np.argmax(model.predict(X_new), axis=1)
        new_accuracy = accuracy_score(y_new, y_pred_new)
        
        print(f"\n新数据评估结果:")
        print(f"损失: {new_loss:.4f}")
        print(f"准确率: {new_acc:.4f}")
        print(f"准确率(手工计算): {new_accuracy:.4f}")
        
        # 分类报告
        class_report = classification_report(y_new, y_pred_new, target_names=CLASS_NAMES, zero_division=0)
        print("\n分类报告:")
        print(class_report)
        
        # 可视化混淆矩阵
        cm = visualize_confusion_matrix(y_new, y_pred_new, CLASS_NAMES, "新数据混淆矩阵", "confusion_matrix_new_data.png")
        
        # 评估新数据中的相似动作
        similar_results_new = evaluate_similar_actions(model, X_new, y_new)
        
        # 保存结果
        new_results = {
            "新数据统计": data_stats,
            "新数据损失": float(new_loss),
            "新数据准确率": float(new_acc),
            "分类报告": class_report,
            "相似动作评估": similar_results_new
        }
        
        with open(os.path.join(RESULTS_DIR, 'new_data_results.json'), 'w') as f:
            json.dump(new_results, f, indent=2)
            
        # 置信度分析
        predictions = model.predict(X_new)
        confidences = np.max(predictions, axis=1)
        confidence_thresholds = [0.6, 0.7, 0.8, 0.9]
        confidence_analysis = []
        
        for threshold in confidence_thresholds:
            mask = confidences > threshold
            if np.sum(mask) > 0:
                accuracy_above_threshold = accuracy_score(y_new[mask], y_pred_new[mask])
                coverage = np.mean(mask)
                confidence_analysis.append({
                    "置信度阈值": float(threshold),
                    "准确率": float(accuracy_above_threshold),
                    "覆盖率": float(coverage)
                })
                print(f"置信度 > {threshold:.2f}: 准确率 = {accuracy_above_threshold:.4f}, 覆盖率 = {coverage:.4f}")
        
        if confidence_analysis:
            with open(os.path.join(RESULTS_DIR, 'confidence_analysis.json'), 'w') as f:
                json.dump(confidence_analysis, f, indent=2)
            
            # 可视化置信度分布
            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=20, alpha=0.7, color='blue')
            plt.title('预测置信度分布')
            plt.xlabel('置信度')
            plt.ylabel('样本数量')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.savefig(os.path.join(RESULTS_DIR, 'confidence_distribution.png'))
            plt.close()
    
    print("\n评估完成! 结果保存在 'results' 目录中")

if __name__ == "__main__":
    # 配置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
    evaluate_model()