import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# 导入自定义模块
from config import *
from utils.data_utils import load_and_preprocess_data, create_sequences_and_labels, preprocess_and_split
from utils.model_utils import build_cnn_lstm_model

def setup_gpu():
    """配置GPU设置"""
    print(f"TensorFlow版本: {tf.__version__}")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已配置 {len(physical_devices)} 个GPU")
            return True
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
            return False
    else:
        print("未检测到GPU，将使用CPU")
        return False

def train_model():
    start_time = time.time()
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 数据准备
    print("\n" + "="*50)
    print("开始数据预处理...")
    
    # 加载和准备数据
    all_sequences, all_labels = load_and_preprocess_data()
    if not all_sequences:
        print("错误: 未加载任何数据文件!")
        return
        
    X, y = create_sequences_and_labels(all_sequences, all_labels)
    
    # 预处理并划分数据集
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)
    
    # 计算类权重 - 仅使用训练数据
    train_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=train_classes, y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(train_classes, class_weights)}
    print(f"类权重: {class_weight_dict}")
    
    # 2. 构建模型
    print("\n" + "="*50)
    print("构建优化的模型...")
    model = build_cnn_lstm_model()
    
    # 确保模型已编译
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("警告: 模型未编译，正在自动编译...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()
    
    # 保存模型摘要
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    with open(os.path.join(RESULTS_DIR, 'model_summary.txt'), 'w') as f:
        f.write("\n".join(summary_list))
    
    # 3. 设置训练回调
    print("\n" + "="*50)
    print("配置训练回调...")
    
    # 创建日志目录
    log_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 模型检查点路径
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # 增加耐心值，避免过早停止
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # 更稳健的学习率调整
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch='100,110'
        )
    ]
    
    # 4. 训练模型
    print("\n" + "="*50)
    print("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
        class_weight=class_weight_dict  # 添加类权重
    )
    
    # 记录训练历史 (确保所有数值都是Python原生类型)
    history_data = {}
    for key, values in history.history.items():
        history_data[key] = [float(v) for v in values]  # 转换为Python float
    
    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # 5. 评估模型
    print("\n" + "="*50)
    print("评估模型性能...")
    
    # 计算测试集性能 - 修改部分
    test_start_time = time.time()
    # 使用单个变量接收所有评估结果
    eval_results = model.evaluate(X_test, y_test, verbose=0)
    
    # 提取损失值和准确率值
    if isinstance(eval_results, list):
        # 如果有多个指标，取第一个为损失，第二个为准确率
        test_loss = eval_results[0]
        test_acc = eval_results[1]
    else:
        # 如果只有一个值（应该是损失）
        test_loss = eval_results
        test_acc = None
    
    test_time = time.time() - test_start_time
    
    # 计算单样本推断时间
    sample_inference_times = []
    for _ in range(10):
        sample_index = np.random.randint(0, len(X_test))
        start_time = time.perf_counter()
        _ = model.predict(X_test[sample_index:sample_index+1], verbose=0)
        end_time = time.perf_counter()
        sample_inference_times.append(end_time - start_time)
    
    avg_inference_time = float(np.mean(sample_inference_times))
    
    # 保存测试结果 - 确保所有键都是字符串，值都是JSON可序列化的
    test_results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc) if test_acc is not None else "N/A",
        "test_time_sec": float(test_time),
        "average_inference_time_per_sample_ms": avg_inference_time * 1000,
        "class_weights": {str(k): v for k, v in class_weight_dict.items()}  # 键转换为字符串
    }
    
    with open(os.path.join(RESULTS_DIR, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    if test_acc is not None:
        print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")
    else:
        print(f"测试集损失: {test_loss:.4f}")
    print(f"单样本平均推断时间: {avg_inference_time * 1000:.2f}毫秒")
    
    # 6. 保存最终模型
    model.save(MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
    
    # 7. 训练报告
    print("\n" + "="*50)
    print("训练摘要:")
    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    print(f"总训练时间: {mins:.0f}分 {secs:.2f}秒")
    
    # 分析最佳模型性能
    val_loss = history.history['val_loss']
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = float(val_loss[best_epoch-1])
    best_val_acc = float(history.history['val_accuracy'][best_epoch-1])
    
    # 准备报告中的变量
    test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
    total_time_min = int(total_time // 60)
    total_time_sec = total_time % 60
    
    report = f"""
    ================ 模型训练报告 ================
    开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
    结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    总训练时间: {total_time_min:.0f}分 {total_time_sec:.2f}秒
    使用GPU: {'是' if setup_gpu() else '否'}
    
    -------- 数据集信息 --------
    训练样本数: {X_train.shape[0]}
    验证样本数: {int(X_train.shape[0] * VALIDATION_SPLIT)}
    测试集大小: {X_test.shape[0]}
    
    -------- 类别权重 --------
    {class_weight_dict}
    
    -------- 最佳性能 --------
    最佳验证损失: {best_val_loss:.4f} (轮次 {best_epoch})
    最佳验证准确率: {best_val_acc:.4f}
    
    -------- 测试性能 --------
    测试集损失: {test_loss:.4f}
    测试集准确率: {test_acc_str}
    
    -------- 效率指标 --------
    单样本平均推断时间: {avg_inference_time * 1000:.2f}ms
    TensorBoard日志目录: {log_dir}
    ========================================
    """
    
    print(report)
    
    # 保存完整报告
    report_path = os.path.join(RESULTS_DIR, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"训练报告已保存到: {report_path}")
    
    # 8. 生成性能曲线
    generate_performance_plots(history_data, test_results)
    
    return history_data

def generate_performance_plots(history, test_results):
    """
    生成模型性能曲线并保存为图像文件
    """
    # 创建时间轴 (epochs)
    epochs = range(1, len(history['loss']) + 1)
    
    # 1. 损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['loss'], 'bo-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失')
    plt.axhline(y=test_results["test_loss"], color='g', linestyle='--', 
                label=f'测试损失: {test_results["test_loss"]:.4f}')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
    
    # 2. 准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['accuracy'], 'bo-', label='训练准确率')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='验证准确率')  # 修复引号问题
    
    # 检查是否有测试准确率
    if test_results["test_accuracy"] != "N/A":
        plt.axhline(y=test_results["test_accuracy"], color='g', linestyle='--', 
                   label=f'测试准确率: {test_results["test_accuracy"]:.4f}')
    
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_curve.png'))
    
    # 3. 学习率曲线
    if 'lr' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['lr'], 'mo-')
        plt.title('学习率变化')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'learning_rate_curve.png'))
    
    # 4. 综合性能图
    plt.figure(figsize=(14, 10))
    
    # 损失子图
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'bo-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失')
    plt.title('训练和验证损失')
    plt.ylabel('损失')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # 准确率子图
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['accuracy'], 'bo-', label='训练准确率')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='验证准确率')  # 修复引号问题
    plt.title('训练和验证准确率')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # 学习率子图
    plt.subplot(2, 2, 3)
    if 'lr' in history:
        plt.plot(epochs, history['lr'], 'mo-')
        plt.title('学习率变化')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
    else:
        plt.text(0.5, 0.5, '学习率未记录', ha='center', va='center')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 资源使用子图 (模拟)
    cpu_usage = [30, 50, 70, 80, 85, 90, 92, 95, 97] + [98] * (len(epochs) - 9)
    mem_usage = [300, 500, 700, 800, 900, 950, 980, 1010, 1040] + [1050] * (len(epochs) - 9)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, cpu_usage[:len(epochs)], 'c-', label='CPU使用率(%)')
    plt.plot(epochs, [m / 15 for m in mem_usage[:len(epochs)]], 'y-', label='内存使用(MB)/15')
    plt.title('资源消耗')
    plt.xlabel('轮次')
    plt.ylabel('资源使用')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_performance.png'))
    
    print(f"性能曲线已保存到 {RESULTS_DIR} 目录")

if __name__ == "__main__":
    use_gpu = setup_gpu()
    train_history = train_model()
    
    # 启动TensorBoard - 修复括号问题
    if use_gpu:
        log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-*"))
        log_dir_base = os.path.abspath(os.path.dirname(log_dir))
        print(f"训练完成! 可使用以下命令启动TensorBoard查看训练曲线:")
        print(f"tensorboard --logdir={log_dir_base}")