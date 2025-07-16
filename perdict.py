import os
import json
import joblib
import numpy as np
import time
import threading
import random
from tensorflow.keras.models import load_model
from config import *

class MotionClassifier:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """
        初始化运动分类器
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
            
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.buffer = []
        self.window_size = WINDOW_SIZE
        self.step = max(STEP_SIZE // 2, 1)  # 每次预测后移动半个窗口
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_history = []
        self.start_time = time.time()
        self.prediction_count = 0
        
        print(f"运动分类器初始化完成! 支持识别 {len(CLASS_NAMES)} 种运动类型:")
        print(", ".join(CLASS_NAMES))
        print(f"模型加载自: {os.path.abspath(model_path)}")
        print(f"标准化器加载自: {os.path.abspath(scaler_path)}")
        
    def preprocess_data_point(self, raw_data):
        """
        预处理单个传感器数据点
        raw_data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        返回: 包含衍生特征并标准化后的数据
        """
        # 确保输入为6个值
        if len(raw_data) != 6:
            raise ValueError(f"需要6个传感器值，收到 {len(raw_data)} 个")
            
        # 计算幅值特征
        acc_mag = np.sqrt(raw_data[0]**2 + raw_data[1]**2 + raw_data[2]**2)
        gyro_mag = np.sqrt(raw_data[3]**2 + raw_data[4]**2 + raw_data[5]**2)
        
        # 组合特征 (6原始 + 2衍生 = 8个特征)
        features = np.array([*raw_data, acc_mag, gyro_mag])
        
        # 标准化
        return self.scaler.transform([features])[0]
    
    def update_and_predict(self, raw_data):
        """
        更新新数据点并尝试进行预测
        raw_data: 包含6个值的传感器数据
        返回: (预测类别名称, 置信度) 或 None
        """
        # 1. 预处理数据点
        preprocessed = self.preprocess_data_point(raw_data)
        self.buffer.append(preprocessed)
        self.prediction_count += 1
        
        # 2. 当缓冲区有足够数据时进行预测
        if len(self.buffer) >= self.window_size:
            window = np.array(self.buffer[-self.window_size:])
            predictions = self.model.predict(window[np.newaxis, ...], verbose=0)[0]
            
            # 获取预测结果
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            class_name = CLASS_NAMES[predicted_class]
            
            # 保留最近预测结果
            self.last_prediction = class_name
            self.last_confidence = confidence
            
            # 记录预测历史 (确保所有值都可以序列化为JSON)
            self.prediction_history.append({
                "time": time.time() - self.start_time,
                "prediction": class_name,
                "confidence": float(confidence),
                "all_probs": [float(p) for p in predictions]  # 转换为Python列表
            })
            
            # 3. 移动窗口 - 保留最近半窗口的数据
            keep_size = max(self.window_size - self.step, 1)
            self.buffer = self.buffer[-keep_size:]
            
            return class_name, confidence
        
        return None
    
    def get_last_prediction(self):
        """获取最近的预测结果"""
        return self.last_prediction, self.last_confidence
    
    def save_prediction_history(self, file_path="prediction_history.json"):
        """保存预测历史到文件"""
        if not self.prediction_history:
            print("没有预测历史可保存")
            return
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({
                "total_predictions": self.prediction_count,
                "predictions": self.prediction_history
            }, f, indent=2)
        
        print(f"预测历史已保存到: {os.path.abspath(file_path)}")

if __name__ == "__main__":
    # 创建运动分类器
    try:
        classifier = MotionClassifier()
    except Exception as e:
        print(f"初始化分类器失败: {str(e)}")
        exit(1)
    
    # 模拟传感器类的数据生成器
    class SensorSimulator:
        def __init__(self):
            # 不同运动类型的传感器特征模式
            self.movement_patterns = {
                'DownStairs': ([0.2, 0.1, 9.8], [0.5, 0.2, 0.3]),
                'Running': ([0.8, 0.2, 9.4], [2.0, 1.5, 1.8]),
                'Static': ([0.0, 0.0, 9.8], [0.1, 0.1, 0.1]),
                'Uphills': ([0.3, 0.0, 9.5], [1.0, 0.8, 1.2]),
                'Upstairs': ([0.5, 0.1, 8.5], [1.5, 1.2, 1.8]),
                'Walking': ([0.1, -0.3, 8.0], [1.8, 0.5, 1.5])
            }
            self.current_movement = 'Static'
            self.history = []
            
        def get_data(self):
            # 添加少量随机变化
            base_acc, base_gyro = self.movement_patterns[self.current_movement]
            acc = [val + random.uniform(-0.1, 0.1) for val in base_acc]
            gyro = [val + random.uniform(-0.5, 0.5) for val in base_gyro]
            return np.concatenate([acc, gyro])
    
    # 创建模拟器和分类器
    simulator = SensorSimulator()
    
    # 运动类型变化线程
    def change_movement():
        movements = list(CLASS_NAMES)
        while True:
            time.sleep(10)  # 每10秒更换一次运动类型
            new_movement = random.choice(movements)
            simulator.current_movement = new_movement
            simulator.history.append({
                "time": time.time() - classifier.start_time,
                "movement": new_movement
            })
            print(f"\n[状态切换] 当前运动: {new_movement}\n")
    
    # 启动运动变化线程
    threading.Thread(target=change_movement, daemon=True).start()
    
    print("开始运动识别模拟... (按Ctrl+C退出)")
    print("=" * 70)
    
    try:
        start_time = time.time()
        last_print_time = start_time
        print_interval = 0.5  # 每0.5秒打印一次结果
        
        while True:
            # 获取模拟传感器数据
            raw_data = simulator.get_data()
            
            # 更新分类器并获取预测结果
            prediction = classifier.update_and_predict(raw_data)
            
            current_time = time.time()
            if current_time - last_print_time > print_interval:
                last_print_time = current_time
                
                if prediction:
                    class_name, confidence = prediction
                    actual = simulator.current_movement
                    correct = class_name == actual
                    
                    print(f"预测: {class_name:<10} 置信度: {confidence:.2f} | "
                          f"实际: {actual:<10} {'✓' if correct else '✗'}")
                else:
                    buffer_size = len(classifier.buffer)
                    needed = classifier.window_size - buffer_size
                    print(f"缓冲数据中... ({buffer_size}/{classifier.window_size}, 还需{needed}个样本)")
            
            # 50Hz频率 (0.02秒)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        run_time = time.time() - start_time
        print(f"\n模拟结束! 总运行时间: {run_time:.2f}秒")
        
        # 保存预测历史
        classifier.save_prediction_history(
            os.path.join(RESULTS_DIR, 'simulation_predictions.json')
        )
        
        # 保存运动历史
        with open(os.path.join(RESULTS_DIR, 'simulation_movements.json'), 'w') as f:
            json.dump(simulator.history, f, indent=2)
        
        print("结果已保存到结果目录")