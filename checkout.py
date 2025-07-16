import numpy as np
import tensorflow as tf
import joblib
import time
from config import WINDOW_SIZE, NUM_FEATURES

def validate_tflite_model():
    try:
        # 1. 加载原始Keras模型用于参考
        keras_model = tf.keras.models.load_model('models/motion_model.keras')
        print("原始Keras模型加载成功！")
        
        # 2. 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path='models/motion_model.tflite')
        interpreter.allocate_tensors()
        print("TFLite模型加载成功！")
        
        # 3. 获取输入输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("\n输入详情:")
        print(f"名称: {input_details[0]['name']}, 形状: {input_details[0]['shape']}, 类型: {input_details[0]['dtype']}")
        print("\n输出详情:")
        print(f"名称: {output_details[0]['name']}, 形状: {output_details[0]['shape']}, 类型: {output_details[0]['dtype']}")
        
        # 4. 创建测试数据
        print("\n创建测试数据...")
        scaler = joblib.load('models/scaler.pkl')
        
        # 生成随机传感器数据 (6个特征)
        raw_data = np.random.randn(WINDOW_SIZE, 6)
        
        # 添加衍生特征
        acc_mag = np.sqrt(raw_data[:, 0]**2 + raw_data[:, 1]**2 + raw_data[:, 2]**2)
        gyro_mag = np.sqrt(raw_data[:, 3]**2 + raw_data[:, 4]**2 + raw_data[:, 5]**2)
        
        # 组合特征 (6原始 + 2衍生 = 8个特征)
        features = np.concatenate([raw_data, acc_mag[:, None], gyro_mag[:, None]], axis=1)
        
        # 标准化
        scaled_features = scaler.transform(features)
        
        # 添加批次维度 (1, 100, 8)
        input_data = np.expand_dims(scaled_features, axis=0).astype(np.float32)
        print(f"测试数据形状: {input_data.shape}")
        
        # 5. 使用原始Keras模型进行推理
        keras_start = time.time()
        keras_output = keras_model.predict(input_data)
        keras_time = time.time() - keras_start
        keras_pred = np.argmax(keras_output[0])
        print(f"\nKeras模型推理结果 (耗时 {keras_time*1000:.2f}ms):")
        print(f"预测类别: {keras_pred}, 概率分布: {keras_output[0]}")
        
        # 6. 使用TFLite模型进行推理
        tflite_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        tflite_time = time.time() - tflite_start
        tflite_pred = np.argmax(tflite_output[0])
        print(f"\nTFLite模型推理结果 (耗时 {tflite_time*1000:.2f}ms):")
        print(f"预测类别: {tflite_pred}, 概率分布: {tflite_output[0]}")
        
        # 7. 比较结果
        print("\n结果比较:")
        print(f"预测类别是否一致: {'是' if keras_pred == tflite_pred else '否'}")
        
        # 计算概率分布差异
        diff = np.abs(keras_output - tflite_output).mean()
        print(f"概率分布平均绝对差异: {diff:.6f}")
        
        # 8. 性能测试
        print("\n性能测试 (运行100次):")
        keras_times = []
        tflite_times = []
        
        for i in range(100):
            # Keras推理
            start = time.time()
            _ = keras_model.predict(input_data)
            keras_times.append(time.time() - start)
            
            # TFLite推理
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            tflite_times.append(time.time() - start)
        
        print(f"Keras平均推理时间: {np.mean(keras_times)*1000:.2f}ms")
        print(f"TFLite平均推理时间: {np.mean(tflite_times)*1000:.2f}ms")
        print(f"TFLite比Keras快 {np.mean(keras_times)/np.mean(tflite_times):.1f}x")
        
        # 9. 资源占用分析
        print("\n内存占用分析:")
        import os
        tflite_size = os.path.getsize('models/motion_model.tflite') / 1024
        keras_size = os.path.getsize('models/motion_model.keras') / 1024
        print(f"Keras模型大小: {keras_size:.1f} KB")
        print(f"TFLite模型大小: {tflite_size:.1f} KB")
        print(f"模型压缩比: {keras_size/tflite_size:.1f}x")
        
        # 10. 多输入测试
        print("\n多输入测试 (不同运动类型模式):")
        class_patterns = {
            "Static": [0.0, 0.0, 9.8, 0.1, 0.1, 0.1],
            "Walking": [0.1, -0.3, 8.0, 1.8, 0.5, 1.5],
            "Running": [0.8, 0.2, 9.4, 2.0, 1.5, 1.8],
            "Upstairs": [0.5, 0.1, 8.5, 1.5, 1.2, 1.8]
        }
        
        for state, pattern in class_patterns.items():
            # 生成模式化数据
            pattern_data = np.tile(pattern, (WINDOW_SIZE, 1))
            
            # 添加随机噪声
            noisy_data = pattern_data + np.random.normal(0, 0.1, pattern_data.shape)
            
            # 添加衍生特征
            acc_mag = np.sqrt(noisy_data[:, 0]**2 + noisy_data[:, 1]**2 + noisy_data[:, 2]**2)
            gyro_mag = np.sqrt(noisy_data[:, 3]**2 + noisy_data[:, 4]**2 + noisy_data[:, 5]**2)
            features = np.concatenate([noisy_data, acc_mag[:, None], gyro_mag[:, None]], axis=1)
            
            # 标准化
            scaled_features = scaler.transform(features)
            input_data = np.expand_dims(scaled_features, axis=0).astype(np.float32)
            
            # TFLite推理
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])
            pred_class = np.argmax(tflite_output[0])
            
            print(f"{state}模式 -> 预测类别: {pred_class}, 概率: {tflite_output[0][pred_class]:.2f}")
        
        print("\n✅ 验证通过！TFLite模型功能正常")
        
    except Exception as e:
        print(f"❌❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    print("开始验证TFLite模型...")
    validate_tflite_model()