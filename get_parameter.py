import tensorflow as tf
import numpy as np
import joblib
import json

def get_model_info(model_path):
    """获取TFLite模型的输入输出信息"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return {
        "input_shape": input_details[0]['shape'].tolist(),
        "input_dtype": input_details[0]['dtype'].__name__,
        "output_shape": output_details[0]['shape'].tolist(),
        "output_dtype": output_details[0]['dtype'].__name__
    }

def get_scaler_info(scaler_path):
    """获取标准化器的均值和标准差"""
    scaler = joblib.load(scaler_path)
    return {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist()
    }

def main():
    # 配置文件路径
    MODEL_PATH = 'models/motion_model.tflite'
    SCALER_PATH = 'models/scaler.pkl'
    OUTPUT_FILE = 'android_config.json'
    
    # 获取模型信息
    model_info = get_model_info(MODEL_PATH)
    
    # 获取标准化参数
    scaler_info = get_scaler_info(SCALER_PATH)
    
    # 合并所有配置
    config = {
        "model_config": model_info,
        "scaler_config": scaler_info,
        "feature_names": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "accel_mag", "gyro_mag"],
        "class_names": ["DownStairs", "Running", "Static", "Uphills", "Upstairs", "Walking"]
    }
    
    # 保存为JSON文件
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 打印信息以便复制
    print("="*50)
    print("Android应用所需配置信息：")
    print("="*50)
    print(f"模型输入形状: {model_info['input_shape']}")
    print(f"模型输入类型: {model_info['input_dtype']}")
    print(f"模型输出形状: {model_info['output_shape']}")
    print(f"模型输出类型: {model_info['output_dtype']}")
    print("-"*50)
    print(f"标准化均值: {scaler_info['mean']}")
    print(f"标准化标准差: {scaler_info['std']}")
    print("-"*50)
    print(f"特征名称: {config['feature_names']}")
    print(f"类别名称: {config['class_names']}")
    print("="*50)
    print(f"完整配置已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()