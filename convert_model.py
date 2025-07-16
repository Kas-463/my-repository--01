import tensorflow as tf
import numpy as np
import joblib
from tensorflow.lite.python import lite_constants

def convert_model():
    try:
        # 1. 加载训练好的模型
        model = tf.keras.models.load_model('models/motion_model.keras')
        
        # 2. 创建一个具体的TensorFlow函数
        input_shape = (1, 100, 8)  # (batch_size, window_size, num_features)
        func = tf.function(model).get_concrete_function(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32)
        )
        
        # 3. 创建转换器并设置优化选项
        converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用TFLite内置操作
            tf.lite.OpsSet.SELECT_TF_OPS     # 启用TensorFlow操作（如果需要）
        ]
        
        # 4. 尝试转换模型
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"标准转换失败: {e}")
            # 回退到实验性转换器
            converter.experimental_new_converter = True
            tflite_model = converter.convert()
        
        # 5. 保存TFLite模型
        with open('models/motion_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("模型转换成功！保存为 models/motion_model.tflite")
        
        # 6. 保存标准化参数
        scaler = joblib.load('models/scaler.pkl')
        np.save('models/scaler_mean.npy', scaler.mean_)
        np.save('models/scaler_std.npy', scaler.scale_)
        
        print("标准化参数已保存")
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        # 详细错误信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_model()