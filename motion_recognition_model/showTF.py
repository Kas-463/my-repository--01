import tensorflow as tf

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path='models/motion_model.tflite')
interpreter.allocate_tensors()

# 获取模型详情
print("="*50)
print("模型输入详情:")
for detail in interpreter.get_input_details():
    print(f"- 名称: {detail['name']}")
    print(f"  形状: {detail['shape']}")
    print(f"  数据类型: {detail['dtype']}")
    print(f"  量化参数: {detail['quantization_parameters']}")

print("\n" + "="*50)
print("模型输出详情:")
for detail in interpreter.get_output_details():
    print(f"- 名称: {detail['name']}")
    print(f"  形状: {detail['shape']}")
    print(f"  数据类型: {detail['dtype']}")

print("\n" + "="*50)
print("操作符列表:")
for op in interpreter._get_ops_details():
    print(f"- {op['index']}: {op['op_name']} (输入: {op['inputs']}, 输出: {op['outputs']})")