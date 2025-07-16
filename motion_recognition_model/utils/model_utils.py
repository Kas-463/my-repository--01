import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, 
    Dropout, BatchNormalization, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.optimizers import Adam
from config import WINDOW_SIZE, NUM_FEATURES, NUM_CLASSES, LEARNING_RATE

def build_cnn_lstm_model():
    """使用功能API构建优化的CNN-LSTM模型"""
    # 输入层
    inputs = Input(shape=(WINDOW_SIZE, NUM_FEATURES))
    
    # 1. 卷积块
    x = Conv1D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)
    
    # 2. LSTM时序处理
    lstm_out = LSTM(256, return_sequences=True, kernel_initializer='he_normal')(x)
    lstm_out = LayerNormalization()(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    
    # 3. 自注意力机制
    # 对卷积输出的query和value使用相同的输入
    att_out = MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        dropout=0.2
    )(query=lstm_out, value=lstm_out)
    
    # 4. 整合LSTM和注意力输出
    x = tf.keras.layers.Concatenate(axis=-1)([lstm_out, att_out])
    
    # 5. 二次LSTM处理
    x = LSTM(128, return_sequences=False, kernel_initializer='he_normal')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # 6. 分类头
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name="MotionSense-CNN-LSTM-Att")
    
    # 编译模型
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    
    print("优化的功能API模型构建成功 (带MultiHeadAttention)")
    return model