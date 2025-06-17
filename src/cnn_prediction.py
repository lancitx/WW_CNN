import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import tensorflow as tf
import random
import os


# 设置随机种子以确保结果可重现
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()


set_seeds(42)

# 修复字体显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 如果有中文需求，可以取消下面的注释
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# 原始数据 - 修正第17行数据
data = np.array([
    [5.33, 5.39, 5.29, 5.41, 5.45, 5.50],
    [5.39, 5.29, 5.41, 5.50, 5.57, 5.57],
    [5.29, 5.41, 5.45, 5.50, 5.57, 5.58],
    [5.41, 5.45, 5.50, 5.57, 5.58, 5.61],
    [5.45, 5.50, 5.57, 5.58, 5.61, 5.69],
    [5.50, 5.57, 5.58, 5.61, 5.69, 5.78],
    [5.57, 5.58, 5.61, 5.69, 5.78, 5.78],
    [5.58, 5.61, 5.69, 5.78, 5.78, 5.81],
    [5.61, 5.69, 5.78, 5.78, 5.81, 5.86],
    [5.69, 5.78, 5.78, 5.81, 5.86, 5.90],
    [5.78, 5.78, 5.81, 5.86, 5.90, 5.97],
    [5.78, 5.81, 5.86, 5.90, 5.97, 6.49],
    [5.81, 5.86, 5.90, 5.97, 6.49, 6.60],
    [5.86, 5.90, 5.97, 6.49, 6.60, 6.64],
    [5.90, 5.97, 6.49, 6.60, 6.64, 6.74],
    [5.97, 6.49, 6.60, 6.64, 6.74, 6.87],
    [6.49, 6.60, 6.64, 6.74, 6.87, 7.01]  # 修正了7.87为6.87
])


# 简化的数据预处理
def prepare_data_simple(data, test_size=3):
    X = data[:, :5]  # 前5列作为输入特征
    y = data[:, 5]  # 最后一列作为输出目标

    # 划分数据集
    split_idx = len(data) - test_size
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 使用MinMaxScaler，范围设定为略宽一些，避免过度约束
    scaler_X = MinMaxScaler(feature_range=(-0.5, 1.5))
    scaler_y = MinMaxScaler(feature_range=(-0.5, 1.5))

    # 数据标准化
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # 重塑数据为3D格式
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    return (X_train_3d, X_test_3d, y_train, y_test,
            y_train_scaled, scaler_X, scaler_y)


# 简化的CNN模型 - 针对小数据集优化
def build_simple_cnn_model(input_shape=(5, 1)):
    model = Sequential()

    # 非常简单的架构，避免过拟合
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(Dropout(0.1))  # 很小的dropout

    model.add(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())

    # 简单的全连接层
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))  # 输出层

    return model


# 训练函数 - 使用更保守的参数
def train_model_conservative(model, X_train, y_train, epochs=200, batch_size=2):
    # 编译模型 - 使用更小的学习率
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='mse',
                  metrics=['mae'])

    # 简单的早停，防止过拟合
    callbacks = [
        EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
    ]

    # 训练模型 - 不使用验证集，因为数据太少
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=callbacks)

    return history


# 评估函数
def evaluate_model_safe(model, X_train, X_test, y_train, y_test, scaler_y):
    # 预测
    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_test_pred_scaled = model.predict(X_test, verbose=0)

    # 反标准化
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

    # 计算评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    # 测试集评估
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    test_mae = np.mean(np.abs(y_test_pred - y_test))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    return {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_mape': test_mape
    }


# 简化的可视化函数
def plot_results_simple(history, y_train, y_test, results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 损失曲线
    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax1.set_title('Model Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 预测结果比较
    ax2 = axes[1]
    train_indices = range(1, len(y_train) + 1)
    test_indices = range(len(y_train) + 1, len(y_train) + len(y_test) + 1)

    ax2.plot(train_indices, y_train, 'bo-', label='Training Actual', markersize=8, linewidth=2)
    ax2.plot(train_indices, results['y_train_pred'], 'b^--', label='Training Predicted',
             alpha=0.8, markersize=8, linewidth=2)
    ax2.plot(test_indices, y_test, 'ro-', label='Test Actual', markersize=10, linewidth=2)
    ax2.plot(test_indices, results['y_test_pred'], 'rs--', label='Test Predicted',
             alpha=0.8, markersize=10, linewidth=2)

    ax2.set_title(f"Prediction Results (Train R²={results['train_r2']:.3f}, Test R²={results['test_r2']:.3f})",
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Mileage (10^4 km)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 真实值vs预测值散点图
    ax3 = axes[2]
    ax3.scatter(y_train, results['y_train_pred'], color='blue', alpha=0.8, s=80,
                label=f"Training (R²={results['train_r2']:.3f})")
    ax3.scatter(y_test, results['y_test_pred'], color='red', alpha=0.8, s=120,
                label=f"Test (R²={results['test_r2']:.3f})")

    # 理想预测线
    all_values = np.concatenate([y_train, y_test])
    min_val, max_val = all_values.min() * 0.95, all_values.max() * 1.05
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

    ax3.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Actual Values (10^4 km)')
    ax3.set_ylabel('Predicted Values (10^4 km)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('../img/cnn_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    print("=== Fixed CNN Time Series Prediction Model ===\n")

    # 数据准备
    (X_train_3d, X_test_3d, y_train, y_test,
     y_train_scaled, scaler_X, scaler_y) = prepare_data_simple(data)

    print(f"Training set shape: {X_train_3d.shape}")
    print(f"Test set shape: {X_test_3d.shape}")
    print(f"Training data range: {y_train.min():.3f} - {y_train.max():.3f}")
    print(f"Test data range: {y_test.min():.3f} - {y_test.max():.3f}\n")

    # 构建简化模型
    model = build_simple_cnn_model()
    model.summary()

    # 训练模型
    print("\nStarting training...")
    history = train_model_conservative(model, X_train_3d, y_train_scaled)

    # 评估模型
    results = evaluate_model_safe(model, X_train_3d, X_test_3d, y_train, y_test, scaler_y)

    # 打印结果
    print(f"\n=== Model Evaluation Results ===")
    print(f"Training R²: {results['train_r2']:.4f}")
    print(f"Training MAPE: {results['train_mape']:.2f}%")
    print(f"Test R²: {results['test_r2']:.4f}")
    print(f"Test MSE: {results['test_mse']:.6f}")
    print(f"Test MAE: {results['test_mae']:.6f}")
    print(f"Test MAPE: {results['test_mape']:.2f}%")

    print(f"\n=== Prediction Results ===")
    for i, (true, pred) in enumerate(zip(y_test, results['y_test_pred'])):
        sample_num = len(y_train) + i + 1
        error = abs(pred - true)
        error_pct = (error / true) * 100
        print(f"Sample {sample_num}: Actual = {true:.4f}, Predicted = {pred:.4f}, "
              f"Error = {error:.4f} ({error_pct:.2f}%)")

    # 可视化
    plot_results_simple(history, y_train, y_test, results)

    return model, results


# 运行主函数
if __name__ == "__main__":
    model, results = main()

    # 额外的模型诊断
    print(f"\n=== Model Diagnostics ===")
    print(f"Model complexity: {model.count_params()} parameters")
    print(f"Overfitting indicator: {results['train_r2'] - results['test_r2']:.4f}")
    if results['train_r2'] - results['test_r2'] > 0.3:
        print("⚠️  Warning: Possible overfitting detected!")
    elif results['test_r2'] > 0.7:
        print("✅ Good generalization!")
    elif results['test_r2'] > 0.5:
        print("🔶 Acceptable performance")
    else:
        print("❌ Poor performance - consider simpler model")