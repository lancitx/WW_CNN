import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import warnings
import os

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')


# 设置随机种子
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)

# 配置matplotlib字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 1. 数据准备与预处理
# -----------------------------
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
    [6.49, 6.60, 6.64, 6.74, 6.87, 7.01]
])

X = data[:, :5]
y = data[:, 5]

# 训练集：前14条；测试集：后3条
X_train = X[:14, :]
y_train = y[:14]
X_test = X[14:, :]
y_test_true = y[14:]

# 归一化/标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# 重塑为CNN输入格式
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# -----------------------------
# 2. 模型构建函数（优化版）
# -----------------------------
def build_cnn_model(hyperparams):
    """构建CNN模型，与WOA保持一致的5个参数"""
    f1 = max(4, min(32, int(round(hyperparams[0]))))  # 限制范围
    f2 = max(2, min(16, int(round(hyperparams[1]))))
    fc = max(4, min(32, int(round(hyperparams[2]))))
    lr = max(1e-4, min(1e-2, float(hyperparams[3])))
    dropout = max(0.1, min(0.4, float(hyperparams[4])))

    model = Sequential([
        Conv1D(filters=f1, kernel_size=2, activation='relu', padding='same', input_shape=(5, 1)),
        Conv1D(filters=f2, kernel_size=2, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(fc, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


# -----------------------------
# 3. 简化的适应度函数（与WOA一致）
# -----------------------------
def fitness_function_simplified(hyperparams):
    """
    简化的适应度函数，使用留出验证而非交叉验证，减少模型创建次数
    """
    # 参数范围检查
    f1, f2, fc, lr, dropout = hyperparams
    if not (4 <= f1 <= 32 and 2 <= f2 <= 16 and 4 <= fc <= 32 and 1e-4 <= lr <= 1e-2 and 0.1 <= dropout <= 0.4):
        return 1e6

    try:
        # 使用简单的留出验证
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_3d, y_train_scaled, test_size=0.25, random_state=42, shuffle=True
        )

        # 构建并训练模型
        model = build_cnn_model(hyperparams)

        # 短时间训练，避免过度拟合
        history = model.fit(
            X_tr, y_tr,
            epochs=15,  # 减少训练轮数
            batch_size=2,
            verbose=0,
            validation_data=(X_val, y_val)
        )

        # 获取最后几个epoch的平均验证损失
        val_losses = history.history['val_loss']
        avg_val_loss = np.mean(val_losses[-3:])  # 最后3个epoch的平均

        # 预测验证集计算R²
        y_pred_scaled = model.predict(X_val, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

        r2 = r2_score(y_true, y_pred)

        # 复杂度惩罚
        complexity = (f1 + f2 + fc) / 100

        # 综合得分：验证损失 + R²惩罚 + 复杂度惩罚
        score = avg_val_loss + 0.3 * max(0, 1 - r2) + 0.05 * complexity

        # 清理模型，释放内存
        del model
        tf.keras.backend.clear_session()

        return float(score)

    except Exception as e:
        print(f"适应度评估出错: {e}")
        return 1e6


# -----------------------------
# 4. BWO算法（统一风格）
# -----------------------------
def beluga_whale_optimization_unified(n_whales=6, max_iter=8):
    """
    BWO算法，统一打印风格
    """
    # 5维参数：[f1, f2, fc, lr, dropout]
    lb = np.array([4, 2, 4, 1e-4, 0.1])
    ub = np.array([32, 16, 32, 1e-2, 0.4])

    dim = 5

    # 初始化白鲸群体
    X = np.random.uniform(low=lb, high=ub, size=(n_whales, dim))
    fitness = np.full(n_whales, np.inf)

    # 记录历史
    best_fitness_history = []

    print("🐳 初始化白鲸群体...")
    # 评估初始群体
    for i in range(n_whales):
        fitness[i] = fitness_function_simplified(X[i])
        print(f"白鲸 {i + 1}: 适应度 = {fitness[i]:.4f}")

    # 找到初始最优
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    best_fit = fitness[best_idx]
    best_fitness_history.append(best_fit)

    print(f"初始最优适应度: {best_fit:.6f}")

    # 迭代优化
    for t in tqdm(range(max_iter), desc="🔍 BWO优化进度"):
        # Beluga Whale 特有参数
        a = 2 - 2 * (t / max_iter)

        improved_count = 0
        for i in range(n_whales):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = random.uniform(-1, 1)
            p = random.random()

            if p < 0.5:
                # Beluga"包围猎物"/"随机搜索"机制
                if abs(A) < 1:
                    # 靠近当前全局最优
                    D = np.abs(C * X_best - X[i])
                    X_new = X_best - A * D
                else:
                    # 选一只随机白鲸进行探索
                    rand_idx = random.randint(0, n_whales - 1)
                    X_rand = X[rand_idx]
                    D = np.abs(C * X_rand - X[i])
                    X_new = X_rand - A * D
            else:
                # Beluga"螺旋更新"机制
                D_best = np.abs(X_best - X[i])
                X_new = D_best * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

            # 边界处理
            X_new = np.clip(X_new, lb, ub)

            # 评估新位置
            f_new = fitness_function_simplified(X_new)

            # 更新位置
            if f_new < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = f_new
                improved_count += 1

                # 更新全局最优
                if f_new < best_fit:
                    X_best = X_new.copy()
                    best_fit = f_new

        best_fitness_history.append(best_fit)
        print(f"第 {t + 1} 代: 最优适应度 = {best_fit:.6f}, 改进 {improved_count} 个个体")

    return X_best, best_fit, best_fitness_history


# -----------------------------
# 5. 模型评估函数（与WOA一致）
# -----------------------------
def evaluate_final_model(model, X_train_3d, X_test_3d, y_train, y_test, scaler_y):
    """评估最终模型"""
    # 训练集预测
    y_train_pred_scaled = model.predict(X_train_3d, verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()

    # 测试集预测
    y_test_pred_scaled = model.predict(X_test_3d, verbose=0)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = np.mean((y_train_pred - y_train) ** 2)
    train_mae = np.mean(np.abs(y_train_pred - y_train))

    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    test_mae = np.mean(np.abs(y_test_pred - y_test))

    return {
        'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
        'train_r2': train_r2, 'train_mse': train_mse, 'train_mae': train_mae,
        'test_r2': test_r2, 'test_mse': test_mse, 'test_mae': test_mae
    }


# -----------------------------
# 6. 可视化函数（与WOA一致）
# -----------------------------
def plot_results_clean(history_opt, bwo_history, y_train, y_test_true, results, best_params):
    """清晰的结果可视化，与WOA保持一致"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 减小图形尺寸

    # 1. BWO优化过程
    ax1 = axes[0, 0]
    ax1.plot(range(len(bwo_history)), bwo_history, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('BWO Optimization Process', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness Value')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.7, 0.8, f'Best: {min(bwo_history):.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 2. 训练过程
    ax2 = axes[0, 1]
    ax2.plot(history_opt.history['loss'], label='Training Loss', linewidth=2, color='blue')
    if 'val_loss' in history_opt.history:
        ax2.plot(history_opt.history['val_loss'], label='Validation Loss', linewidth=2, color='red', linestyle='--')
    ax2.set_title('Model Training Process', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 预测结果对比
    ax3 = axes[1, 0]
    train_indices = range(1, len(y_train) + 1)
    test_indices = range(len(y_train) + 1, len(y_train) + len(y_test_true) + 1)

    ax3.plot(train_indices, y_train, 'bo-', label='Train True', markersize=8, linewidth=2)
    ax3.plot(train_indices, results['y_train_pred'], 'b^--', label='Train Pred',
             alpha=0.8, markersize=8, linewidth=2)
    ax3.plot(test_indices, y_test_true, 'ro-', label='Test True', markersize=10, linewidth=3)
    ax3.plot(test_indices, results['y_test_pred'], 'rs--', label='Test Pred',
             alpha=0.8, markersize=10, linewidth=3)

    ax3.set_title(f'Prediction Results (Train R2={results["train_r2"]:.3f}, Test R2={results["test_r2"]:.3f})',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Mileage (10^4 km)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 散点图
    ax4 = axes[1, 1]
    ax4.scatter(y_train, results['y_train_pred'], color='blue', alpha=0.8, s=100,
                label=f'Train (R2={results["train_r2"]:.3f})')
    ax4.scatter(y_test_true, results['y_test_pred'], color='red', alpha=0.8, s=150,
                label=f'Test (R2={results["test_r2"]:.3f})')

    # 理想线
    all_values = np.concatenate([y_train, y_test_true])
    min_val, max_val = all_values.min() * 0.98, all_values.max() * 1.02
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal Prediction')

    ax4.set_title('True vs Predicted', fontsize=14, fontweight='bold')
    ax4.set_xlabel('True Value')
    ax4.set_ylabel('Predicted Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 尝试保存图片，如果失败则显示
    try:
        # 确保目录存在
        import os
        os.makedirs('../img', exist_ok=True)
        # 降低dpi以减少内存使用
        plt.savefig('../img/bwo_cnn_optimized_results.png', dpi=150, bbox_inches='tight')
        print("图表已保存到: ../img/bwo_cnn_optimized_results.png")
    except Exception as e:
        print(f"保存图片失败: {e}")
        print("将直接显示图表...")

    plt.show()

    # 打印最优参数
    print(f"\n🎉 BWO优化的最优超参数:")
    print(f"   Conv1滤波器: {int(best_params[0])}")
    print(f"   Conv2滤波器: {int(best_params[1])}")
    print(f"   全连接单元: {int(best_params[2])}")
    print(f"   学习率: {best_params[3]:.5f}")
    print(f"   Dropout率: {best_params[4]:.3f}")


# -----------------------------
# 7. 主流程
# -----------------------------
if __name__ == "__main__":
    print("=== 🚀 优化版 BWO-CNN 超参数搜索系统 ===\n")

    # 1. BWO超参数优化
    print("开始BWO超参数优化...")
    best_params, best_val_score, bwo_history = beluga_whale_optimization_unified(n_whales=5, max_iter=6)

    print(f"\n✅ BWO优化完成!")
    f1_opt, f2_opt, fc_opt, lr_opt, dropout_opt = best_params
    print(f"最优超参数组合:")
    print(f"  - Conv1滤波器: {int(f1_opt)}")
    print(f"  - Conv2滤波器: {int(f2_opt)}")
    print(f"  - 全连接单元: {int(fc_opt)}")
    print(f"  - 学习率: {lr_opt:.5f}")
    print(f"  - Dropout率: {dropout_opt:.3f}")
    print(f"  - 最优适应度: {best_val_score:.6f}")

    # 2. 训练最终模型
    print(f"\n🏋️ 开始训练最终模型...")
    model_final = build_cnn_model(best_params)

    # 使用更长的训练时间来训练最终模型
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history_final = model_final.fit(
        X_train_3d, y_train_scaled,
        epochs=100,
        batch_size=2,
        verbose=1,
        callbacks=[early_stop]
    )

    # 3. 评估模型
    results = evaluate_final_model(model_final, X_train_3d, X_test_3d, y_train, y_test_true, scaler_y)

    # 4. 打印详细结果
    print(f"\n📊 === 最终模型性能评估 ===")
    print(f"训练集:")
    print(f"  R² = {results['train_r2']:.4f}")
    print(f"  MSE = {results['train_mse']:.6f}")
    print(f"  MAE = {results['train_mae']:.6f}")

    print(f"\n测试集:")
    print(f"  R² = {results['test_r2']:.4f}")
    print(f"  MSE = {results['test_mse']:.6f}")
    print(f"  MAE = {results['test_mae']:.6f}")

    print(f"\n🎯 详细预测结果:")
    for i, (true_val, pred_val) in enumerate(zip(y_test_true, results['y_test_pred'])):
        error = abs(pred_val - true_val)
        error_pct = (error / true_val) * 100
        print(f"  样本{i + 15}: 真实={true_val:.4f}, 预测={pred_val:.4f}, "
              f"误差={error:.4f} ({error_pct:.1f}%)")

    # 5. 性能诊断
    overfitting = results['train_r2'] - results['test_r2']
    print(f"\n🔍 模型诊断:")
    print(f"  过拟合指标: {overfitting:.4f}")
    print(f"  模型复杂度: {model_final.count_params()} 参数")

    if overfitting > 0.2:
        print("  ⚠️  轻微过拟合")
    elif results['test_r2'] > 0.6:
        print("  ✅ 性能优秀!")
    elif results['test_r2'] > 0.3:
        print("  🔶 性能良好")
    else:
        print("  💡 建议进一步优化")

    # 6. 可视化
    plot_results_clean(history_final, bwo_history, y_train, y_test_true, results, best_params)

    print(f"\n🎉 优化完成！")