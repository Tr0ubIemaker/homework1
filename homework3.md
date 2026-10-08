```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# 加载数据（用diabetes数据集替代Boston，sklearn新版本移除Boston）
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = data.feature_names

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Z-score标准化！正则化必须做
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 生成λ网格
alphas = np.logspace(-3, 3, 100)

# ---------------------- Ridge 岭回归 ----------------------
ridge_cv = RidgeCV(alphas=alphas, cv=10, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_coef = ridge_cv.coef_
ridge_nonzero = np.sum(np.abs(ridge_coef) > 1e-8)

# ---------------------- Lasso L1正则 ----------------------
lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=30000)
lasso_cv.fit(X_train_scaled, y_train)
lasso_pred = lasso_cv.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_coef = lasso_cv.coef_
lasso_nonzero = np.sum(np.abs(lasso_coef) > 1e-8)

# ---------------------- ElasticNet 弹性网 ----------------------
enet_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=10, random_state=42, max_iter=30000)
enet_cv.fit(X_train_scaled, y_train)
enet_pred = enet_cv.predict(X_test_scaled)
enet_rmse = np.sqrt(mean_squared_error(y_test, enet_pred))
enet_coef = enet_cv.coef_
enet_nonzero = np.sum(np.abs(enet_coef) > 1e-8)

# =====输出汇总结果=====
print("="*40)
print(f"Ridge回归 | 最优λ={ridge_cv.alpha_:.4f} | 测试RMSE={ridge_rmse:.2f} | 非零变量数={ridge_nonzero}")
print(f"Lasso回归 | 最优λ={lasso_cv.alpha_:.4f} | 测试RMSE={lasso_rmse:.2f} | 非零变量数={lasso_nonzero}")
print(f"ElasticNet| 最优λ={enet_cv.alpha_:.4f} | 测试RMSE={enet_rmse:.2f} | 非零变量数={enet_nonzero}")
print("="*40)

# =====绘制系数随λ变化的路径图=====
def plot_coef_path(estimator, title):
    coef_list = []
    for a in alphas:
        model = estimator(alpha=a, max_iter=30000)
        model.fit(X_train_scaled, y_train)
        coef_list.append(model.coef_)
    coef_array = np.array(coef_list)
    plt.figure(figsize=(8,5))
    for i, fname in enumerate(feature_names):
        plt.plot(alphas, coef_array[:,i], label=fname)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (log scale)")
    plt.ylabel("系数 Coefficient")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="right", bbox_to_anchor=(1.3,0.5))
    plt.show()

plot_coef_path(Ridge, "Ridge系数路径图")
plot_coef_path(Lasso, "Lasso系数路径图")
plot_coef_path(ElasticNet, "ElasticNet系数路径图")

# =====1-SE法则文字分析=====
print("\n【1-SE法则讨论】")
print("1. 1-SE法则：不直接取交叉验证误差最小的λ；选择在最小CV误差±1个标准误区间内，**最大的λ**。")
print("2. λ越大代表正则惩罚越强，模型越简单，变量越少，牺牲很小一点预测精度换取更好泛化能力。")
print("3. Ridge只能压缩系数，不能将系数压缩到0，无法得到稀疏模型，1-SE法则对它没有变量选择意义。")
print("4. Lasso与ElasticNet具备变量选择能力：λ足够大时部分系数被置零。可以使用1-SE法则，挑选更稀疏的模型，减少特征数量，提升模型可解释性。")
```