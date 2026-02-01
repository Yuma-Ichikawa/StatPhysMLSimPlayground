# レプリカ法モジュール

本モジュールは, 学習の統計力学における**レプリカ・トリック**から導出される**鞍点方程式**のソルバーを提供する. サンプル数 $n$ と次元 $d$ が共に無限大に発散し, その比 $\alpha = n/d$ が一定に保たれる比例レジームにおける高次元学習問題の理論解析を可能にする.

## 目次

1. [概要](#概要)
2. [数学的背景](#数学的背景)
3. [オーダーパラメータ](#オーダーパラメータ)
4. [利用可能な鞍点方程式](#利用可能な鞍点方程式)
5. [SaddlePointSolver](#saddlepointsolver)
6. [積分ユーティリティ](#積分ユーティリティ)
7. [使用例](#使用例)
8. [参考文献](#参考文献)

---

## 概要

**レプリカ法**は, 乱れた系の典型的な振る舞いを解析するために統計物理学で用いられる強力な手法である. 機械学習においては, 以下の計算を可能にする:

- サンプル複雑度 $\alpha$ の関数としての**汎化誤差**
- 学習における**相転移**（例: 補間閾値）
- **最適正則化**パラメータ
- 学習された解を特徴づける**オーダーパラメータ**

### 主要コンポーネント

| コンポーネント | 説明 |
|---------------|------|
| `SaddlePointSolver` | ダンピング付き固定点反復ソルバー |
| `ReplicaEquations` | 鞍点方程式の抽象基底クラス |
| `*Equations` クラス群 | 一般的な問題に対する事前定義された方程式 |
| `integration.py` | ガウス積分ユーティリティ |

---

## 数学的背景

### Teacher-Student フレームワーク

**Teacher-Student** 設定を考える:

1. **Teacher**: 既知のルールでラベルを生成

$$
y = f_{\text{teacher}}(\mathbf{x}; \mathbf{W}_0) + \varepsilon
$$

ここで $\mathbf{W}_0 \in \mathbb{R}^d$ は teacher の重み, $\varepsilon$ はノイズである.

2. **Student**: $n$ 個のサンプル $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ から以下を最小化して学習

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell\bigl(y_i, f_{\text{student}}(\mathbf{x}_i; \mathbf{w})\bigr) + \lambda R(\mathbf{w})
$$

### スケーリング規約

全てのモデルは **$1/\sqrt{d}$ スケーリング**に従う:

$$
z = \frac{\mathbf{w}^\top \mathbf{x}}{\sqrt{d}} = O(1)
$$

これにより, $d \to \infty$ においても前活性が $O(1)$ に保たれる.

### レプリカ計算

レプリカ法は以下の手順で進行する:

1. **平均自由エネルギー**: 

$$
f = -\lim_{n \to 0} \frac{1}{n} \frac{\partial}{\partial n} \log \mathbb{E}[Z^n]
$$

2. **レプリカ対称仮定**: レプリカ対称性を仮定
3. **鞍点方程式**: オーダーパラメータに関して極値化
4. **固定点反復**: 自己無撞着方程式を解く

---

## オーダーパラメータ

主要なオーダーパラメータは学習された解を特徴づける:

| 記号 | 定義 | 解釈 |
|------|------|------|
| $m$ | $\mathbf{w}^\top \mathbf{W}_0 / d$ | **Teacher-student オーバーラップ**（整合度） |
| $q$ | $\|\mathbf{w}\|^2 / d$ | **自己オーバーラップ**（重みノルムの二乗） |
| $\rho$ | $\|\mathbf{W}_0\|^2 / d$ | **Teacher ノルム**（信号強度） |
| $\eta$ | $\text{Var}(\varepsilon)$ | **ノイズ分散** |

### 汎化誤差

**回帰**（MSE 損失）の場合:

$$
E_g = \frac{1}{2}(\rho - 2m + q) = \frac{1}{2} \mathbb{E}\left[\left(\frac{(\mathbf{w} - \mathbf{W}_0)^\top \mathbf{x}}{\sqrt{d}}\right)^2\right]
$$

**分類**の場合:

$$
P(\text{error}) = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q\rho}}\right)
$$

---

## 利用可能な鞍点方程式

### 1. `RidgeRegressionEquations`

**問題**: 線形 teacher を持つ Ridge 回帰

```python
from statphys.theory.replica import RidgeRegressionEquations

equations = RidgeRegressionEquations(
    rho=1.0,        # Teacher ノルム ||W₀||²/d
    eta=0.1,        # ノイズ分散
    reg_param=0.01  # Ridge パラメータ λ
)
```

**鞍点方程式**:

残差分散:

$$
V = \rho - 2m + q + \eta
$$

共役変数:

$$
\hat{m} = \frac{\alpha \cdot m}{1 + \alpha q / (\lambda + \epsilon)}, \quad
\hat{q} = \frac{\alpha (V + m^2)}{(1 + \alpha q / (\lambda + \epsilon))^2}
$$

更新方程式:

$$
m_{\text{new}} = \frac{\rho \cdot \hat{m}}{\lambda + \hat{q}}, \quad
q_{\text{new}} = \frac{\rho \cdot \hat{m}^2 + \hat{q}(\rho + \eta)}{(\lambda + \hat{q})^2}
$$

**汎化誤差**: $E_g = \frac{1}{2}(\rho - 2m + q)$

---

### 2. `LassoEquations`

**問題**: L1 正則化を持つ LASSO 回帰

```python
from statphys.theory.replica import LassoEquations

equations = LassoEquations(
    rho=1.0,
    eta=0.1,
    reg_param=0.1  # L1 ペナルティ強度
)
```

**主要な特徴**: **ソフト閾値処理** proximal 作用素を使用:

$$
\text{prox}_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)
$$

**更新方程式**は有効場分布上のガウス積分を含む:

$$
m_{\text{new}} = \int \mathcal{D}z \, \text{prox}_{\lambda/\sqrt{\hat{q}}}(\omega + \sqrt{\hat{q}} z) \cdot \frac{\sqrt{\rho} \, m}{\sqrt{q}}
$$

$$
q_{\text{new}} = \int \mathcal{D}z \, \left[\text{prox}_{\lambda/\sqrt{\hat{q}}}(\omega + \sqrt{\hat{q}} z)\right]^2
$$

ここで $\mathcal{D}z = \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz$ はガウス測度である.

---

### 3. `LogisticRegressionEquations`

**問題**: ロジスティック損失を持つ二値分類

```python
from statphys.theory.replica import LogisticRegressionEquations

equations = LogisticRegressionEquations(
    rho=1.0,
    reg_param=0.01
)
```

**Teacher**: 線形分離器から二値ラベル $y \in \{-1, +1\}$ を生成.

**ロジスティック損失**: $\ell(y, z) = \log(1 + e^{-yz})$

**汎化誤差**（分類誤り率）:

$$
P(\text{error}) = \frac{1}{\pi} \arccos\left(\frac{m}{\sqrt{q\rho}}\right)
$$

---

### 4. `PerceptronEquations`

**問題**: ヒンジ損失を持つパーセプトロン/SVM 学習

```python
from statphys.theory.replica import PerceptronEquations

equations = PerceptronEquations(
    rho=1.0,
    margin=0.0,     # マージン κ（Gardner 体積用）
    reg_param=0.0
)
```

**ヒンジ損失**: $\ell(y, z) = \max(0, \kappa - yz)$

**応用**:
- パーセプトロン学習則（$\kappa = 0$）
- サポートベクターマシン（$\kappa > 0$）
- Gardner 記憶容量解析

**Gardner 体積**条件:

$$
y_i \cdot \frac{\mathbf{w}^\top \mathbf{x}_i}{\sqrt{d} \|\mathbf{w}\|} \geq \kappa, \quad \forall i
$$

---

### 5. `ProbitEquations`

**問題**: プロビット回帰（ガウス CDF teacher）

```python
from statphys.theory.replica import ProbitEquations

equations = ProbitEquations(
    rho=1.0,
    reg_param=0.01
)
```

**Teacher**: 

$$
P(y=1|\mathbf{x}) = \Phi\left(\frac{\mathbf{W}_0^\top \mathbf{x}}{\sqrt{d}}\right)
$$

ここで $\Phi(\cdot)$ はガウス CDF である.

**利点**: プロビットモデルのガウス構造により, ガウス積分が閉形式で解ける.

---

### 6. `CommitteeMachineEquations`

**問題**: 2 層ニューラルネットワーク（委員会マシン）

```python
from statphys.theory.replica import CommitteeMachineEquations

equations = CommitteeMachineEquations(
    K=2,            # Student 隠れユニット数
    M=2,            # Teacher 隠れユニット数
    rho=1.0,
    eta=0.0,
    activation='erf',  # 'erf', 'tanh', 'sign', 'relu'
    reg_param=0.01
)
```

**アーキテクチャ**:

$$
\text{Student}: \quad f(\mathbf{x}) = \frac{1}{\sqrt{K}} \sum_{k=1}^K \phi\left(\frac{\mathbf{v}_k^\top \mathbf{x}}{\sqrt{d}}\right)
$$

$$
\text{Teacher}: \quad y = \frac{1}{\sqrt{M}} \sum_{m=1}^M \phi\left(\frac{\mathbf{v}_m^{*\top} \mathbf{x}}{\sqrt{d}}\right)
$$

**オーダーパラメータ**（対称仮定）:
- $Q_{kk'} = \frac{1}{d} \mathbf{v}_k^\top \mathbf{v}_{k'}$ : Student-Student オーバーラップ
- $R_{km} = \frac{1}{d} \mathbf{v}_k^\top \mathbf{v}_m^*$ : Student-Teacher オーバーラップ
- $T_{mm'} = \frac{1}{d} \mathbf{v}_m^{*\top} \mathbf{v}_{m'}^*$ : Teacher-Teacher オーバーラップ

---

## SaddlePointSolver

`SaddlePointSolver` クラスは, いくつかの高度な機能を備えた**固定点反復**を実装している:

### 基本的な使用法

```python
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# 方程式を作成
equations = RidgeRegressionEquations(rho=1.0, eta=0.1, reg_param=0.01)

# ソルバーを作成
solver = SaddlePointSolver(
    equations=equations,
    order_params=['m', 'q'],
    damping=0.5,
    tol=1e-8,
    max_iter=10000,
    verbose=True
)

# α の範囲で解く
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
result = solver.solve(alpha_values, rho=1.0, eta=0.1, reg_param=0.01)

# 結果にアクセス
print(result.order_params['m'])  # Teacher-student オーバーラップ
print(result.order_params['q'])  # 自己オーバーラップ
```

### ソルバーパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `damping` | 0.5 | 更新のダンピング係数: $x_{\text{new}} = \gamma f(x) + (1-\gamma) x$ |
| `adaptive_damping` | True | 振動時にダンピングを自動的に減少 |
| `damping_decay` | 0.9 | ダンピング減少係数 |
| `min_damping` | 0.01 | 最小許容ダンピング |
| `tol` | $10^{-8}$ | 収束許容誤差 |
| `max_iter` | 10000 | 最大反復回数 |
| `n_restarts` | 3 | ロバスト性のためのランダム再起動回数 |
| `use_continuation` | True | 前の解を初期値として使用 |

### 汎化誤差付きで解く

```python
result = solver.solve_with_generalization_error(
    alpha_values,
    eg_formula=equations.generalization_error,
    rho=1.0,
    eta=0.1,
    reg_param=0.01
)

# 汎化誤差にアクセス
print(result.order_params['eg'])
```

### カスタム方程式

独自の鞍点方程式を定義可能:

```python
def my_equations(m, q, alpha, **params):
    rho = params.get('rho', 1.0)
    lam = params.get('reg_param', 0.01)
    
    # 独自の更新方程式をここに記述
    new_m = ...
    new_q = ...
    
    return new_m, new_q

solver = SaddlePointSolver(
    equations=my_equations,
    order_params=['m', 'q']
)
```

---

## 積分ユーティリティ

`integration.py` モジュールはガウス積分のための数値積分を提供する.

### 1 変数ガウス積分

$\mathbb{E}_{z \sim \mathcal{N}(\mu, \sigma^2)}[f(z)]$ を計算:

```python
from statphys.theory.replica import gaussian_integral

# z ~ N(0, 1) に対する E[z²]
result = gaussian_integral(lambda z: z**2, mean=0.0, var=1.0)
# result ≈ 1.0

# 手法: 'quadrature', 'hermite', 'monte_carlo'
result = gaussian_integral(func, mean=0, var=1, method='hermite', n_points=100)
```

### 2 変数ガウス積分

$\mathbb{E}_{(z_1, z_2) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})}[f(z_1, z_2)]$ を計算:

```python
from statphys.theory.replica import double_gaussian_integral

# 相関ガウス分布に対する E[z₁·z₂]
result = double_gaussian_integral(
    lambda z1, z2: z1 * z2,
    mean1=0.0, mean2=0.0,
    var1=1.0, var2=1.0,
    cov=0.5  # 共分散
)
# result ≈ 0.5
```

### Proximal 作用素

関数 $f$ の **proximal 作用素**は以下で定義される:

$$
\text{prox}_{\gamma f}(x) = \arg\min_y \left[ f(y) + \frac{1}{2\gamma} \|x - y\|^2 \right]
$$

**Moreau エンベロープ**は:

$$
M_{\gamma f}(x) = \min_y \left[ f(y) + \frac{1}{2\gamma} \|x - y\|^2 \right]
$$

```python
from statphys.theory.replica import proximal_operator, moreau_envelope

# ソフト閾値処理（L1 の proximal）
from statphys.theory.replica.integration import soft_threshold
x_thresholded = soft_threshold(x=2.0, threshold=0.5)  # = 1.5

# 一般的な proximal 作用素
prox_value = proximal_operator(func=lambda y: abs(y), x=2.0, gamma=1.0)

# Moreau エンベロープ
envelope = moreau_envelope(func=lambda y: abs(y), x=2.0, gamma=1.0)
```

---

## 使用例

### 例 1: Ridge 回帰の相図

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# セットアップ
rho, eta = 1.0, 0.1
lambda_values = [0.001, 0.01, 0.1, 1.0]
alpha_values = np.linspace(0.1, 5.0, 50)

fig, ax = plt.subplots()

for lam in lambda_values:
    equations = RidgeRegressionEquations(rho=rho, eta=eta, reg_param=lam)
    solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])
    
    result = solver.solve(alpha_values, rho=rho, eta=eta, reg_param=lam)
    
    # 汎化誤差を計算
    m_vals = np.array(result.order_params['m'])
    q_vals = np.array(result.order_params['q'])
    eg = 0.5 * (rho - 2*m_vals + q_vals)
    
    ax.plot(alpha_values, eg, label=f'λ={lam}')

ax.set_xlabel(r'$\alpha = n/d$')
ax.set_ylabel(r'$E_g$ (汎化誤差)')
ax.set_title('Ridge 回帰: 理論 vs サンプル複雑度')
ax.legend()
ax.set_yscale('log')
plt.show()
```

### 例 2: 補間閾値

**ダブルディセント**現象は $\alpha = 1$ で発生する:

```python
from statphys.theory.replica import SaddlePointSolver, RidgeRegressionEquations

# Ridgeless 回帰（λ → 0）
equations = RidgeRegressionEquations(rho=1.0, eta=0.1, reg_param=1e-6)
solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])

alpha_values = np.linspace(0.5, 2.0, 100)
result = solver.solve(alpha_values, rho=1.0, eta=0.1, reg_param=1e-6)

# 汎化誤差は α = 1（補間閾値）で発散
m_vals = np.array(result.order_params['m'])
q_vals = np.array(result.order_params['q'])
eg = 0.5 * (1.0 - 2*m_vals + q_vals)

# プロットは α = 1 でピークを示す
```

$\alpha < 1$（アンダーパラメータ化）: 一意の最小ノルム補間器  
$\alpha > 1$（オーバーパラメータ化）: パラメータ増加で汎化が改善

### 例 3: 分類誤差

```python
from statphys.theory.replica import SaddlePointSolver, PerceptronEquations

equations = PerceptronEquations(rho=1.0, margin=0.0)
solver = SaddlePointSolver(equations=equations, order_params=['m', 'q'])

alpha_values = np.linspace(0.5, 10.0, 50)
result = solver.solve(alpha_values, rho=1.0)

# 分類誤差
m_vals = np.array(result.order_params['m'])
q_vals = np.array(result.order_params['q'])
rho = 1.0

error_rate = np.arccos(np.clip(m_vals / np.sqrt(q_vals * rho), -1, 1)) / np.pi
```

分類誤差は以下のように減少する:

$$
P(\text{error}) \sim \frac{1}{\sqrt{\alpha}} \quad \text{as } \alpha \to \infty
$$

---

## 参考文献

### 基礎論文

1. **学習のレプリカ法**
   - Seung, Sompolinsky, Tishby (1992). "Statistical mechanics of learning from examples." *Phys. Rev. A*

2. **Ridge 回帰**
   - Advani, Saxe (2017). "High-dimensional dynamics of generalization error in neural networks." *arXiv:1710.03667*
   - Hastie, Montanari, Rosset, Tibshirani (2022). "Surprises in high-dimensional ridgeless least squares interpolation." *Ann. Statist.*

3. **LASSO**
   - Bayati, Montanari (2011). "The LASSO risk for Gaussian matrices." *IEEE Trans. Inf. Theory*
   - Thrampoulidis, Oymak, Hassibi (2018). "Precise error analysis of regularized M-estimators." *IEEE Trans. Inf. Theory*

4. **分類**
   - Gardner (1988). "The space of interactions in neural network models." *J. Phys. A*
   - Dietrich, Opper, Sompolinsky (1999). "Statistical mechanics of support vector networks." *Phys. Rev. Lett.*

5. **委員会マシン**
   - Saad, Solla (1995). "Exact solution for on-line learning in multilayer neural networks." *Phys. Rev. Lett.*
   - Goldt, Mézard, Krzakala, Zdeborová (2020). "Modeling the influence of data structure on learning in neural networks." *Phys. Rev. X*

### 教科書

- Engel, Van den Broeck (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
- Mézard, Montanari (2009). *Information, Physics, and Computation*. Oxford University Press.

---

## デフォルトでサポートされているモデル

本モジュールは以下のモデルに対する鞍点方程式をデフォルトでサポートしている.

### モデル一覧

| モデル名 | クラス | 説明 | ファイル |
|---------|-------|------|---------|
| **Ridge 回帰** | `RidgeRegressionEquations` | L2 正則化付き線形回帰 | `models/linear.py` |
| **LASSO** | `LassoEquations` | L1 正則化付き線形回帰 | `models/lasso.py` |
| **ロジスティック回帰** | `LogisticRegressionEquations` | 二値分類（ロジスティック損失） | `models/logistic.py` |
| **パーセプトロン/SVM** | `PerceptronEquations` | パーセプトロン・SVM（ヒンジ損失） | `models/perceptron.py` |
| **プロビット回帰** | `ProbitEquations` | ガウス CDF Teacher | `models/probit.py` |
| **委員会マシン** | `CommitteeMachineEquations` | 2 層ニューラルネットワーク | `models/committee.py` |

### ファイル構造

```
theory/replica/
├── __init__.py          # モジュールエントリポイント
├── solver.py            # SaddlePointSolver
├── equations.py         # レガシー方程式（後方互換性）
├── integration.py       # ガウス積分ユーティリティ
└── models/              # モデル別鞍点方程式
    ├── __init__.py      # モデルレジストリ
    ├── base.py          # ReplicaEquations 基底クラス
    ├── linear.py        # RidgeRegressionEquations
    ├── lasso.py         # LassoEquations
    ├── logistic.py      # LogisticRegressionEquations
    ├── perceptron.py    # PerceptronEquations
    ├── probit.py        # ProbitEquations
    └── committee.py     # CommitteeMachineEquations
```

### モデルの簡単な取得方法

```python
from statphys.theory.replica import get_replica_equations, REPLICA_MODELS

# 利用可能なモデルを確認
print(REPLICA_MODELS.keys())
# dict_keys(['ridge', 'lasso', 'logistic', 'perceptron', 'probit', 'committee'])

# 名前でモデルを取得
equations = get_replica_equations("ridge", rho=1.0, eta=0.1, reg_param=0.01)
```

---

## デフォルトでサポートされているモデル

本モジュールは以下のモデルに対する鞍点方程式をデフォルトでサポートしている.

### モデル一覧

| モデル名 | クラス | 説明 | ファイル |
|---------|-------|------|---------|
| **Ridge 回帰** | `RidgeRegressionEquations` | L2 正則化付き線形回帰 | `models/linear.py` |
| **LASSO** | `LassoEquations` | L1 正則化付き線形回帰 | `models/lasso.py` |
| **ロジスティック回帰** | `LogisticRegressionEquations` | 二値分類（ロジスティック損失） | `models/logistic.py` |
| **パーセプトロン/SVM** | `PerceptronEquations` | パーセプトロン・SVM（ヒンジ損失） | `models/perceptron.py` |
| **プロビット回帰** | `ProbitEquations` | ガウス CDF Teacher | `models/probit.py` |
| **委員会マシン** | `CommitteeMachineEquations` | 2 層ニューラルネットワーク | `models/committee.py` |

### ファイル構造

```
theory/replica/
├── __init__.py          # モジュールエントリポイント
├── solver.py            # SaddlePointSolver
├── equations.py         # レガシー方程式（後方互換性）
├── integration.py       # ガウス積分ユーティリティ
├── README.md            # 英語ドキュメント
├── README_ja.md         # 日本語ドキュメント
└── models/              # モデル別鞍点方程式
    ├── __init__.py      # モデルレジストリ
    ├── base.py          # ReplicaEquations 基底クラス
    ├── linear.py        # RidgeRegressionEquations
    ├── lasso.py         # LassoEquations
    ├── logistic.py      # LogisticRegressionEquations
    ├── perceptron.py    # PerceptronEquations
    ├── probit.py        # ProbitEquations
    └── committee.py     # CommitteeMachineEquations
```

### モデルの簡単な取得方法

```python
from statphys.theory.replica import get_replica_equations, REPLICA_MODELS

# 利用可能なモデルを確認
print(REPLICA_MODELS.keys())
# dict_keys(['ridge', 'lasso', 'logistic', 'perceptron', 'probit', 'committee'])

# 名前でモデルを取得
equations = get_replica_equations("ridge", rho=1.0, eta=0.1, reg_param=0.01)
```

### 鞍点方程式の形式

鞍点方程式は2つの形式で記述できる:

1. **固定点形式**（現在のソルバーが使用）:
   ```
   (m_new, q_new, ...) = F(m, q, ...; α, params)
   ```
   ソルバーは以下を反復: x_{n+1} = (1-γ)x_n + γF(x_n)

2. **残差形式**（0 = 右辺）:
   ```
   0 = G(m, q, ...; α, params) = F(m, q, ...) - (m, q, ...)
   ```

`ReplicaEquations` 基底クラスは `residual()` メソッドを提供し, どちらの形式でも使用可能.

### カスタムモデルの追加方法

新しいモデルを追加するには:

1. `models/` フォルダに新しいファイルを作成（例: `models/my_model.py`）
2. `ReplicaEquations` を継承したクラスを定義
3. `__call__` メソッドで固定点更新則を実装
4. `generalization_error` メソッドで汎化誤差を実装
5. `models/__init__.py` の `REPLICA_MODELS` に追加

```python
# models/my_model.py
from statphys.theory.replica.models.base import ReplicaEquations

class MyCustomEquations(ReplicaEquations):
    def __init__(self, rho=1.0, eta=0.0, reg_param=0.01, **params):
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
    
    def __call__(self, m, q, alpha, **kwargs):
        rho = kwargs.get('rho', self.rho)
        eta = kwargs.get('eta', self.eta)
        lam = kwargs.get('reg_param', self.reg_param)
        
        # 固定点更新則を定義
        # 鞍点方程式: 0 = m_new - m, 0 = q_new - q
        V = rho - 2*m + q + eta
        hat_m = alpha * m / (1 + alpha*q/(lam + 1e-6))
        hat_q = alpha * (V + m**2) / (1 + alpha*q/(lam + 1e-6))**2
        
        new_m = rho * hat_m / (lam + hat_q + 1e-6)
        new_q = (rho * hat_m**2 + hat_q*(rho + eta)) / (lam + hat_q + 1e-6)**2
        
        return new_m, new_q
    
    def generalization_error(self, m, q, **kwargs):
        rho = kwargs.get('rho', self.rho)
        return 0.5 * (rho - 2*m + q)
```

---

## カスタム鞍点方程式の詳細な書き方

新しいモデルの鞍点方程式を追加する際の詳細なガイドを示す.

### 基本構造

```python
# models/my_custom_model.py
import numpy as np
from scipy.integrate import quad
from statphys.theory.replica.models.base import ReplicaEquations

# 特殊関数はutilsから取得する（自作しない）
from statphys.utils.special_functions import (
    gaussian_pdf,      # ガウスPDF: φ(x)
    gaussian_cdf,      # ガウスCDF: Φ(x)
    gaussian_tail,     # ガウス尾確率: H(x) = 1 - Φ(x)
    sigmoid,           # シグモイド関数
    erf_activation,    # erf活性化
    soft_threshold,    # ソフト閾値（L1のproximal）
    I2, I3, I4,        # 委員会マシン相関関数
    classification_error_linear,  # 分類誤差
    regression_error_linear,      # 回帰誤差
)
from statphys.utils.integration import (
    gaussian_integral_1d,      # 1次元ガウス積分
    gaussian_integral_2d,      # 2次元ガウス積分
    teacher_student_integral,  # Teacher-Student積分
    conditional_expectation,   # 条件付き期待値
)


class MyCustomReplicaEquations(ReplicaEquations):
    """
    カスタムモデルの鞍点方程式.
    
    鞍点方程式は2つの形式で書ける:
    
    1. 固定点形式（ソルバーが使用）:
       (m_new, q_new) = F(m, q; α, params)
       
    2. 残差形式（0 = 右辺）:
       0 = F(m, q; α, params) - (m, q)
    
    ここで:
        - α = n/d : サンプル比
        - m = w^T W_0 / d : Teacher-Studentオーバーラップ
        - q = ||w||^2 / d : 自己オーバーラップ
    """
    
    def __init__(
        self,
        rho: float = 1.0,       # Teacherノルム ||W_0||^2/d
        eta: float = 0.0,       # ノイズ分散 σ^2
        reg_param: float = 0.01, # 正則化パラメータ λ
        custom_param: float = 1.0,  # カスタムパラメータ
        **params,
    ):
        super().__init__(
            rho=rho, 
            eta=eta, 
            reg_param=reg_param,
            custom_param=custom_param,
            **params
        )
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
        self.custom_param = custom_param
    
    def __call__(
        self, 
        m: float, 
        q: float, 
        alpha: float, 
        **kwargs
    ) -> tuple[float, float]:
        """
        固定点更新則を計算.
        
        ソルバーは以下を反復:
            x_{n+1} = (1-γ)x_n + γF(x_n)
        
        収束条件: ||x_{n+1} - x_n|| < tol
        
        Args:
            m: 現在のTeacher-Studentオーバーラップ
            q: 現在の自己オーバーラップ
            alpha: サンプル比 n/d
            **kwargs: パラメータ（__init__を上書き可能）
        
        Returns:
            (m_new, q_new): 更新されたオーダーパラメータ
        """
        # パラメータ取得（引数から上書き可能）
        rho = kwargs.get('rho', self.rho)
        eta = kwargs.get('eta', self.eta)
        lam = kwargs.get('reg_param', self.reg_param)
        
        # 数値安定性のための制約
        q = max(q, 1e-10)
        
        # ========================================
        # ここに鞍点方程式を実装
        # ========================================
        
        # 例: Ridge回帰の鞍点方程式
        # 残差分散
        V = rho - 2*m + q + eta
        
        # 共役変数（レプリカ計算から導出）
        denom = 1 + alpha * q / (lam + 1e-6)
        hat_m = alpha * m / denom
        hat_q = alpha * (V + m**2) / denom**2
        
        # 更新方程式
        new_m = rho * hat_m / (lam + hat_q + 1e-6)
        new_q = (rho * hat_m**2 + hat_q*(rho + eta)) / (lam + hat_q + 1e-6)**2
        
        # ガウス積分を使う場合の例（LASSO等）
        # def integrand_m(z):
        #     effective = sqrt(rho)*m/sqrt(q) + sqrt(hat_q)*z
        #     prox = soft_threshold(effective, lam/sqrt(hat_q))
        #     return prox * sqrt(rho)/sqrt(q) * gaussian_pdf(z)
        # new_m = gaussian_integral_1d(integrand_m, mean=0, variance=1)
        
        return new_m, new_q
    
    def residual(
        self, 
        m: float, 
        q: float, 
        alpha: float, 
        **kwargs
    ) -> tuple[float, float]:
        """
        残差形式（0 = 右辺）を計算.
        
        鞍点条件: 0 = F(m, q) - (m, q)
        
        これは収束チェックやNewton法で使用可能.
        """
        new_m, new_q = self(m, q, alpha, **kwargs)
        return (new_m - m, new_q - q)
    
    def generalization_error(
        self, 
        m: float, 
        q: float, 
        **kwargs
    ) -> float:
        """
        汎化誤差を計算.
        
        回帰: E_g = (1/2)(ρ - 2m + q)
        分類: P(error) = (1/π) arccos(m / √(qρ))
        """
        rho = kwargs.get('rho', self.rho)
        
        # 回帰の場合
        return regression_error_linear(m, q, rho)
        
        # 分類の場合
        # return classification_error_linear(m, q, rho)
    
    def get_order_param_names(self) -> list[str]:
        """オーダーパラメータ名を返す"""
        return ['m', 'q']
    
    def get_default_init(self) -> tuple[float, ...]:
        """デフォルト初期値（収束しやすい値を選ぶ）"""
        return (0.5, 0.5)
    
    def is_physical(self, m: float, q: float, **kwargs) -> bool:
        """物理的制約のチェック"""
        rho = kwargs.get('rho', self.rho)
        # q >= 0 かつ m^2 <= q*rho（Cauchy-Schwarz）
        return q >= 0 and m**2 <= q * rho * (1 + 1e-6)
```

### 使用例

```python
from statphys.theory.replica import SaddlePointSolver
from models.my_custom_model import MyCustomReplicaEquations
import numpy as np

# 方程式を作成
equations = MyCustomReplicaEquations(
    rho=1.0,
    eta=0.1,
    reg_param=0.01,
    custom_param=2.0,
)

# ソルバーを作成
solver = SaddlePointSolver(
    equations=equations,
    order_params=equations.get_order_param_names(),
    damping=0.5,       # ダンピング係数（振動防止）
    tol=1e-8,
    max_iter=10000,
    n_restarts=3,      # ランダム再起動回数
    verbose=True,
)

# αの範囲で解く
alpha_values = np.linspace(0.1, 5.0, 50)
result = solver.solve(
    alpha_values=alpha_values,
    use_continuation=True,  # 前のαの解を次の初期値に使用
    rho=1.0,
    eta=0.1,
    reg_param=0.01,
)

# 汎化誤差付きで解く
result = solver.solve_with_generalization_error(
    alpha_values=alpha_values,
    eg_formula=equations.generalization_error,
    rho=1.0,
    eta=0.1,
    reg_param=0.01,
)

# 結果にアクセス
m_values = result.order_params['m']
q_values = result.order_params['q']
eg_values = result.order_params['eg']
```

### 特殊関数の使用（utils）

**重要**: 特殊関数は自作せず、必ず `statphys.utils` から取得すること.

```python
# 正しい使用法
from statphys.utils.special_functions import gaussian_pdf, sigmoid, H, soft_threshold
from statphys.utils.integration import gaussian_integral_1d, teacher_student_integral

# 間違った使用法（自作しない）
# def my_gaussian_pdf(x):  # ← これは避ける
#     return np.exp(-x**2/2) / np.sqrt(2*np.pi)
```

#### 利用可能な特殊関数（`statphys.utils.special_functions`）

| 関数 | 説明 | 数式 |
|------|------|------|
| `gaussian_pdf(x)` / `phi(x)` | ガウスPDF | φ(x) = (1/√(2π)) exp(-x²/2) |
| `gaussian_cdf(x)` / `Phi(x)` | ガウスCDF | Φ(x) |
| `gaussian_tail(x)` / `H(x)` | 尾確率 | H(x) = 1 - Φ(x) |
| `sigmoid(x)` | シグモイド | σ(x) = 1/(1+exp(-x)) |
| `erf_activation(x)` | erf活性化 | erf(x/√2) |
| `soft_threshold(x, λ)` | ソフト閾値 | sign(x)·max(\|x\|-λ, 0) |
| `I2(Q, activation)` | 2点相関 | 委員会マシン用 |
| `I3(Q_ab, Q_ac, Q_bc)` | 3点相関 | 委員会マシン用 |
| `classification_error_linear(m, q, ρ)` | 分類誤差 | (1/π) arccos(m/√(qρ)) |
| `regression_error_linear(m, q, ρ)` | 回帰誤差 | (1/2)(ρ - 2m + q) |
| `moreau_envelope(f, x, γ)` | Moreauエンベロープ | proximal計算用 |

#### 利用可能な積分ユーティリティ（`statphys.utils.integration`）

| 関数 | 説明 | 使用例 |
|------|------|--------|
| `gaussian_integral_1d(f, μ, σ²)` | 1変数ガウス積分 | E[f(z)], z ~ N(μ, σ²) |
| `gaussian_integral_2d(f, μ, Σ)` | 2変数ガウス積分 | 相関変数の期待値 |
| `gaussian_integral_nd(f, μ, Σ)` | 多変数ガウス積分 | 委員会マシン等 |
| `teacher_student_integral(f, m, q, ρ)` | T-S積分 | E[f(u,z)]の計算 |
| `conditional_expectation(f, u, m, q, ρ)` | 条件付き期待値 | E[f(z)\|u=value] |

### 鞍点方程式導出のヒント

レプリカ法から鞍点方程式を導出する際の一般的な構造:

1. **自由エネルギー**を計算（レプリカ・トリック使用）
2. **レプリカ対称仮定**を適用
3. **鞍点条件**: ∂f/∂m = 0, ∂f/∂q = 0, ...
4. **固定点形式**に変形: (m, q, ...) = F(m, q, ...)

典型的な鞍点方程式の構造:

```
# 共役変数（チャネル側）
hat_m = α × (mに関する項) / (正則化+分散項)
hat_q = α × (qに関する項) / (正則化+分散項)²

# 更新方程式（事前分布側）
m_new = ガウス積分[proximal(有効場) × Teacher成分]
q_new = ガウス積分[proximal(有効場)²]
```
