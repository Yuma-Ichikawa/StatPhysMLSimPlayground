# オンライン学習ダイナミクスモジュール

本モジュールは, 高次元極限におけるオンライン学習ダイナミクスを解析するための **ODE ソルバー**を提供する. データが逐次的に到着し, 各サンプル観測後にモデルが更新される場合, $d \to \infty$ において学習ダイナミクスはオーダーパラメータに関する常微分方程式（ODE）で記述される.

## 目次

1. [概要](#概要)
2. [数学的背景](#数学的背景)
3. [ODESolver](#odesolver)
4. [利用可能な ODE 方程式](#利用可能な-ode-方程式)
5. [使用例](#使用例)
6. [参考文献](#参考文献)

---

## 概要

**オンライン学習**では, データが逐次的に到着し, 各サンプル観測後にモデルが更新される. 高次元極限 $d \to \infty$ において正規化時間 $t = n/d$ を導入すると, オーダーパラメータのダイナミクスは決定論的 ODE で支配される.

### 主要コンポーネント

| コンポーネント | 説明 |
|---------------|------|
| `ODESolver` | `scipy.integrate.solve_ivp` を使用した ODE ソルバー |
| `AdaptiveODESolver` | イベント検出付き ODE ソルバー |
| `OnlineEquations` | ODE 方程式の抽象基底クラス |
| `Online*Equations` | 一般的な問題に対する事前定義された方程式 |

### レプリカ法との比較

| 側面 | レプリカ法 | オンライン学習 |
|------|-----------|---------------|
| **設定** | バッチ学習（全データ一括） | 逐次学習 |
| **解析** | 固定点方程式 | ODE |
| **パラメータ** | $\alpha = n/d$（サンプル比） | $t = n/d$（正規化時間） |
| **結果** | 平衡状態の性質 | 学習ダイナミクス/軌跡 |

---

## 数学的背景

### オンライン SGD の設定

オンライン SGD を用いた **Teacher-Student** フレームワークを考える:

1. **データストリーム**: 各ステップ $\mu$ でサンプル $(\mathbf{x}^\mu, y^\mu)$ を受信
2. **SGD 更新**:

$$
\mathbf{w}^{\mu+1} = \mathbf{w}^\mu - \eta \nabla_\mathbf{w} \ell(y^\mu, f(\mathbf{x}^\mu; \mathbf{w}^\mu))
$$

### 高次元極限

正規化時間 $t = \mu / d$ を用いた $d \to \infty$ 極限において, オーダーパラメータは決定論的 ODE に従って発展する:

$$
\frac{dm}{dt} = F_m(m, q; \eta, \rho, \lambda, \ldots)
$$

$$
\frac{dq}{dt} = F_q(m, q; \eta, \rho, \lambda, \ldots)
$$

### オーダーパラメータ

| 記号 | 定義 | 解釈 |
|------|------|------|
| $m(t)$ | $\mathbf{w}(t)^\top \mathbf{W}_0 / d$ | Teacher-student オーバーラップ |
| $q(t)$ | $\|\mathbf{w}(t)\|^2 / d$ | 自己オーバーラップ（重みノルム） |
| $\eta$ | 学習率 | ステップサイズ |
| $\rho$ | $\|\mathbf{W}_0\|^2 / d$ | Teacher ノルム |

### 汎化誤差

**回帰**の場合:

$$
E_g(t) = \frac{1}{2}(\rho - 2m(t) + q(t))
$$

**分類**の場合:

$$
P(\text{error}, t) = \frac{1}{\pi} \arccos\left(\frac{m(t)}{\sqrt{q(t)\rho}}\right)
$$

---

## ODESolver

`ODESolver` クラスは, オンライン学習 ODE を解くために `scipy.integrate.solve_ivp` をラップしている.

### 基本的な使用法

```python
from statphys.theory.online import ODESolver, OnlineSGDEquations

# 方程式を作成
equations = OnlineSGDEquations(
    rho=1.0,        # Teacher ノルム
    eta_noise=0.1,  # ノイズ分散
    lr=0.5,         # 学習率
    reg_param=0.0   # 正則化
)

# ソルバーを作成
solver = ODESolver(
    equations=equations,
    order_params=['m', 'q'],
    method='RK45',
    tol=1e-8,
    verbose=True
)

# ODE を解く
result = solver.solve(
    t_span=(0, 10),           # 時間範囲
    init_values=(0.0, 0.01),  # 初期値 (m₀, q₀)
    n_points=100              # 出力点数
)

# 結果にアクセス
t_values = result.param_values  # 時間点
m_values = result.order_params['m']  # m(t) 軌跡
q_values = result.order_params['q']  # q(t) 軌跡
```

### ソルバーパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `equations` | - | ODE 右辺関数 |
| `order_params` | - | オーダーパラメータの名前 |
| `method` | `'RK45'` | 積分法（`'RK45'`, `'RK23'`, `'Radau'`, `'BDF'`, `'LSODA'`） |
| `tol` | $10^{-8}$ | 相対許容誤差 |
| `max_step` | 0.1 | 最大ステップサイズ |

### 汎化誤差付きで解く

```python
result = solver.solve_with_generalization_error(
    t_span=(0, 10),
    eg_formula=equations.generalization_error,
    init_values=(0.0, 0.01),
    rho=1.0
)

# E_g 軌跡にアクセス
eg_values = result.order_params['eg']
```

### 複数の学習率の比較

```python
results = solver.solve_multiple_lr(
    t_span=(0, 10),
    learning_rates=[0.1, 0.5, 1.0, 2.0],
    init_values=(0.0, 0.01),
    rho=1.0
)

for lr, result in results.items():
    print(f"lr={lr}: 最終 m = {result.order_params['m'][-1]:.4f}")
```

### イベント検出付き AdaptiveODESolver

```python
from statphys.theory.online.solver import AdaptiveODESolver

# 停止イベントを定義: E_g < 閾値 で停止
def eg_threshold_event(t, y, params):
    m, q = y
    rho = params.get('rho', 1.0)
    eg = 0.5 * (rho - 2*m + q)
    return eg - 0.01  # E_g = 0.01 で停止

solver = AdaptiveODESolver(
    equations=equations,
    order_params=['m', 'q'],
    events=[eg_threshold_event]
)

result = solver.solve(t_span=(0, 100), init_values=(0.0, 0.01), rho=1.0)
# E_g が 0.01 に達すると早期終了
```

---

## 利用可能な ODE 方程式

### 1. `OnlineSGDEquations`

**問題**: MSE 損失を持つ線形回帰のオンライン SGD

```python
from statphys.theory.online import OnlineSGDEquations

equations = OnlineSGDEquations(
    rho=1.0,        # Teacher ノルム
    eta_noise=0.1,  # ノイズ分散 σ²
    lr=0.5,         # 学習率 η
    reg_param=0.01  # L2 正則化 λ
)
```

**ODE 系**:

$$
\frac{dm}{dt} = \eta(\rho - m) - \eta\lambda m
$$

$$
\frac{dq}{dt} = \eta^2 V + 2\eta(m - q) - 2\eta\lambda q
$$

ここで $V = \rho - 2m + q + \sigma^2$ は残差分散（訓練損失）である.

**主要なダイナミクス**:
- $m(t) \to m^*$: 最適オーバーラップに収束
- $q(t) \to q^*$: 最適ノルムに収束
- 収束率は $\eta$ と $\lambda$ に依存

---

### 2. `OnlinePerceptronEquations`

**問題**: 二値分類のオンラインパーセプトロン学習

```python
from statphys.theory.online import OnlinePerceptronEquations

equations = OnlinePerceptronEquations(
    rho=1.0,   # Teacher ノルム
    lr=1.0     # 学習率
)
```

**ODE 系**（Saad & Solla 形式）:

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \frac{\phi(\kappa)}{\sqrt{q}}
$$

$$
\frac{dq}{dt} = 2\eta^2 \epsilon(\kappa)
$$

ここで:
- $\kappa = m / \sqrt{q\rho}$ は安定性パラメータ
- $\phi(\kappa) = \frac{1}{\sqrt{2\pi}} e^{-\kappa^2/2}$ はガウス PDF
- $\epsilon(\kappa) = H(\kappa)$ は誤り率（相補ガウス CDF）

**分類誤差**:

$$
P(\text{error}) = \frac{1}{\pi} \arccos(\kappa)
$$

---

### 3. `OnlineRidgeEquations`

**問題**: オンライン Ridge 回帰（`OnlineSGDEquations` のエイリアス）

```python
from statphys.theory.online import OnlineRidgeEquations

equations = OnlineRidgeEquations(
    rho=1.0,
    eta_noise=0.1,
    lr=0.5,
    reg_param=0.1  # Ridge パラメータ λ
)
```

明示的な Ridge 正則化を持つ `OnlineSGDEquations` と同じ ODE 系.

---

### 4. `OnlineLogisticEquations`

**問題**: 二値分類のオンラインロジスティック回帰

```python
from statphys.theory.online import OnlineLogisticEquations

equations = OnlineLogisticEquations(
    rho=1.0,
    lr=0.1,
    reg_param=0.01
)
```

**ロジスティック損失**: $\ell(y, z) = \log(1 + e^{-yz})$

**ODE 系**:

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \mathbb{E}[g(y,z) \cdot u / \sqrt{\rho}] - \eta\lambda m
$$

$$
\frac{dq}{dt} = 2\eta \sqrt{q} \cdot \mathbb{E}[g(y,z) \cdot z / \sqrt{q}] + \eta^2 \mathbb{E}[g(y,z)^2] - 2\eta\lambda q
$$

ここで $g(y, z) = y \cdot \sigma(-yz)$ はロジスティック勾配であり, 期待値は teacher 場 $u$ と student 場 $z$ の同時分布上で取られる.

---

### 5. `OnlineHingeEquations`

**問題**: ヒンジ損失を持つオンライン SVM

```python
from statphys.theory.online import OnlineHingeEquations

equations = OnlineHingeEquations(
    rho=1.0,
    lr=0.1,
    margin=1.0,     # ヒンジマージン κ
    reg_param=0.01
)
```

**ヒンジ損失**: $\ell(y, z) = \max(0, \kappa - yz)$

**ODE 系**:

$$
\frac{dm}{dt} = \eta \sqrt{\rho} \cdot \frac{\phi(\theta)}{\Delta} - \eta\lambda m
$$

$$
\frac{dq}{dt} = 2\eta^2 H(-\theta) - 2\eta\lambda q
$$

ここで $\Delta$ と $\theta$ はマージン条件から導出される.

---

### 6. `OnlineCommitteeEquations`

**問題**: 委員会マシン（2 層ネットワーク）のオンライン学習

```python
from statphys.theory.online import OnlineCommitteeEquations

equations = OnlineCommitteeEquations(
    k_student=2,      # Student 隠れユニット数
    k_teacher=2,      # Teacher 隠れユニット数
    rho=1.0,
    lr=0.1,
    activation='erf'  # 'erf', 'relu'
)
```

**注意**: このクラスはテンプレートを提供する. 完全な実装には, Saad & Solla (1995) に従ってオーバーラップ行列 $Q_{ij}$（student-student）と $R_{in}$（student-teacher）の処理が必要である.

**オーダーパラメータ**:
- $Q_{ij} = \frac{1}{d} \mathbf{w}_i^\top \mathbf{w}_j$ : Student-student オーバーラップ
- $R_{in} = \frac{1}{d} \mathbf{w}_i^\top \mathbf{W}_n^*$ : Student-teacher オーバーラップ

---

## 使用例

### 例 1: 学習率の比較

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.online import ODESolver, OnlineSGDEquations

# セットアップ
rho = 1.0
eta_noise = 0.1
t_span = (0, 20)
learning_rates = [0.1, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for lr in learning_rates:
    equations = OnlineSGDEquations(rho=rho, eta_noise=eta_noise, lr=lr)
    solver = ODESolver(equations=equations, order_params=['m', 'q'])
    
    result = solver.solve(t_span=t_span, init_values=(0.0, 0.01), n_points=200)
    
    t = result.param_values
    m = result.order_params['m']
    q = result.order_params['q']
    eg = 0.5 * (rho - 2*np.array(m) + np.array(q))
    
    axes[0].plot(t, m, label=f'η={lr}')
    axes[1].plot(t, eg, label=f'η={lr}')

axes[0].set_xlabel('t = n/d')
axes[0].set_ylabel('m(t)')
axes[0].set_title('Teacher-Student オーバーラップ')
axes[0].legend()

axes[1].set_xlabel('t = n/d')
axes[1].set_ylabel('$E_g(t)$')
axes[1].set_title('汎化誤差')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 例 2: 最適学習率

正則化なしのオンライン SGD における最適学習率は:

$$
\eta^* = \frac{1}{1 + \sigma^2/\rho}
$$

```python
import numpy as np
from statphys.theory.online import ODESolver, OnlineSGDEquations

rho = 1.0
sigma_sq = 0.1
eta_optimal = 1 / (1 + sigma_sq / rho)
print(f"最適学習率: η* = {eta_optimal:.4f}")

# 最適 vs 準最適の比較
for lr in [0.5 * eta_optimal, eta_optimal, 2.0 * eta_optimal]:
    equations = OnlineSGDEquations(rho=rho, eta_noise=sigma_sq, lr=lr)
    solver = ODESolver(equations=equations, order_params=['m', 'q'])
    result = solver.solve(t_span=(0, 50), init_values=(0.0, 0.01))
    
    final_m = result.order_params['m'][-1]
    final_q = result.order_params['q'][-1]
    final_eg = 0.5 * (rho - 2*final_m + final_q)
    
    print(f"η={lr:.4f}: 最終 E_g = {final_eg:.6f}")
```

### 例 3: パーセプトロン学習曲線

```python
import numpy as np
import matplotlib.pyplot as plt
from statphys.theory.online import ODESolver, OnlinePerceptronEquations

equations = OnlinePerceptronEquations(rho=1.0, lr=1.0)
solver = ODESolver(equations=equations, order_params=['m', 'q'])

result = solver.solve(t_span=(0, 50), init_values=(0.01, 0.01), n_points=500)

t = np.array(result.param_values)
m = np.array(result.order_params['m'])
q = np.array(result.order_params['q'])

# 分類誤差
kappa = m / np.sqrt(q * 1.0)
error_rate = np.arccos(np.clip(kappa, -1, 1)) / np.pi

plt.figure(figsize=(8, 5))
plt.plot(t, error_rate)
plt.xlabel('t = n/d')
plt.ylabel('分類誤差')
plt.title('オンラインパーセプトロン学習')
plt.yscale('log')
plt.grid(True)
plt.show()
```

### 例 4: カスタム ODE 方程式

```python
import numpy as np
from statphys.theory.online import ODESolver

def custom_equations(t, y, params):
    """モメンタム付きオンライン学習のカスタム ODE."""
    m, q, v_m, v_q = y  # モメンタム変数を含む
    
    rho = params.get('rho', 1.0)
    lr = params.get('lr', 0.1)
    beta = params.get('momentum', 0.9)
    
    # 勾配
    grad_m = rho - m
    grad_q = m - q
    
    # モメンタム更新
    dv_m = beta * v_m + lr * grad_m
    dv_q = beta * v_q + lr * grad_q
    
    # パラメータ更新
    dm = dv_m
    dq = 2 * dv_q
    
    return np.array([dm, dq, dv_m - v_m, dv_q - v_q])

solver = ODESolver(
    equations=custom_equations,
    order_params=['m', 'q', 'v_m', 'v_q']
)

result = solver.solve(
    t_span=(0, 20),
    init_values=(0.0, 0.01, 0.0, 0.0),
    rho=1.0, lr=0.1, momentum=0.9
)
```

---

## 参考文献

### 基礎論文

1. **オンライン学習理論**
   - Saad, Solla (1995). "On-line learning in soft committee machines." *Phys. Rev. E*
   - Biehl, Schwarze (1995). "Learning by on-line gradient descent." *J. Phys. A*

2. **線形モデル**
   - Werfel, Xie, Seung (2005). "Learning curves for stochastic gradient descent in linear feedforward networks." *Neural Computation*

3. **パーセプトロン**
   - Opper (1996). "Online versus offline learning from random examples." *Europhys. Lett.*
   - Kinzel, Opper (1991). "Dynamics of learning." *Physics of Neural Networks*

4. **一般的なフレームワーク**
   - Engel, Van den Broeck (2001). *Statistical Mechanics of Learning*. Cambridge University Press.
   - 「On-line Learning」の章

### 主要な結果

- オンライン SGD の**最適学習率**: $\eta^* = 1/(1 + \sigma^2/\rho)$
- **漸近誤差**減衰: 最適 $\eta$ に対して $E_g(t) \sim 1/t$
- **臨界学習率**: $\eta_c = 2$（正則化なし SGD の発散閾値）
