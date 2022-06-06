# DSAI-HW3-2021

## 執行需求
- Python 3
- PyTorch
- Numpy
- Pandas
- OpenAI Gym
- stable_baselines3

## 執行方式

```
python main.py --consumption ./sample_data/consumption.csv --generation ./sample_data/generation.csv --bidresult ./sample_data/bidresult.csv --output output.csv
```

## 想法

使用 Reinforcement Learning 的 PPO（Proximal Policy Optimization），可以輸入目前 state，產生連續型態的 action，然後盡可能獲得最大的 reward（這邊是讓電費越少越好，所以可以把電費取負號再 normalize）。

## State 與 Action

以每小時為單位，一次丟入七天的資料

State：consumption、generation

共 336 個輸入

以每小時為單位，輸出未來一天投標資訊

Action：action、target_price、target_volume

共 72 個輸出

Reward：由 bidresult 與其他資料去計算電費，並且 normalize，盡量控制在 -1 到 1 之間

## Q-Learning

Value-based 方法，可以用 TD 或 MC

論文：[Learning from Delayed Rewards](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

照下面的原則找到新的 policy 一定會比較好（證明略）

![](https://i.imgur.com/czUT82f.png)

![](https://i.imgur.com/ZP7Ia0F.png)

使用目前的狀態下最大的 Q 來更新：

$Q^{\pi}(s_{t+1}, a_{t+1}) := \max\limits_{a} Q^{\pi}(s_{t+1}, a)$

傳統方式（非 NN，更新 Q-Table，只是用離散數值且少量）：

![](https://i.imgur.com/xwF09Bi.png)

DQN（Deep Q-Learning）：用 NN 來學 Q Function

![](https://i.imgur.com/iEk8kN7.png)

原始論文：[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

後來才知道 DQN 沒辦法處理連續動作，所以又去研究 PPO

## Policy-based Methods

選擇 Policy

- 又稱作 Actor

### Policy Gradient
- 定義：軌跡 Trajectory $\tau=\left\{s_1,a_1,s_2,a_2,...,s_T,a_T\right\}$
- 假設：對於環境的機率分佈 $P(s'|s,a)$，一個 state 只跟前面的 action 和前一個 state 有關
    - 不可以調（跟 policy 無關，微分不影響）
- Policy 是一個機率分佈 $\pi_\theta(a \vert s)$，可以用 network 學出來
    - 可以調
- Log Likelihood
    - $p_\theta(\tau)=P(s_1)\pi_\theta(a_1|s_1)P(s_2|s_1,a_1)\pi_\theta(a_2|s_2)P(s_3|s_2,a_2)...$
      $\displaystyle = P(s_1)\prod_{t=1}^T \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)$
    - $\displaystyle \log p_\theta(\tau)= \sum_{t=1}^T \log \pi_\theta(a_t|s_t) + \log P(s_1) + \sum_{t=1}^T \log P(s_{t+1}|s_t,a_t)$
        - 後面跟 $\theta$ 無關，對 $\theta$ 微分等於 0
- 目標：最大化以 Advantage 作權重的 log likelihood
    - $\nabla J(\theta) = E_{\tau \sim p_\theta(\tau)}\left[A(\tau) \nabla \log p_\theta(\tau)\right]$，$p_\theta(\tau)$ 未知 $\Rightarrow$ sample
      $\displaystyle \approx \dfrac{1}{N} \sum^N_{n=1} A(\tau^n) \nabla \log p_\theta(\tau^n)$
      $\displaystyle =\dfrac{1}{N} \sum^N_{n=1} \sum^{T_n}_{t=1} A(\tau^n) \nabla \log \pi_\theta(a^n_t \vert s^n_t)$
    - $l(\theta) = -J(\theta) = E_{\tau \sim Environment\ and\ Policy}[-A(\tau) \log p_\theta(\tau)]$
    - 可想成是權重為 Advantage 的 Weighted Cross Entropy
    - 這個 Loss Function $J(\theta)$ 的梯度就是 Policy Gradient
    - 最佳化的過程就是用 Policy Gradient 更新參數
    - Advantage A = Return - Baseline
        - 常用 Q 當作 Return
        - 常用 V（Q 的期望值） 當作 Baseline
        - Baseline 目的是希望有正有負，好訓練
    - 一般而言要使用一整個 Trajectory $\tau$ 訓練，除非 Advantage 是使用 Temporal Difference（TD） 之類的
    - 這種最基本的（Vanilla）Policy Gradient 叫做 REINFORCE 演算法
- 限制：On-policy
- 論文：[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)


### Actor-Critic Methods（AC）

延伸自 Policy Gradient（Actor），並加入了估計 Value 的 Network（Critic）

Actor-Critic 的概念其實很早就有了

論文：[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

#### Advantage 的選擇
$A^{\pi_\theta}(s_t^n,a_t^n) = G_t^n - b = Q^{\pi_\theta}(s_t^n,a_t^n) - V^{\pi_\theta}(s_t^n)$
- 上標 $n$ 代表從 $p_\theta(\tau)$ sample 出的樣本
- 用 $G_t^n$ 的期望值（V, Q）代替 sample 的值會比較穩定
    - 用 Value-based 的方式估測！
    - baseline 選 V 是因為 V 又是 Q 的期望值
    - 訓練時就不用丟一整個 Trajectory $\tau$ 了
- 但同時估兩個 Network 不准的風險高，為了減少 Network，我們把 Q 的值用 V 表示
    - $Q^{\pi}(s_t^n,a_t^n) = E[r_t^n + \gamma V^{\pi}(s_{t+1}^n)|s=s_t^n] \approx r_t^n + \gamma V^{\pi}(s_{t+1}^n)$
    - 但這樣會有一點 $r_t$ 的隨機性，有些 variance
    - 不過比原來直接用 $G_t$ 好很多
    - A3C 實驗證明這樣的組合最好
- 結論：Advantage $A^{\pi}(s_t^n,a_t^n) = r_t^n + \gamma V^{\pi}(s_{t+1}^n) - V^{\pi}(s_t^n)$
    - $A(s,a) = Q(s,a) - V(s) \approx r + \gamma V(s') - V(s)$
    - $J(\theta) = E_{\tau}[E_{(s,a)\ in\ \tau}[-A(s,a) \log \pi_\theta(a|s)]]$

### A2C（Advantage Actor-Critic）

由 Policy-Gradient 延伸

![](https://i.imgur.com/z9HtuJx.png)

#### 技巧
- 共用 Network
      ![](https://i.imgur.com/JJY7htn.png)
    - Actor Network 與 Critic Network 共用前幾層 Network
    - 理由：前幾層特徵是大家共享的，特別對於影像輸入

### A3C（Asynchronous Advantage Actor-Critic）

A2C 的平行運算版本

- 開分身（worker）同時訓練
- Copy 參數
- 個別收集環境資料
- 算 Gradient，Update「Global」的參數
    - 不用在意別人
    - 注意回傳的是 Gradient 不是環境資料

### TRPO（Trust Region Policy Optimization）

Actor-Critic 系列

#### Off-policy
- 不用像 On-policy 每一輪訓練都要重新 sample
- Importance Sampling
  ![](https://i.imgur.com/zAcu0jm.png)
- 這技巧在很多其他地方也可以用

#### 假設
- 兩個 policy 分佈不會差太遠
    - 機率分佈差距不大
    - 用 KL-Divergence 作為 Regularization Term
        - 把 $D_{KL}(p_\theta\|p_{\theta'}) < \delta$ 當作額外的 contrain（限制條件）
    - ![](https://i.imgur.com/lyfV0MK.png)
- 兩個 policy 的 Advantage 不會差太多
- 兩個 policy 看到相同 state 的機率差不多

目標
- 最大化從舊分佈去 sample 「(新的 $\pi(s|a)$ / 舊的 $\pi(s|a)$) * Advantage」的期望值
- 看不到 Log Likelihood 的部份因為被吸收了

論文：[Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)


### PPO（Proximal Policy Optimization）

被很多套件當成是預設的演算法

延伸自 TRPO，實作上比 TRPO 簡單

PPO2 使用 ε clip 技巧控制機率分佈不要差太多：
![](https://i.imgur.com/sgLEZXm.png)
（先試 ε = 0.2～0.1）

與 TRPO 的差別
- TRPO 把 KL-Divergence 當作額外的 constrain
- PPO 直接納入目標函數
- PPO2 用 ε clip（最容易實作）

論文：[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)


## Source

  - [Slide](https://docs.google.com/presentation/d/1JW27_5HXYZhqWmgvDhtXBaFTOfksO_dS/edit#slide=id.p1)
  - [Dashboard](https://docs.google.com/spreadsheets/d/1cjhQewnXT2IbmYkGXRYNC5PlGRafcbVprCjgSFyDAaU/edit?pli=1#gid=0)

## Rules

- SFTP

```

┣━ upload/
┗━ download/
   ┣━ information/
   ┃  ┗━ info-{mid}.csv
   ┣━ student/
   ┃  ┗━ {student_id}/
   ┃     ┣━ bill-{mid}.csv
   ┃     ┗━ bidresult-{mid}.csv
   ┗━ training_data/
      ┗━ target{household}.csv  
      
```

1. `mid` 為每次媒合編號
2. `household` 為住戶編號，共 50 組
3. 請使用發給組長的帳號密碼，將檔案上傳至 `upload/`
4. 相關媒合及投標資訊皆在 `download/` 下可以找到，可自行下載使用


- File

```

┗━ {student_id}-{version}.zip
   ┗━ {student_id}-{version}/
      ┣━ Pipfile
      ┣━ Pipfile.lock
      ┣━ main.py
      ┗━ {model_name}.hdf5

```

1. 請務必遵守上述的架構進行上傳 (model 不一定要有)
2. 檔案壓縮請使用 `zip`，套件管理請使用 `pipenv`，python 版本請使用 `3.8`
3. 檔名：{學號}-{版本號}.zip，例：`E11111111-v1.zip`
4. 兩人一組請以組長學號上傳
5. 傳新檔案時請往上加版本號，程式會自動讀取最大版本
6. 請儲存您的模型，不要重新訓練

- Bidding

1. 所有輸入輸出的 csv 皆包含 header
2. 請注意輸入的 `bidresult` 資料初始值為空
3. 輸出時間格式為 `%Y-%m-%d %H:%M:%S` ，請利用三份輸入的 data 自行選一份，往後加一天即為輸出時間  
   例如: 輸入 `2018-08-25 00:00:00 ~ 2018-08-31 23:00:00` 的資料，請輸出 `2018-09-01 00:00:00 ~ 2018-09-01 23:00:00` 的資料(一次輸出`一天`，每筆單位`一小時`)
4. 程式每次執行只有 `120 秒`，請控制好您的檔案執行時間
5. 每天的交易量限制 `100 筆`，只要有超出會全部交易失敗，請控制輸出數量
