

# 隐变量蒸馏, 特征蒸馏, 表征蒸馏

- student encoder 的梯度只来自 MSE loss， 不会反传到 Actor 或 Privilege Encoder
- 反向时 latent_targets 用 torch.no_grad() 包裹，teacher encoder 不更新
- 所以 teacher 和 student 在阶段 1 共享数据但不共享梯度
## 为什么这样设计
这是 "latent distillation" 的典型做法（参考 Privileged Learning / Asymmetric Actor-Critic 系列）：

1. Teacher 用特权观测（不可部署的信息）训出强策略，同时学出 privilege_encoder 把特权信息压缩到 64 维 latent
2. Student 用可部署观测（深度图）去 重建同一个 latent 表征
3. 因为 Actor 是共享的，只要 student 的 latent 和 teacher 的 latent 越接近，Actor 接到 student latent 时输出的 action 就和 teacher 越接近
4. 阶段 1 不让 student 进 Actor，是为了避免 PPO 的 policy gradient 和 latent MSE 的梯度在 Actor 上互相干扰


所以代码刻意把两次拆开： teacher 先完整走一遍 PPO 更新完毕，student 再用更新后的 teacher encoder 作为 target 跑一遍 MSE 。这是一种"先更新 target，再拟合 target"的串行策略。


# 阶段 1: PPO + Latent 蒸馏
~~~python
Teacher (privileged):
   privileged_obs ──→ [Privilege Encoder] ──→ latent_t (64) ──→ Actor ──→ action
                                                  ▲
                                                  │
                                              MSE 蒸馏
                                                  │
                                                  ▼
Student (deployable):
   depth + actor_obs ──→ [CNN + GRU] ──→ latent_s (64) ──→ (阶段1到此结束)


═════════════════════════════════════════════════════════════════════════
                        阶段 1: PPO + Latent 蒸馏
═════════════════════════════════════════════════════════════════════════
  [Teacher 分支 — 走完整 PPO]
   privileged_obs ──► [Privilege Encoder] ──► latent_t (64) ──┐
                                                              │
   actor_obs ─────────────────────────────────────────────────┤
                                                              ▼
                                                       ┌────────────┐
                                                       │  Actor     │──► action
                                                       │  (shared)  │
                                                       └────────────┘

   critic_obs ──────────────────────────────────────────►┌────────────┐
                                                         │  Critic    │──► value
                                                         │  (shared)  │
                                                         └────────────┘

                                       PPO Loss = value_loss + surrogate - entropy
                                                          │
                                                          ▼
                                    teacher_optimizer.step()  (lr=1e-3)
                             更新: privilege_encoder, actor, critic, std

   ─────────────────────────────────────────────────────────────────────
                       同一 mini-batch 跑第二遍
   ─────────────────────────────────────────────────────────────────────
   [Student 分支 — 只算 latent MSE]

    depth + actor_obs ──► [CNN + GRU] ──► latent_s (64) ──┐
                                                          │
                                                          │ MSE
                                                          ▼
    privileged_obs ──► [Privilege Encoder (no_grad)] ──► latent_t (64)
                                                          ▲
                                                          │
                                                          │
                                        Loss = MSE(latent_s, latent_t.detach())
                                                          │
                                                          ▼
                                    student_optimizer.step()  (lr=2e-4)
                               只更新: depth_history_encoder
════════════════════════════════════════════════════════════════════════
~~~

# 阶段 2: 行为克隆
+ 训练完全不用奖励函数, 阶段 2 的 reward 是 评估指标 ，不是 训练信号
+ 让 student 侧的 Actor 和 CNN+GRU 协同微调 ，使 student 在 只用深度图 （不能看特权信息）的条件下，仍能输出和 teacher 接近的 action。

~~~python
阶段 2: Action 行为蒸馏
═══════════════════════════════════════════════════════════════════
Teacher (frozen, 加载自阶段1 ckpt):
   privileged_obs ──→ [Privilege Encoder] ──→ latent_t (64) ──→ Actor ──→ action_mean_teacher
                                                                        ▲
                                                                        │
                                                                    MSE 监督
                                                                        │
                                                                        ▼
Student (训练, 可部署):                                                  
   depth + actor_obs ──→ [CNN + GRU] ──→ latent_s (64) ──→ Actor ──→ action_mean_student
                1阶段上微调                              (shared, 训练)

   ─────────────────────────────────────────────────────────────────
   • Loss = MSE(action_mean_student, action_mean_teacher.detach())
            + entropy_coef * entropy  (实际 entropy_coef=0)
   • Student 训练: depth_history_encoder + actor + std  (lr=2e-4)
   • Teacher: 整体 frozen (no_grad)
   • Critic: 不参与训练
   • PPO: 没有 ratio/surrogate/value_loss
   • mini-batch 跑 1 遍
   • Env 用 student 的 action step
   ─────────────────────────────────────────────────────────────────
~~~

# depth + actor_obs ──► [CNN + GRU] ──► latent_s (64) 

~~~python
阶段                    操作                         维度
────────────────────────────────────────────────────────────
输入                    depth_image                   (N, 1, 30, 40)
                        actor_obs                    (N, 45)

CNN 分支
  ①                     Conv2d(1→8, k5, s1)           (N, 8, 26, 36)
  ②                     MaxPool2d(2, 2)              (N, 8, 13, 18)
  ③                     ELU                          (N, 8, 13, 18)
  ④                     Conv2d(8→8, k3, s1)           (N, 8, 11, 16)
  ⑤                     ELU                          (N, 8, 11, 16)
  ⑥                     Flatten                      (N, 1408)
  ⑦                     Linear(1408→128) + ELU       (N, 128)
  ⑧                     Linear(128→64) + ELU         (N, 64)

Combination MLP
  ⑨                     cat(actor_obs, depth_feat)   (N, 109)
  ⑩                     Linear(109→128) + ELU        (N, 128)
  ⑪                     Linear(128→32)                (N, 32)

GRU
                        GRU(in=32, h=512, L=1)       (N, 512)

Latent Output MLP
  ⑫                     Linear(512→64)               (N, 64)  ← latent_s
~~~