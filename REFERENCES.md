# 建议起步的近期模型（微观结构 / LOB）

实现时请在 `solution.py` 或 `models/` 中 **cite 论文**，并说明改动。

| 模型 | 文献 | 要点 | 本仓库起点 |
|------|------|------|------------|
| **DeepLOB** | Zhang et al., *DeepLOB*, 2019 | CNN + Inception + LSTM，LOB 快照序列 | `models/deeplob_lite.py` |
| **TLOB** | Berti & Kasneci, *TLOB*, 2024 | Transformer 编码 LOB 序列 | `models/tlob_starter.py` |
| **HLOB** | 2024 hierarchical LOB | 层次化表示 + 时序模型 | 可自行添加 |
| **STAN** | 2024 spatiotemporal attention | 跨股票 + 时间注意力 | 可自行添加 |
| **TabPFN / FT-Transformer** | 表格深度模型 | 行级特征、无显式序列 | 适合先作强 baseline |

## 建模提示

1. **序列构造**：按 `(symbol, date)` 对 `interval` 排序，把 LOB 列堆成 `(T, F)` 或 `(T, 4, L)` 再喂网络。
2. **截面**：同一 `interval` 上多只股票可做 cross-sectional norm 或 attention。
3. **损失**：MSE 在 `fret12` 上可行；也可直接优化 correlation（eval 指标）。
4. **勿泄漏**：`fret12` 是未来收益；特征只能用当前及过去 interval。
5. **算力**：默认 CPU + 分块；可用 GPU，注意 `n_chunks` 与 batch size。

## 评分

`run_benchmark.py` 输出测试集 **Pearson(fret12, forecast)**，越高越好。
