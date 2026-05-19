# MEOW 数据集说明（给 Agent 读）

## 文件布局

- **实体数据（唯一）：** `/data/moew/data/{YYYYMMDD}.h5`
- 本目录下的 `data/` 是指向上述路径的符号链接
- `resources/calendar`：交易日列表（整数 `YYYYMMDD`）

代码里用 `data_io.default_h5dir()` 或 `data_io.verify_data_dir()` 解析路径。

## 样本粒度

每一行 = 一个 `(symbol, date, interval)`：

| 列 | 含义 |
|----|------|
| `symbol` | 股票代码 |
| `date` | 交易日 |
| `interval` | 日内时间片编号（约 226 个/天） |
| `fret12` | **标签**：未来 12 个 interval 的收益（预测目标） |

约 **309 只股票 × 226 interval × 交易日数** 行/天。

## 原始字段（62 列，节选）

**价格 / 盘口：** `midpx`, `lastpx`, `bid0`–`bid19`, `ask0`–`ask19`, `bsize*`, `asize*`, `btr*`, `atr*`

**成交：** `tradeBuyQty`, `tradeSellQty`, `tradeBuyTurnover`, …

**挂单 / 撤单：** `nAddBuy`, `addBuyQty`, `nCxlSell`, …

完整列名：在 worktree 中执行 `python -c "import pandas as pd; print(list(pd.read_hdf('data/20230601.h5').columns))"`

## 固定划分（勿改）

| 集合 | 日期 | 天数 |
|------|------|------|
| 训练 | 20230601 – 20231130 | 123 |
| 测试 | 20231201 – 20231229 | 21 |

## 推荐加载方式

```python
from data_io import load_day, train_test_dates, frame_to_xy, LOB_PRICE_SIZE_COLS

train_dates, test_dates = train_test_dates()
df = load_day("data", 20230601)
X, y = frame_to_xy(df, feature_cols=LOB_PRICE_SIZE_COLS, max_rows=200_000)
```

## 内存注意

- 单次读入 123 天 ≈ **4GB+**，会 OOM
- 按日或按周分批；训练时可 `max_rows` 子采样

## 旧版 baseline（仅供对照）

`reference/legacy_baseline/` 是工作坊自带的 **Ridge + 6 手工特征**（Pearson≈0.022），**不是**本次迭代目标。
