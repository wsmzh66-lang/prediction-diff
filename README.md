# prediction-diff

用样本编号对齐同一批分类样本的两次预测，找出新版出错或修对的样本。

## 安装并比较示例

在项目根目录用 Python 3.10 或更高版本创建虚拟环境。Windows PowerShell 下运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS 或 Linux 下运行：

```sh
python3 -m venv .venv
. .venv/bin/activate
```

安装后，依次传入旧文件和新文件：

```sh
python -m pip install .
prediction-diff examples/old.csv examples/new.csv
```

示例输出：

```text
成功比较 4 条，新错 1，修对 1
新错 002 | 原文 屏幕出现坏点 | 真实标签 负面 | 旧预测 负面 | 新预测 正面
修对 001 | 原文 客服很耐心 | 真实标签 正面 | 旧预测 负面 | 新预测 正面
```

这里的“新错”表示旧版预测正确、新版预测错误。“修对”方向相反。同一样本两版都错，即使两次猜测不同，也归入“都错”。查看全部四类明细：

```sh
prediction-diff examples/old.csv examples/new.csv --all
```

## 比较自己的文件

两份 UTF-8 CSV 都需要编号列和预测列，可以带有原文列 `text`。编号作为字符串匹配，保留前导零；文件行序不影响结果。真实标签可以在任一预测文件中，也可以来自独立标签表。工具按编号合并各处的非空标签。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--id-col` | 所有输入共用的编号列名 | `sample_id` |
| `--pred-col` | 两份预测文件共用的预测列名 | `prediction` |
| `--label-col` | 所有标签来源共用的真实标签列名 | `true_label` |
| `--labels` | 包含编号列和真实标签列的独立 CSV 路径 | 仅用内嵌标签 |
| `--all` | 同时列出都对和都错的样本 | 仅列新错、修对 |

例如，两份预测表使用 `id,guess`，独立标签表使用 `id,gold`：

```sh
prediction-diff old.csv new.csv --id-col id --pred-col guess --label-col gold --labels labels.csv
```

只有两边共有且真实标签、两版预测齐全的样本计入比较。输出分别列出单边独有的编号和共有样本缺少的字段。纯空白按缺失处理，“未知”是有效类别。空编号、同表重复编号或同编号的标签冲突会报错并退出。

## Python 调用与汇总

```python
from prediction_diff import compare_predictions

result = compare_predictions("examples/old.csv", "examples/new.csv")
print(result.counts)  # {'都对': 1, '新错': 1, '修对': 1, '都错': 1}
```

`prediction-diff` 负责按编号匹配、检查缺失与冲突，并生成逐条明细。它将对齐后的真实标签、旧预测、新预测依次交给 mlxtend 的 `mcnemar_table`，使用返回的 2×2 表得到四类计数。Python 返回值也包含样本列表 `samples` 和按原因归类的编号 `diagnostics`。
