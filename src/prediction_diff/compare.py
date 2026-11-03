import csv
from dataclasses import dataclass

import numpy as np
from mlxtend.evaluate import mcnemar_table


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    true_label: str
    old_prediction: str
    new_prediction: str

    @property
    def status(self):
        old_correct = self.old_prediction == self.true_label
        new_correct = self.new_prediction == self.true_label
        if old_correct:
            return "都对" if new_correct else "新错"
        return "修对" if new_correct else "都错"


@dataclass
class Comparison:
    samples: list[Sample]
    counts: dict[str, int]
    diagnostics: dict[str, list[str]]


def _read(path, id_col, required_col):
    records = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = {id_col, required_col} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} 缺少列 {', '.join(sorted(missing))}")
        for row in reader:
            sample_id = (row[id_col] or "").strip()
            if not sample_id:
                raise ValueError(f"{path} 第 {reader.line_num} 行编号为空")
            if sample_id in records:
                raise ValueError(f"{path} 第 {reader.line_num} 行编号重复 {sample_id}")
            records[sample_id] = row
    return records


def compare_predictions(old_path, new_path, *, id_col="sample_id", pred_col="prediction",
                        label_col="true_label", labels=None):
    """按字符串编号对齐两份 CSV，返回样本明细和四类计数。"""
    old = _read(old_path, id_col, pred_col)
    new = _read(new_path, id_col, pred_col)
    external = _read(labels, id_col, label_col) if labels is not None else {}
    truth = {}
    for source in (old, new, external):
        for sample_id, row in source.items():
            label = (row.get(label_col) or "").strip()
            if label:
                previous = truth.setdefault(sample_id, label)
                if label != previous:
                    raise ValueError(f"编号 {sample_id} 的真实标签冲突：{previous} / {label}")
    samples = []
    diagnostics = {
        "仅旧版有": sorted(old.keys() - new.keys()),
        "仅新版有": sorted(new.keys() - old.keys()),
        "缺少真实标签": [],
        "缺少旧预测": [],
        "缺少新预测": [],
    }
    for sample_id in sorted(old.keys() & new.keys()):
        before, after = old[sample_id], new[sample_id]
        label = truth.get(sample_id, "")
        old_pred = (before[pred_col] or "").strip()
        new_pred = (after[pred_col] or "").strip()
        for reason, value in (("缺少真实标签", label), ("缺少旧预测", old_pred), ("缺少新预测", new_pred)):
            if not value:
                diagnostics[reason].append(sample_id)
        if label and old_pred and new_pred:
            samples.append(Sample(sample_id, before.get("text") or after.get("text") or "",
                                  label, old_pred, new_pred))

    table = mcnemar_table(
        np.array([s.true_label for s in samples], dtype=str),
        np.array([s.old_prediction for s in samples], dtype=str),
        np.array([s.new_prediction for s in samples], dtype=str),
    )
    counts = dict(zip(("都对", "新错", "修对", "都错"), map(int, table.flat)))
    return Comparison(samples, counts, diagnostics)
