import argparse
import csv

from .compare import compare_predictions


def main():
    parser = argparse.ArgumentParser(description="按样本编号比较两版分类预测")
    parser.add_argument("old", help="旧版预测 CSV")
    parser.add_argument("new", help="新版预测 CSV")
    parser.add_argument("--id-col", default="sample_id", help="样本编号列名")
    parser.add_argument("--pred-col", default="prediction", help="预测类别列名")
    parser.add_argument("--label-col", default="true_label", help="真实标签列名")
    parser.add_argument("--labels", help="独立真实标签 CSV，包含编号列和真实标签列")
    parser.add_argument("--all", action="store_true", help="同时列出都对、都错的样本")
    args = parser.parse_args()
    try:
        result = compare_predictions(args.old, args.new, id_col=args.id_col, pred_col=args.pred_col,
                                     label_col=args.label_col, labels=args.labels)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(2, f"错误：{exc}\n")
    print(f"成功比较 {len(result.samples)} 条，新错 {result.counts['新错']}，修对 {result.counts['修对']}")
    statuses = ("新错", "修对", "都对", "都错") if args.all else ("新错", "修对")
    for status in statuses:
        for sample in result.samples:
            if sample.status == status:
                print(f"{status} {sample.sample_id} | 原文 {sample.text} | 真实标签 {sample.true_label}"
                      f" | 旧预测 {sample.old_prediction} | 新预测 {sample.new_prediction}")
    for reason, ids in result.diagnostics.items():
        if ids:
            print(f"{reason}：{', '.join(ids)}")
