"""Black-box acceptance tests. Run with the interpreter of an installed copy."""
import csv
import os
from pathlib import Path
import sys
import subprocess
import sysconfig
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = Path(sysconfig.get_path("scripts")) / ("prediction-diff.exe" if os.name == "nt" else "prediction-diff")
FIELDS = ["sample_id", "prediction", "true_label", "text"]
OLD = [
    ["001", "old-error", "class-a", "修对原文"],
    ["002", "class-b", "class-b", "新错原文"],
    ["003", "wrong-c", "class-c", "都错原文"],
    ["004", "class-d", "class-d", "都对原文"],
]
NEW = [
    ["001", "class-a", "class-a", "修对原文"],
    ["002", "new-error", "class-b", "新错原文"],
    ["003", "wrong-again", "class-c", "都错原文"],
    ["004", "class-d", "class-d", "都对原文"],
]


class Acceptance(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def csv(self, name, rows, fields=FIELDS):
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def run_cli(self, *args, ok=True):
        result = subprocess.run([CLI, *args], cwd=ROOT, capture_output=True,
                                encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return result.stdout
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stderr + result.stdout

    def counts(self, output, total, new_errors, fixes):
        for label, count in [("比较", total), ("新错", new_errors), ("修对", fixes)]:
            self.assertRegex(output, rf"{label}[：:\s]*{count}(?!\d)")

    def changes(self, output, with_text=True):
        self.counts(output, 4, 1, 1)
        start = output.index("002")
        end = output.index("001")
        self.assertLess(start, end, output)
        for block, label, before, after in [
            (output[start:end], "class-b", "class-b", "new-error"),
            (output[end:], "class-a", "old-error", "class-a"),
        ]:
            for key, value in [("真实标签", label), ("旧预测", before), ("新预测", after)]:
                self.assertRegex(block, rf"{key}[：:\s]*{value}\b")
        if with_text:
            self.assertIn("新错原文", output[start:end])
            self.assertIn("修对原文", output[end:])
        for hidden in ["003", "004"]:
            self.assertNotIn(hidden, output)

    def test_shipped_example(self):
        output = self.run_cli("examples/old.csv", "examples/new.csv")
        self.counts(output, 4, 1, 1)
        self.assertLess(output.index("002"), output.index("001"))
        for name in ["003", "004"]:
            self.assertNotIn(name, output)

    def test_row_order_and_details(self):
        for a, b in [(OLD, NEW[::-1]), (OLD[::-1], NEW[2:] + NEW[:2])]:
            with self.subTest(first_old=a[0][0]):
                output = self.run_cli(self.csv("old.csv", a), self.csv("new.csv", b))
                self.changes(output)

    def test_optional_text(self):
        old = self.csv("old.csv", [r[:3] for r in OLD], FIELDS[:3])
        new = self.csv("new.csv", [r[:3] for r in reversed(NEW)], FIELDS[:3])
        self.changes(self.run_cli(old, new), with_text=False)

    def test_public_api(self):
        from prediction_diff import compare_predictions
        expected = {"001": "修对", "002": "新错", "003": "都错", "004": "都对"}
        for a, b in [(OLD, NEW[::-1]), (OLD[::-1], NEW)]:
            result = compare_predictions(self.csv("old.csv", a), self.csv("new.csv", b))
            self.assertEqual({s.sample_id: s.status for s in result.samples}, expected)
            self.assertEqual(result.counts, {label: 1 for label in expected.values()})


    def test_runtime_uses_mlxtend(self):
        from mlxtend.evaluate import mcnemar_table
        from prediction_diff import compare_predictions
        calls = []
        def trace(frame, event, arg):
            if event == "call" and frame.f_code is mcnemar_table.__code__:
                calls.append(frame.f_code.co_name)

        old = self.csv("old.csv", OLD)
        new = self.csv("new.csv", NEW[::-1])
        previous = sys.getprofile()
        sys.setprofile(trace)
        try:
            compare_predictions(old, new)
        finally:
            sys.setprofile(previous)
        self.assertIn("mcnemar_table", calls)

    def test_clear_new_prediction(self):
        with (ROOT / "examples/new.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            if row["sample_id"] == "002":
                row["prediction"] = ""
        new = self.csv("blank-new.csv", ([r[k] for k in FIELDS] for r in records))
        output = self.run_cli("examples/old.csv", new)
        self.counts(output, 3, 0, 1)
        self.assertRegex(output, r"缺[^\n]*新[^\n]*预测[^\n]*002")
        self.assertIn("修对 001", output)

    def test_all_diagnostics_and_unknown(self):
        old = [["only-old", "x", "x"], ["no-label", "x", " "],
               ["no-old-pred", "  ", "x"], ["no-new-pred", "x", "x"],
               ["literal", "x", "未知"]]
        new = [["only-new", "x", "x"], ["no-label", "x", ""],
               ["no-old-pred", "x", "x"], ["no-new-pred", " ", "x"],
               ["literal", "未知", "未知"]]
        output = self.run_cli(self.csv("old.csv", old, FIELDS[:3]),
                              self.csv("new.csv", new[::-1], FIELDS[:3]))
        self.counts(output, 1, 0, 1)
        self.assertIn("修对 literal", output)
        for label, key in [("仅.*旧", "only-old"), ("仅.*新", "only-new"),
                           ("缺.*标签", "no-label"), ("缺.*旧.*预测", "no-old-pred"),
                           ("缺.*新.*预测", "no-new-pred")]:
            self.assertRegex(output, label + r"[^\n]*" + key)

    def test_validation_locations(self):
        for rows, diagnostic, identifier in [
            ([[" ", "x", "x"]], "空", "2"),
            ([["002", "x", "x"]] * 2, "重复", "002"),
            ([["002", "x", "contradiction"]], "冲突", "002"),
        ]:
            with self.subTest(diagnostic=diagnostic):
                output = self.run_cli(self.csv("old.csv", OLD), self.csv("bad.csv", rows, FIELDS[:3]), ok=False)
                self.assertIn(diagnostic, output)
                self.assertIn(identifier, output)

    def test_custom_headers_and_label_locations(self):
        from prediction_diff import compare_predictions
        labels = self.csv("truth.csv", [[r[0], r[2]] for r in OLD[1:] + OLD[:1]], ["uid", "truth"])
        for location in ["old", "new", "external"]:
            with self.subTest(location=location):
                files = []
                for side, rows in [("old", OLD[::-1]), ("new", NEW[2:] + NEW[:2])]:
                    fields = ["uid", "guess", "text"]
                    data = [[r[0], r[1], r[3]] for r in rows]
                    if side == location:
                        fields.append("truth")
                        for row, source in zip(data, rows):
                            row.append(source[2])
                    files.append(self.csv(side + ".csv", data, fields))
                options = ["--id-col", "uid", "--pred-col", "guess", "--label-col", "truth"]
                kwargs = dict(id_col="uid", pred_col="guess", label_col="truth")
                if location == "external":
                    options.extend(["--labels", labels])
                    kwargs["labels"] = labels
                self.changes(self.run_cli(*files, *options))
                result = compare_predictions(*files, **kwargs)
                self.assertEqual({s.sample_id: s.status for s in result.samples},
                                 {"001": "修对", "002": "新错", "003": "都错", "004": "都对"})
                self.assertEqual(result.counts, dict.fromkeys(["修对", "新错", "都对", "都错"], 1))

    def test_external_label_conflict(self):
        labels = self.csv("truth.csv", [["001", "contradiction"]], ["sample_id", "true_label"])
        output = self.run_cli(self.csv("old.csv", OLD), self.csv("new.csv", NEW[::-1]),
                              "--labels", labels, ok=False)
        self.assertIn("冲突", output)
        self.assertIn("001", output)

    def test_identical_predictions(self):
        old = self.csv("old.csv", OLD)
        new = self.csv("new.csv", OLD[::-1])
        output = self.run_cli(old, new)
        self.counts(output, 4, 0, 0)
        self.assertEqual(len(output.strip().splitlines()), 1)
        for row in OLD:
            self.assertNotIn(row[0], output)

    def test_all_includes_each_compared_id(self):
        output = self.run_cli(self.csv("old.csv", OLD), self.csv("new.csv", NEW[::-1]), "--all")
        self.counts(output, 4, 1, 1)
        for row, status in zip(OLD, ["修对", "新错", "都错", "都对"]):
            self.assertIn(f"{status} {row[0]}", output)
            self.assertEqual(output.count(row[0]), 1)

    def test_asymmetric_changes(self):
        old = [["a07", "蓝", "蓝"], ["b83", "黑", "白"], ["c12", "紫", "绿"]]
        new = [["c12", "绿", "绿"], ["a07", "黄", "蓝"], ["b83", "白", "白"]]
        output = self.run_cli(self.csv("old.csv", old, FIELDS[:3]), self.csv("new.csv", new, FIELDS[:3]))
        self.counts(output, 3, 1, 2)
        for sample, status, before, after in [("a07", "新错", "蓝", "黄"), ("b83", "修对", "黑", "白"),
                                               ("c12", "修对", "紫", "绿")]:
            self.assertRegex(output, rf"{status} {sample}[^\n]*旧预测 {before}[^\n]*新预测 {after}")
        for key in ["b83", "c12"]:
            self.assertLess(output.index("a07"), output.index(key))
