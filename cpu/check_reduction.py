"""CPU 分块归约的正确性实验。从仓库根目录运行 python cpu/check_reduction.py。"""

import json
import math
import platform
from pathlib import Path


PARTS = 4
REL_TOL = 1e-12
ABS_TOL = 1e-12


def row_ranges(rows, parts):
    """整数边界划分半开区间。4097 行分成 1024、1024、1024、1025 行。"""
    return [(k * rows // parts, (k + 1) * rows // parts) for k in range(parts)]


def patterned_input(rows, cols):
    """按需生成实数元素，用整数等差数列公式独立计算参考平方和。"""
    offset = rows // 2 + cols // 2

    def value(i, j):
        return (i + j - offset) / 10

    # 对每行先在整数域计算 Σ(a+j)² = C·a² + 2a·Σj + Σj²。
    sum_j = cols * (cols - 1) // 2
    sum_j_squared = cols * (cols - 1) * (2 * cols - 1) // 6
    numerator = 0
    for i in range(rows):
        a = i - offset
        numerator += cols * a * a + 2 * a * sum_j + sum_j_squared

    reference = numerator / 100
    description = {
        "formula": "(row + col - (rows // 2 + cols // 2)) / 10",
        "exact_reference_numerator": numerator,
        "exact_reference_denominator": 100,
    }
    return value, reference, description


def compare_case(name, rows, cols, value, reference, description):
    ranges = row_ranges(rows, PARTS)
    tiles = []
    for start, end in ranges:
        partial = math.fsum(value(i, j) ** 2
                            for i in range(start, end) for j in range(cols))
        tiles.append({"row_start": start, "row_stop": end,
                      "row_count": end - start, "partial_sum_squares": partial})

    result = math.fsum(tile["partial_sum_squares"] for tile in tiles)
    norm = math.sqrt(result)
    reference_norm = math.sqrt(reference)
    checks = {
        "covers_all_rows_without_overlap": (
            ranges[0][0] == 0 and ranges[-1][1] == rows
            and all(0 <= a <= b <= rows for a, b in ranges)
            and all(ranges[k][1] == ranges[k + 1][0] for k in range(PARTS - 1))
        ),
        "total_element_count_matches": (
            sum(tile["row_count"] * cols for tile in tiles) == rows * cols
        ),
        "sum_squares_matches_reference": (
            math.isclose(result, reference, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ),
        "frobenius_norm_matches_reference": (
            math.isclose(norm, reference_norm, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"{name}: {checks}")
    return {
        "case": name, "shape": [rows, cols], "input": description,
        "tiles": tiles, "sum_squares": result, "reference_sum_squares": reference,
        "sum_squares_abs_error": abs(result - reference),
        "frobenius_norm": norm, "reference_frobenius_norm": reference_norm,
        "norm_abs_error": abs(norm - reference_norm), "checks": checks,
    }


def main():
    # 手算样例先锁定局部和及最终结果：9 + 4 + 8 + 4 = 25，开方得 5。
    hand = [[-3, 0], [0, -2], [2, -2], [0, 2]]
    sample = compare_case("hand_verifiable", 4, 2, lambda i, j: hand[i][j],
                          25, {"values": hand, "reference": "9 + 4 + 8 + 4"})
    sample["checks"]["hand_computed_partial_sums"] = (
        [tile["partial_sum_squares"] for tile in sample["tiles"]] == [9, 4, 8, 4]
    )
    if not sample["checks"]["hand_computed_partial_sums"]:
        raise AssertionError("手算示例中的局部平方和不一致")

    cases = [sample]
    for name, rows, cols in [
        ("lecture_shape", 4096, 1024),
        ("uneven_tail", 4097, 17),
        ("fewer_rows_than_tiles", 2, 7),
        ("empty_matrix", 0, 7),
    ]:
        value, reference, description = patterned_input(rows, cols)
        cases.append(compare_case(name, rows, cols, value, reference, description))

    report = {
        "experiment": "CPU 分块平方和与 Frobenius 范数功能验证",
        "execution": "单进程顺序执行四个逻辑数据块",
        "reproduce_from_repo_root": "python cpu/check_reduction.py",
        "python_version": platform.python_version(),
        "partitions": PARTS, "relative_tolerance": REL_TOL,
        "absolute_tolerance": ABS_TOL, "case_count": len(cases), "cases": cases,
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases passed; cpu/results.json")


if __name__ == "__main__":
    main()
