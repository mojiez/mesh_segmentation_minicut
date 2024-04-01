import argparse
import multiprocessing
from pathlib import Path
import logging

# 导包
from ..utils.constants import SegmenterType
from ..utils.utils import parse_ply, write_ply
from ..graph import DualGraph
from ..segmenters import BinaryRecursive, BinarySegmenter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=Path,
        required=True,
        help="Input .ply file path",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default=Path("output_decompose_new.ply"),
        help="Output .ply filename",
    )
    parser.add_argument(
        "-s",
        "--segmenter",
        type=str,
        default=SegmenterType.binary,
        choices=list(SegmenterType),
    )
    parser.add_argument(
        "-k",
        "--num_levels",
        type=int,
        default=1,
        # 分割的等级 默认是一级 如果是2的话应该是分成4个部分？
        help="Number of segmentation levels for the binary segmenter.",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=multiprocessing.cpu_count(),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    # TODO: add some validation fof arguments
    return parser.parse_args()


def main():
    # 基于模糊聚类和最小割的层次化网格分割算法
    # 这个代码支持二元分割 和 二元递归分割 接下来将对代码进行解析

    # 整理步骤：
    # 1. 计算网格中所有相邻面片之间的距离
    # 2. 计算每个面片属于不同分割区域的概率
    # 3. 迭代调整每个面片的概率，直到收敛
    # 4. 在模糊区域里寻找准确的分割边界
    args = _parse_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level))

    # Parse ply file, form Mesh
    # 从Mesh模型中解析ply文件 文件路径是必须要输入的
    mesh = parse_ply(ply_path=args.input_file)

    # todo 猜测是进行 计算对偶图的操作 依此得到对偶图
    dual_graph = DualGraph(mesh)

    # Choose segmenter
    # 选择 分割方法？
    if args.segmenter == SegmenterType.binary:
        if args.num_levels == 1:
            # Binary
            # 构造函数
            segmenter = BinarySegmenter(num_workers=args.num_threads)
        else:
            # Binary recursive
            segmenter = BinaryRecursive(
                num_levels=args.num_levels,
                num_workers=args.num_threads,
            )

    # 进行分割操作 核心代码 segmenter是一个类
    # Segment
    # 调用了BinarySegmenter的call方法
    out_mesh = segmenter(mesh=mesh, dual_graph=dual_graph)

    # Output results
    write_ply(mesh=out_mesh, out_path=args.output_file)


if __name__ == "__main__":
    main()
