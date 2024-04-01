from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import random
import multiprocessing
import logging

from ..utils.mesh import Mesh, Face
from ..graph import DualGraph
from ..segmenters.binary import BinarySegmenter
from ..utils.utils import random_colours


class BinaryRecursive:
    # 递归调用二元分割 以获取更多的分割片段
    """Recursively call binary segmenter for more segments."""

    def __init__(
        self, num_levels: int, num_workers=multiprocessing.cpu_count()
    ):
        # Number of sub-clusters
        self._num_levels = num_levels
        self._num_workers = num_workers
        assert num_levels > 0

    def _divide_mesh(self, mesh: Mesh) -> tuple[Mesh, Mesh]:
        """Divide a mesh into 2 parts, based on colour."""
        # 将一个mesh模型分成两个部分 通过颜色？？
        # TODO: do I need to count like this? Better way?
        # Get colours
        # Counter对象 key是face value是对应面的颜色出现的次数
        colours_ctr: Counter[Face] = Counter()
        for face in mesh.faces:
            # 统计每个颜色出现的次数
            colours_ctr[face.colour] += 1
        # 将出现频率最高的两个颜色和对应的次数提取出来
        common_colours = colours_ctr.most_common(2)
        assert len(common_colours) == 2, "Only one colour? Segment first"
        colour_one, colour_two = common_colours[0][0], common_colours[1][0]

        # 二维列表 分别用于存储 颜色是color1 和 color2的面
        out_faces: list[list[Face]] = [[], []]
        for face in mesh.faces:
            if face.colour == colour_one:
                out_faces[0].append(face)
            elif face.colour == colour_two:
                out_faces[1].append(face)
            else:
                # Randomly put into 1 or 2
                # 如果不是color1或者color2的话，就把它随机放到 列表1 或者 列表2 中
                # todo 这样不是会导致分出来的mesh不是连续的吗？？
                out_faces[random.randint(0, 1)].append(face)
        # 返回两个Mesh
        return (
            Mesh(vertices=mesh.vertices, faces=out_faces[0]),
            Mesh(vertices=mesh.vertices, faces=out_faces[1]),
        )

    def _combine_meshes(self, meshes: list[Mesh]) -> Mesh:
        faces = []
        for mesh in meshes:
            faces.extend(mesh.faces)

        return Mesh(vertices=meshes[0].vertices, faces=faces)

    # 这里按层分的代码
    def __call__(self, mesh: Mesh, dual_graph: DualGraph) -> Mesh:
        """Recursively clusterize mesh."""
        logging.info(f"Segment into {self._num_levels} levels.")
        # 记录初始的面片数
        orig_num_faces = mesh.num_faces
        # 创建了一个堆栈 将初始的网格对象mesh放入其中
        stack = [mesh]
        # 设置初始的层级数为0
        level = 0
        # while栈不为空
        while stack:
            logging.info(f"Segmenting level {level + 1}")
            colours = random_colours(num_colours=2 ** (level + 1))
            output_meshes = []
            idx = 0
            # Segment submeshes of the original mesh
            while stack:
                mesh = stack.pop()
                segmenter = BinarySegmenter(
                    num_workers=self._num_workers,
                    cluster_colors=(colours[idx * 2], colours[idx * 2 + 1]),
                )
                output_meshes.append(
                    segmenter(
                        mesh=mesh,
                        dual_graph=dual_graph,
                    )
                )
                idx += 1

            logging.info(f"Level {level + 1} segmented")
            level += 1
            # If we want to segment more
            if level != self._num_levels:
                for mesh in output_meshes:
                    # 将分割得到的子网络模型 进行divide_mesh有什么效果？
                    stack.extend(list(self._divide_mesh(mesh)))

        out_mesh = self._combine_meshes(output_meshes)
        if out_mesh.num_faces != orig_num_faces:
            raise ValueError(
                f"Incorrect resulted num faces,"
                f" expected: {orig_num_faces}, got: {out_mesh.num_faces}"
            )

        return out_mesh
