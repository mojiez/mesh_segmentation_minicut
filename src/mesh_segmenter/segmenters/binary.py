import logging
from copy import deepcopy
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import tqdm

from ..graph import DualGraph
from ..utils.mesh import Mesh, Face
from ..utils.constants import (
    MAX_NUM_ITERS,
    COLOUR_BLUE,
    COLOUR_RED,
)
from ..utils.colour import Colour


class BinarySegmenter:
    """Segments a mesh into the 2 segments."""

    def __init__(
        self,
        num_workers: int = multiprocessing.cpu_count(),
        num_iters: int = MAX_NUM_ITERS,
        # 默认是0.5
        prob_threshold: float = 0.5,
        cluster_colors: tuple[Colour, Colour] = (COLOUR_BLUE, COLOUR_RED),
    ):
        self._num_workers = num_workers
        self._num_iters = num_iters
        self._cluster_colors = cluster_colors
        self._color_unsure = sum(cluster_colors)
        self._prob_threshold = prob_threshold

    # 最开始 寻找相距最远的种子面片的方法
    # 返回值是一个列表 包含两个面对象？？ list[Face] 表示一个列表 其中每个元素都是Face类型的对象
    def _init_reprs(self, mesh: Mesh, dual_graph: DualGraph) -> list[Face]:
        # For binary case
        # Choose a pair of nodes with highest distances
        max_dist = 0
        # mesh中的第一个面， 一开始把两个面都设置成第一个面
        repr = [mesh.faces[0], mesh.faces[0]]

        # faces是一个list[Face] 每一个元素是一个Face对象
        # enumerate() 函数用于遍历一个可迭代对象，并返回每个元素的索引以及对应的元素值。
        # i是当前遍历到的面片的索引， face_one是当前遍历到的面片
        for i, face_one in enumerate(mesh.faces):
            # j就是遍历当前面片之后的面片
            for j in range(i + 1, mesh.num_faces):
                # 如果说计算出来的最大距离比之前的最大距离都大 那么就记录这个最大距离
                if (
                    dist := dual_graph.get_distance(face_one, mesh.faces[j])
                ) > max_dist:
                    max_dist = dist
                    repr = [face_one, mesh.faces[j]]
        # 返回最大距离
        # repr 记录的是相距最大的两个面片
        return repr

    def _update_probs(
        self,
        reprs: list[Face],
        probs: dict[Face, list[float]],
        dual_graph: DualGraph,
        mesh: Mesh,
    ) -> None:
        """Updates in-place probabilities of belongings to the REPa or REPb."""

        # 接收一个面片参数 返回一个tuple元组
        def calculate_probs(face: Face) -> tuple[Face, list[float]]:
            # If distance is closer to other repr - probability of beloning lower
            # Update and normalize
            # 计算当前面片属于 种子面片0 的距离
            prob_zero = dual_graph.get_distance(face, reprs[1])
            prob_zero /= dual_graph.get_distance(
                face, reprs[0]
            ) + dual_graph.get_distance(face, reprs[1])
            # 计算当前面片属于 种子面片1 的距离
            prob_one = dual_graph.get_distance(face, reprs[0])
            prob_one /= dual_graph.get_distance(
                face, reprs[0]
            ) + dual_graph.get_distance(face, reprs[1])
            # 返回一个元组 记录这个面片 和 它属于种子面片 0 1 的概率
            return face, [prob_zero, prob_one]
        # result用于存储计算概率的结果
        results = []
        # 遍历网格中的每一个face
        for face in mesh.faces:
            # 将计算的结果加入result中
            results.append(calculate_probs(face))

        for face, prob_list in results:
            # probs：一个hashMap face 对应 它属于两个种子面片的概率
            probs[face] = prob_list

    def _prob_dist_sum(
        self,
        face: Face,
        probs: dict[Face, list[float]],
        cluster_idx: int,
        mesh: Mesh,
        dual_graph: DualGraph,
    ) -> float:
        out_sum = 0.0
        for face_cur in mesh.faces:
            out_sum += probs[face_cur][cluster_idx] * dual_graph.get_distance(
                face_cur, face
            )

        return out_sum

    # 重新计算种子面片
    def _update_reprs(
        self,
        reprs: list[Face],
        probs: dict[Face, list[float]],
        mesh: Mesh,
        dual_graph: DualGraph,
    ) -> list[Face]:
        def calculate_sums(face: Face) -> tuple[Face, float, float]:
            pa_dist_sum = self._prob_dist_sum(
                face=face,
                probs=probs,
                cluster_idx=0,
                mesh=mesh,
                dual_graph=dual_graph,
            )
            pb_dist_sum = self._prob_dist_sum(
                face=face,
                probs=probs,
                cluster_idx=1,
                mesh=mesh,
                dual_graph=dual_graph,
            )
            return (face, pa_dist_sum, pb_dist_sum)

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            p_dist_sums = list(tqdm.tqdm(
                executor.map(calculate_sums, mesh.faces),
                total=mesh.num_faces,
            ))

        min_prob_a_dist = float("inf")
        min_prob_b_dist = float("inf")
        for face, pa_dist_sum, pb_dist_sum in p_dist_sums:
            # Choose a new repr set
            if pa_dist_sum < min_prob_a_dist:
                min_prob_a_dist = pa_dist_sum
                reprs = [face, reprs[1]]

            if pb_dist_sum < min_prob_b_dist:
                min_prob_b_dist = min_prob_b_dist
                reprs = [reprs[0], face]

        return reprs

    # 形成聚类：返回一个字典：键是面对象 值是成员资格的概率列表
    def _form_clusters(
        self, mesh: Mesh, dual_graph: DualGraph
    ) -> dict[Face, list[float]]:
        """Form segmentation clusters, output probabilities of memberships."""
        # 输出日志 表示正在形成初始的粗略聚类
        logging.info("Forming initial coarse clusters.")
        # Initial cluster centers, the 2 most further faces
        # 选取相距最远的两个面片作为种子面片（聚类中心）
        reprs: list[Face] = self._init_reprs(mesh=mesh, dual_graph=dual_graph)
        # 打印日志：已经找到最开始的两个种子面片
        logging.info("Initial clusters were formed.")
        # 断言有两个面片 否则异常
        assert len(reprs) == 2, "Binary segmentation."

        # 打印日志：迭代更新 获得模糊分割
        logging.info("Iteratively update memberships, get fuzzy decompose.")

        # 创建一个字典： 键是mesh里面的每一个面片face  值是初始化为[0,0]的一个列表
        probs = {face: [0.0, 0.0] for face in mesh.faces}
        # Iteratively update list of probabilities of belonging in clusters
        for _ in tqdm.trange(self._num_iters):
            # 执行num_iters次
            # 取出 两个 种子面片
            cur_rep_a, cur_rep_b = reprs

            # 调用了 更新概率 的函数
            self._update_probs(
                reprs=reprs,
                probs=probs,
                mesh=mesh,
                dual_graph=dual_graph,
            )

            # 更新种子面片
            reprs = self._update_reprs(
                reprs=reprs,
                probs=probs,
                mesh=mesh,
                dual_graph=dual_graph,
            )
            # If no updates
            # 如果没有更新了 那么就确定下种子面片 还有每个面片属于种子面片的概率
            if cur_rep_a == reprs[0] and cur_rep_b == reprs[1]:
                break

        # 如果说迭代次数都迭代完了 or 没有更新了 打印日志 返回结果
        logging.info(
            "Fuzzy segmentation memberships updated, get fuzzy decompose."
        )
        # probs：一个hashMap face 对应 它属于两个种子面片的概率
        return probs
    # 给已经分割好的 和 没有分割好的部分都上色
    def _update_segment_colours(
        self, mesh: Mesh, probs: dict[Face, list[float]]
    ) -> None:
        """Update colours according to probabilities."""
        logging.info("Updating face segment colours")
        for face in mesh.faces:
            # threshold 默认是 0.5 那么只有当等于0.5的时候 才会出现边界模糊的情况！！
            if probs[face][0] > self._prob_threshold:
                face.set_colour(self._cluster_colors[0])
            elif probs[face][1] > self._prob_threshold:
                face.set_colour(self._cluster_colors[1])
            else:
                face.set_colour(self._color_unsure)

        # TODO: fuzziness on borders
        # TODO 这里没有实现最小割...
        logging.info("Segment colours update completed.")

    # call方法是python中的一个特殊方法，它允许对象实例像函数一样被调用！
    #  返回一个Mesh类型的对象
    def __call__(self, mesh: Mesh, dual_graph: DualGraph) -> Mesh:
        """Segmented mesh with coloured seg  ments."""
        # 深拷贝了原始的参数， 避免在方法中修改原始对象
        mesh, dual_graph = deepcopy(mesh), deepcopy(dual_graph)
        # 得到初始种子面片 计算概率 迭代 确定种子面片 和 每个面片属于种子面片的概率
        # probs：一个hashMap face 对应 它属于两个种子面片的概率
        probs = self._form_clusters(mesh=mesh, dual_graph=dual_graph)

        self._update_segment_colours(mesh=mesh, probs=probs)

        return mesh
