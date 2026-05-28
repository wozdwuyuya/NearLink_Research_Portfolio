# 文件路径: E:\1Projects\CI_FuzzyImmune_NearLink\src\simulation_env.py
import numpy as np
import math

class NearLinkNetwork:
    """星闪医疗传感网仿真环境
    基于 LEACH 经典能耗模型 + 星闪 NearLink 物理特性修正
    工作频段: 5.8 GHz ISM, TDMA 时隙调度, 1ms 级低延迟
    """

    # === 能耗模型常量 ===
    E_ELEC = 50e-9        # 电路能耗: 50 nJ/bit
    EPS_FS = 10e-12       # 自由空间功率放大系数: 10 pJ/bit/m^2
    EPS_MP = 0.0013e-12   # 多径衰落功率放大系数: 0.0013 pJ/bit/m^4
    D_0 = 87.7            # 交叉距离: sqrt(EPS_FS / EPS_MP) ≈ 87.7 m
    K_BITS = 512          # 数据包大小: 512 bits (医疗心跳数据包)
    E_DA = 5e-9           # 数据融合能耗: 5 nJ/bit
    T_SLOT = 1e-3         # TDMA 时隙: 1 ms
    F_SAMPLE = 100        # 星闪采样率: 100 Hz (高频心跳)

    # === 适应度权重 ===
    W_ENERGY = 0.5        # 能耗权重
    W_SURVIVAL = 0.3      # 存活率权重
    W_DELAY = 0.2         # 延迟权重

    def __init__(self, num_nodes=100, area_size=100, initial_energy=0.5):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.initial_energy = initial_energy
        # 随机生成节点坐标 (x, y)
        self.nodes_pos = np.random.rand(num_nodes, 2) * area_size
        self.nodes_energy = np.ones(num_nodes) * self.initial_energy
        # 基站位于中心
        self.base_station = np.array([area_size / 2.0, area_size / 2.0])
        # 每轮每节点数据量 = 采样率 * 数据包大小
        self.bits_per_round = self.F_SAMPLE * self.K_BITS

    def reset(self):
        """重置网络状态（用于多轮实验）"""
        self.nodes_energy = np.ones(self.num_nodes) * self.initial_energy

    def _calc_tx_energy(self, distance, bits):
        """计算发射能耗: 自由空间 / 多径衰落双模型"""
        d_sq = distance ** 2
        if distance < self.D_0:
            # 自由空间模型: E_tx = k*E_elec + k*eps_fs*d^2
            return bits * self.E_ELEC + bits * self.EPS_FS * d_sq
        else:
            # 多径衰落模型: E_tx = k*E_elec + k*eps_mp*d^4
            return bits * self.E_ELEC + bits * self.EPS_MP * d_sq * d_sq

    def _calc_rx_energy(self, bits):
        """计算接收能耗"""
        return bits * self.E_ELEC

    def _calc_ch_energy(self, ch_idx, cluster_members, d_to_bs):
        """计算簇头总能耗: 接收成员数据 + 数据融合 + 转发到基站"""
        n_members = len(cluster_members)
        bits = self.bits_per_round
        # 接收所有成员数据
        e_rx = n_members * self._calc_rx_energy(bits)
        # 数据融合
        e_da = bits * self.E_DA * n_members
        # 转发到基站
        e_tx = self._calc_tx_energy(d_to_bs, bits)
        return e_rx + e_da + e_tx

    def _select_cluster_heads(self, antibody, p_optimal=0.05):
        """基于抗体编码的 Top-K 簇头选举
        抗体值越大，成为簇头的优先级越高
        """
        k = max(1, round(self.num_nodes * p_optimal))
        ch_indices = np.argsort(antibody)[-k:]
        return ch_indices

    def _assign_members(self, ch_indices):
        """将每个非簇头节点分配到最近的簇头"""
        ch_pos = self.nodes_pos[ch_indices]
        # 计算所有节点到所有簇头的距离矩阵
        dists = np.linalg.norm(
            self.nodes_pos[:, np.newaxis, :] - ch_pos[np.newaxis, :, :],
            axis=2
        )  # shape: (num_nodes, num_ch)
        nearest_ch_local = np.argmin(dists, axis=1)  # 最近簇头的局部索引
        nearest_ch_global = ch_indices[nearest_ch_local]  # 映射回全局索引
        return nearest_ch_global

    def simulate_round(self, antibody, p_optimal=0.05, consume_energy=True):
        """模拟一轮数据传输，返回能耗、延迟、存活数
        consume_energy=True: 实际消耗能量 (用于主循环)
        consume_energy=False: 干跑模式 (用于适应度评估，不修改环境状态)
        """
        # 1. 选举簇头
        ch_indices = self._select_cluster_heads(antibody, p_optimal)
        num_ch = len(ch_indices)

        # 2. 分配成员
        member_assignment = self._assign_members(ch_indices)

        # 3. 计算各节点能耗
        energy_consumed = np.zeros(self.num_nodes)
        delay_per_ch = np.zeros(num_ch)

        for ch_local_idx, ch_global_idx in enumerate(ch_indices):
            # 找到属于该簇头的成员
            members = np.where(member_assignment == ch_global_idx)[0]
            # 簇头到基站距离
            d_to_bs = np.linalg.norm(
                self.nodes_pos[ch_global_idx] - self.base_station
            )
            # 簇头能耗
            energy_consumed[ch_global_idx] = self._calc_ch_energy(
                ch_global_idx, members, d_to_bs
            )
            # 端到端延迟 = (成员数 + 1) * 时隙
            delay_per_ch[ch_local_idx] = (len(members) + 1) * self.T_SLOT

            # 非簇头成员: 发射数据到簇头
            for member_idx in members:
                if member_idx != ch_global_idx:
                    d_member_to_ch = np.linalg.norm(
                        self.nodes_pos[member_idx] - self.nodes_pos[ch_global_idx]
                    )
                    energy_consumed[member_idx] = self._calc_tx_energy(
                        d_member_to_ch, self.bits_per_round
                    )

        # 4. 条件更新节点能量
        if consume_energy:
            self.nodes_energy -= energy_consumed
            self.nodes_energy = np.maximum(self.nodes_energy, 0.0)

        # 5. 统计指标
        total_energy = np.sum(energy_consumed)
        alive_count = np.sum(self.nodes_energy > 0)
        avg_delay = np.mean(delay_per_ch) if num_ch > 0 else 0.0

        return total_energy, avg_delay, alive_count

    def calculate_fitness(self, antibody, p_optimal=0.05):
        """综合适应度函数 (干跑模式，不消耗能量)
        fitness = w1*e_norm + w2*alive_rate + w3*d_norm
        """
        eps = 1e-10
        total_energy, avg_delay, alive_count = self.simulate_round(
            antibody, p_optimal, consume_energy=False
        )
        alive_rate = alive_count / self.num_nodes

        # 归一化各项到 [0, 1] 区间
        e_max = self.num_nodes * self.initial_energy
        e_norm = 1.0 - min(total_energy / (e_max + eps), 1.0)

        d_max = (self.num_nodes + 1) * self.T_SLOT
        d_norm = 1.0 - min(avg_delay / (d_max + eps), 1.0)

        s_norm = alive_rate

        fitness = (self.W_ENERGY * e_norm
                   + self.W_SURVIVAL * s_norm
                   + self.W_DELAY * d_norm)

        return fitness

    def get_alive_count(self):
        """返回当前存活节点数"""
        return int(np.sum(self.nodes_energy > 0))

    def get_total_remaining_energy(self):
        """返回全网剩余总能量"""
        return float(np.sum(self.nodes_energy))
