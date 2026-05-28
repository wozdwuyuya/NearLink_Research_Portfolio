# fuzzy_immune_routing.py
# 基于模糊免疫的星闪传感网路由核心算法
# 对应论文第四章：基于模糊免疫算法的通信协议可靠性优化框架

import numpy as np


class FuzzyImmuneRouter:
    """
    模糊免疫路由引擎
    核心创新：将模糊控制器嵌入免疫克隆选择的变异算子，
    利用通信门禁层的 CRC 错误率作为跨层反馈信号，动态调节变异率。
    """

    def __init__(self, population_size=50, clone_factor=3, max_stall=20):
        self.pop_size = population_size
        self.clone_factor = clone_factor
        self.max_stall = max_stall
        self.base_mutation_rate = 0.05
        self.immune_memory = set()  # 故障链路记忆库

    def fuzzy_controller(self, diversity: float, stagnant_gens: int, crc_error_rate: float) -> float:
        """
        模糊控制器：三维输入 → 自适应变异率 (对应论文 4.3.2 节)

        输入：
            diversity      - 种群多样性 [0, 1]
            stagnant_gens  - 适应度未改善的连续代数
            crc_error_rate - 通信门禁层报告的帧错误率 (跨层信号)
        输出：
            mutation_rate  - 自适应变异率 [0.01, 0.30]
        """
        mutation_rate = self.base_mutation_rate

        # 规则 1: 信道极差 → 强制大步长变异，触发免疫应答寻找新路由
        if crc_error_rate > 0.05:
            mutation_rate = 0.30

        # 规则 2: 早熟收敛（多样性低 + 停滞高）→ 提高变异率跳出局部最优
        elif stagnant_gens > 10 and diversity < 0.2:
            mutation_rate = 0.20

        # 规则 3: 中等停滞且信道有波动
        elif stagnant_gens > 5 and crc_error_rate > 0.02:
            mutation_rate = 0.15

        # 规则 4: 稳定且优质信道 → 保护优秀基因
        elif crc_error_rate < 0.01 and diversity > 0.6:
            mutation_rate = 0.01

        # 规则 5: 中等状态
        elif diversity > 0.4:
            mutation_rate = 0.05

        else:
            mutation_rate = 0.10

        return np.clip(mutation_rate, 0.01, 0.30)

    def calculate_fitness(self, remaining_energy: float, delay: float,
                          crc_pass_rate: float, trust_weight: float = 1.0) -> float:
        """
        适应度函数：融合能量、延迟、链路质量与信任度 (对应论文 5.5.2 节)

        f(R_k) = alpha * 存活率 + beta * 剩余能量 + gamma * (1/延迟) + delta * T_avg + epsilon * P_crc
        """
        alpha = 0.3   # 存活率权重
        beta = 0.3    # 剩余能量权重
        gamma = 0.15  # 延迟倒数权重
        delta = 0.15  # 信任度权重
        epsilon = 0.1 # CRC 通过率权重

        fitness = (alpha * 1.0  # 存活即得分
                   + beta * remaining_energy
                   + gamma * (1.0 / (delay + 1e-5))
                   + delta * trust_weight
                   + epsilon * crc_pass_rate)
        return fitness

    def immune_clonal_selection(self, population: list, fitness_scores: list,
                                env_status: dict) -> list:
        """
        免疫克隆选择主循环 (对应论文 4.3.3 - 4.3.4 节)

        参数：
            population     - 当前抗体群（候选路由方案列表）
            fitness_scores - 各抗体的适应度
            env_status     - 环境状态字典 {diversity, stagnant, crc_err}
        返回：
            new_population - 新一代抗体群
        """
        # 1. 模糊控制器评估当前环境，输出自适应变异率
        pm = self.fuzzy_controller(
            env_status['diversity'],
            env_status['stagnant'],
            env_status['crc_err']
        )

        # 2. 克隆选择：高适应度抗体克隆倍数大
        clones = self._clone(population, fitness_scores)

        # 3. 自适应变异：以动态变异率 pm 进行高频变异
        mutated_clones = self._mutate(clones, pm)

        # 4. 免疫应答：处理故障链路
        if env_status.get('failed_links'):
            for link in env_status['failed_links']:
                self._immune_response(link, population)

        # 5. 选择：保留 Top N 形成新一代路由表
        return self._select(population, mutated_clones)

    def _clone(self, pop: list, scores: list) -> list:
        """克隆扩增：适应度越高，克隆数越多"""
        if not pop:
            return []
        max_score = max(scores) if scores else 1.0
        clones = []
        for i, antibody in enumerate(pop):
            n_clones = max(1, round(self.clone_factor * scores[i] / (max_score + 1e-10)))
            for _ in range(n_clones):
                clones.append(antibody.copy() if isinstance(antibody, dict) else antibody)
        return clones

    def _mutate(self, clones: list, mutation_rate: float) -> list:
        """自适应变异：以模糊控制器输出的变异率执行变异"""
        mutated = []
        for clone in clones:
            if isinstance(clone, dict) and 'routing_table' in clone:
                new_clone = clone.copy()
                for node_id in new_clone['routing_table']:
                    if np.random.random() < mutation_rate:
                        # 从免疫记忆中规避已知故障链路
                        candidates = [c for c in new_clone['routing_table'][node_id].get('neighbors', [])
                                      if (node_id, c) not in self.immune_memory]
                        if candidates:
                            new_clone['routing_table'][node_id]['next_hop'] = np.random.choice(candidates)
                mutated.append(new_clone)
            else:
                mutated.append(clone)
        return mutated

    def _immune_response(self, failed_link: tuple, population: list):
        """故障链路免疫应答：标记故障 → 记忆 → 淘汰含故障链路的方案"""
        self.immune_memory.add(failed_link)
        # 清除包含故障链路的抗体
        population[:] = [ab for ab in population
                         if not (isinstance(ab, dict) and failed_link in ab.get('used_links', set()))]

    def _select(self, old_pop: list, new_pop: list) -> list:
        """精英保留选择：合并后取 Top N"""
        combined = old_pop + new_pop
        if len(combined) <= self.pop_size:
            return combined
        return combined[:self.pop_size]

    def get_mutation_rate_history(self, diversity_series: list, stagnant_series: list,
                                  crc_err_series: list) -> list:
        """返回变异率变化历史（用于论文图表生成）"""
        history = []
        for d, s, c in zip(diversity_series, stagnant_series, crc_err_series):
            history.append(self.fuzzy_controller(d, s, c))
        return history
