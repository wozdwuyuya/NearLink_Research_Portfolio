# 文件路径: E:\1Projects\CI_FuzzyImmune_NearLink\src\core_algorithm.py
import numpy as np
import random

class FuzzyImmuneAlgorithm:
    """模糊免疫算法: 模糊控制器动态变异率 + 免疫克隆选择"""

    def __init__(self, pop_size=50, max_iter=100, node_count=100,
                 clone_factor=3, p_optimal=0.05, max_stall=20):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.node_count = node_count
        self.clone_factor = clone_factor   # 每个优秀抗体的克隆数
        self.p_optimal = p_optimal         # 最优簇头比例
        self.max_stall = max_stall         # 最大停滞代数阈值

    def initialize_population(self):
        """初始化抗体群: 每个抗体 ∈ [0,1]^D 代表节点的簇头优先级"""
        return np.random.rand(self.pop_size, self.node_count)

    # ========== 多样性计算 (Diversity) ==========

    def calculate_diversity(self, population):
        """混合多样性: 0.5*成对欧氏距离 + 0.5*基因位标准差
        域: [0, 1], 0=完全收敛, 1=完全分散
        """
        n, d = population.shape

        # 方案A: 平均成对欧氏距离 (归一化)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(population[i] - population[j])
                count += 1
        max_pair_dist = np.sqrt(d)  # [0,1]^D 空间最大欧氏距离
        d_pair = (total_dist / (count * max_pair_dist)) if count > 0 else 0.0

        # 方案C: 基因位标准差均值 (归一化)
        gene_std = np.std(population, axis=0)  # 每个基因位的标准差
        d_gene = np.mean(gene_std) / 0.5       # 均匀分布 std≈0.5, 归一化到~1

        diversity = 0.5 * d_pair + 0.5 * min(d_gene, 1.0)
        return diversity

    # ========== 模糊控制器 (Fuzzy Controller) ==========

    def _trimf(self, x, params):
        """三角隶属函数 (Triangular Membership Function)
        params = [a, b, c] where b is the peak
        """
        a, b, c = params
        if x < a or x > c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if b > a else 0.0
        else:
            return (c - x) / (c - b) if c > b else 0.0

    def fuzzy_controller(self, diversity, stall_generations):
        """模糊逻辑控制器: 动态输出变异率
        输入: diversity ∈ [0,1], stall_generations (整数)
        输出: mutation_rate ∈ [0.01, 0.30]
        """
        # 归一化停滞代数到 [0, 1]
        s_norm = min(stall_generations / self.max_stall, 1.0)
        d = diversity

        # === 隶属函数参数 ===
        # Diversity
        d_low  = [0.0, 0.0, 0.3]
        d_med  = [0.2, 0.5, 0.8]
        d_high = [0.7, 1.0, 1.0]
        # Stall
        s_low  = [0.0, 0.0, 0.3]
        s_med  = [0.2, 0.5, 0.8]
        s_high = [0.7, 1.0, 1.0]
        # Mutation Rate (输出隶属函数中心值, 归一化)
        m_low_center  = 0.175   # 对应 [0, 0, 0.35] 的中心
        m_med_center  = 0.50    # 对应 [0.25, 0.5, 0.75] 的中心
        m_high_center = 0.825   # 对应 [0.65, 1.0, 1.0] 的中心

        # === 计算输入隶属度 ===
        mu_d_low  = self._trimf(d, d_low)
        mu_d_med  = self._trimf(d, d_med)
        mu_d_high = self._trimf(d, d_high)
        mu_s_low  = self._trimf(s_norm, s_low)
        mu_s_med  = self._trimf(s_norm, s_med)
        mu_s_high = self._trimf(s_norm, s_high)

        # === 9 条模糊规则 (IF-THEN) ===
        # 规则格式: (D隶属, S隶属, 输出中心值)
        # 取 min 作为激活强度 (AND 操作)
        rules = [
            # R1: D=Low AND S=Low   -> M=High
            (min(mu_d_low, mu_s_low), m_high_center),
            # R2: D=Low AND S=Med   -> M=High
            (min(mu_d_low, mu_s_med), m_high_center),
            # R3: D=Low AND S=High  -> M=High
            (min(mu_d_low, mu_s_high), m_high_center),
            # R4: D=Med AND S=Low   -> M=Low
            (min(mu_d_med, mu_s_low), m_low_center),
            # R5: D=Med AND S=Med   -> M=Med
            (min(mu_d_med, mu_s_med), m_med_center),
            # R6: D=Med AND S=High  -> M=High
            (min(mu_d_med, mu_s_high), m_high_center),
            # R7: D=High AND S=Low  -> M=Low
            (min(mu_d_high, mu_s_low), m_low_center),
            # R8: D=High AND S=Med  -> M=Low
            (min(mu_d_high, mu_s_med), m_low_center),
            # R9: D=High AND S=High -> M=Med
            (min(mu_d_high, mu_s_high), m_med_center),
        ]

        # === 重心法去模糊化 (Centroid Defuzzification) ===
        numerator = sum(strength * center for strength, center in rules)
        denominator = sum(strength for strength, _ in rules)

        if denominator < 1e-10:
            m_output = 0.5  # 默认中等变异率
        else:
            m_output = numerator / denominator

        # 映射回实际变异率: [0,1] -> [0.01, 0.30]
        mutation_rate = 0.01 + m_output * 0.29
        return mutation_rate

    # ========== 免疫算子 ==========

    def mutate(self, antibody, mutation_rate):
        """变异算子: 逐位以 mutation_rate 概率随机重置"""
        mutated = np.copy(antibody)
        mask = np.random.rand(len(mutated)) < mutation_rate
        mutated[mask] = np.random.rand(np.sum(mask))
        return mutated

    def clone_and_hypermutate(self, population, fitness_values, mutation_rate):
        """克隆选择 + 超变异
        1. 选择亲和度 Top-M 抗体
        2. 每个克隆 clone_factor 份
        3. 超变异率 = mutation_rate * 2 (克隆体变异率更高)
        4. 贪心选择: 每个原抗体的克隆体中选最优
        """
        n, d = population.shape
        m = max(2, n // 2)  # 选择前 50% 优秀抗体

        # 排序选 Top-M
        sorted_indices = np.argsort(fitness_values)[::-1]  # 降序
        top_indices = sorted_indices[:m]

        new_population = []
        new_fitness = []

        for idx in top_indices:
            parent = population[idx]
            parent_fit = fitness_values[idx]

            # 克隆
            n_clones = self.clone_factor
            clones = np.tile(parent, (n_clones, 1))

            # 超变异 (变异率为普通变异的 2 倍, 上限 0.3)
            hyper_rate = min(mutation_rate * 2.0, 0.3)
            for c in range(n_clones):
                clones[c] = self.mutate(clones[c], hyper_rate)

            # 贪心选择: 克隆体和父代中选最优 (需外部评估适应度)
            new_population.append(parent)  # 先保留父代
            for c in range(n_clones):
                new_population.append(clones[c])

        # 截断到种群大小
        new_population = np.array(new_population[:self.pop_size])
        return new_population

    def select_survivors(self, population, fitness_values):
        """精英保留 + 锦标赛选择: 保留最优个体到下一代"""
        # 按适应度降序排序
        sorted_indices = np.argsort(fitness_values)[::-1]
        return population[sorted_indices[:self.pop_size]]

    # ========== 主循环 (供 main.py 调用) ==========

    def run(self, env, verbose=False):
        """完整迭代循环
        返回: best_fitness_history, alive_history, energy_history
        """
        population = self.initialize_population()
        stall_count = 0
        best_fitness = -1.0

        best_fitness_history = []
        alive_history = []
        energy_history = []

        for gen in range(self.max_iter):
            # 1. 评估适应度 (需要克隆体也评估，这里简化为只评估当前种群)
            fitness_values = np.array([
                env.calculate_fitness(ind, self.p_optimal)
                for ind in population
            ])

            # 2. 记录本代最优
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                stall_count = 0
            else:
                stall_count += 1

            # 3. 计算多样性
            diversity = self.calculate_diversity(population)

            # 4. 模糊控制器输出变异率
            mutation_rate = self.fuzzy_controller(diversity, stall_count)

            # 5. 克隆 + 超变异
            population = self.clone_and_hypermutate(
                population, fitness_values, mutation_rate
            )

            # 6. 变异后的种群重新评估并选择
            fitness_values_new = np.array([
                env.calculate_fitness(ind, self.p_optimal)
                for ind in population
            ])
            population = self.select_survivors(population, fitness_values_new)

            # 7. 最优抗体实际执行一轮 (消耗能量)
            best_antibody = population[np.argmax(fitness_values_new)]
            env.simulate_round(best_antibody, self.p_optimal, consume_energy=True)

            # 8. 记录指标
            best_fitness_history.append(best_fitness)
            alive_history.append(env.get_alive_count())
            energy_history.append(env.get_total_remaining_energy())

            if verbose and (gen % 10 == 0 or gen == self.max_iter - 1):
                print(f"Gen {gen:3d} | Best={best_fitness:.4f} | "
                      f"Div={diversity:.3f} | MutRate={mutation_rate:.3f} | "
                      f"Alive={env.get_alive_count()}/{env.num_nodes} | "
                      f"Stall={stall_count}")

        return best_fitness_history, alive_history, energy_history


class StandardImmuneAlgorithm:
    """标准免疫算法 (对照组): 固定变异率, 无模糊控制器"""

    def __init__(self, pop_size=50, max_iter=100, node_count=100,
                 clone_factor=3, p_optimal=0.05, mutation_rate=0.05):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.node_count = node_count
        self.clone_factor = clone_factor
        self.p_optimal = p_optimal
        self.fixed_mutation_rate = mutation_rate

    def initialize_population(self):
        return np.random.rand(self.pop_size, self.node_count)

    def mutate(self, antibody, mutation_rate):
        mutated = np.copy(antibody)
        mask = np.random.rand(len(mutated)) < mutation_rate
        mutated[mask] = np.random.rand(np.sum(mask))
        return mutated

    def clone_and_hypermutate(self, population, fitness_values, mutation_rate):
        n, d = population.shape
        m = max(2, n // 2)
        sorted_indices = np.argsort(fitness_values)[::-1]
        top_indices = sorted_indices[:m]

        new_population = []
        for idx in top_indices:
            parent = population[idx]
            n_clones = self.clone_factor
            clones = np.tile(parent, (n_clones, 1))
            hyper_rate = min(mutation_rate * 2.0, 0.3)
            for c in range(n_clones):
                clones[c] = self.mutate(clones[c], hyper_rate)
            new_population.append(parent)
            for c in range(n_clones):
                new_population.append(clones[c])

        new_population = np.array(new_population[:self.pop_size])
        return new_population

    def select_survivors(self, population, fitness_values):
        sorted_indices = np.argsort(fitness_values)[::-1]
        return population[sorted_indices[:self.pop_size]]

    def run(self, env, verbose=False):
        population = self.initialize_population()
        best_fitness = -1.0

        best_fitness_history = []
        alive_history = []
        energy_history = []

        for gen in range(self.max_iter):
            fitness_values = np.array([
                env.calculate_fitness(ind, self.p_optimal)
                for ind in population
            ])

            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness

            # 标准免疫: 固定变异率
            population = self.clone_and_hypermutate(
                population, fitness_values, self.fixed_mutation_rate
            )

            fitness_values_new = np.array([
                env.calculate_fitness(ind, self.p_optimal)
                for ind in population
            ])
            population = self.select_survivors(population, fitness_values_new)

            # 最优抗体实际执行一轮 (消耗能量)
            best_antibody = population[np.argmax(fitness_values_new)]
            env.simulate_round(best_antibody, self.p_optimal, consume_energy=True)

            best_fitness_history.append(best_fitness)
            alive_history.append(env.get_alive_count())
            energy_history.append(env.get_total_remaining_energy())

            if verbose and (gen % 10 == 0 or gen == self.max_iter - 1):
                print(f"[Standard] Gen {gen:3d} | Best={best_fitness:.4f} | "
                      f"MutRate={self.fixed_mutation_rate:.3f} | "
                      f"Alive={env.get_alive_count()}/{env.num_nodes}")

        return best_fitness_history, alive_history, energy_history


class LEACHAlgorithm:
    """LEACH 协议 (基准组): 每节点以固定概率 p 随机成为簇头"""

    def __init__(self, max_iter=100, node_count=100, p=0.05):
        self.max_iter = max_iter
        self.node_count = node_count
        self.p = p

    def run(self, env, verbose=False):
        best_fitness_history = []
        alive_history = []
        energy_history = []
        best_fitness = -1.0

        for gen in range(self.max_iter):
            # LEACH: 每节点以概率 p 随机成为簇头
            antibody = np.random.rand(self.node_count)
            fitness = env.calculate_fitness(antibody, self.p)

            if fitness > best_fitness:
                best_fitness = fitness

            # 实际执行一轮 (消耗能量)
            env.simulate_round(antibody, self.p, consume_energy=True)

            best_fitness_history.append(best_fitness)
            alive_history.append(env.get_alive_count())
            energy_history.append(env.get_total_remaining_energy())

            if verbose and (gen % 10 == 0 or gen == self.max_iter - 1):
                print(f"[LEACH] Gen {gen:3d} | Best={best_fitness:.4f} | "
                      f"Alive={env.get_alive_count()}/{env.num_nodes}")

        return best_fitness_history, alive_history, energy_history
