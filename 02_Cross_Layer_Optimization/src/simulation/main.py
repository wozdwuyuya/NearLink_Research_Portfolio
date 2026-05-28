# 文件路径: E:\1Projects\CI_FuzzyImmune_NearLink\src\main.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 将 src 目录加入 path，确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_env import NearLinkNetwork
from core_algorithm import FuzzyImmuneAlgorithm, StandardImmuneAlgorithm, LEACHAlgorithm

# === 实验常量 ===
SEED = 42
NUM_NODES = 100
AREA_SIZE = 100
INITIAL_ENERGY = 0.5
POP_SIZE = 50
MAX_ITER = 50
P_OPTIMAL = 0.05
MAX_STALL = 20
CLONE_FACTOR = 3

# === 可视化配置 ===
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def run_fuzzy_immune(seed, env_template):
    """运行模糊免疫算法"""
    np.random.seed(seed)
    env = NearLinkNetwork(num_nodes=NUM_NODES, area_size=AREA_SIZE,
                          initial_energy=INITIAL_ENERGY)
    env.nodes_pos = env_template.nodes_pos.copy()
    env.nodes_energy = env_template.nodes_energy.copy()

    algo = FuzzyImmuneAlgorithm(
        pop_size=POP_SIZE, max_iter=MAX_ITER, node_count=NUM_NODES,
        clone_factor=CLONE_FACTOR, p_optimal=P_OPTIMAL, max_stall=MAX_STALL
    )
    h_fitness, h_alive, h_energy = algo.run(env, verbose=True)
    return h_fitness, h_alive, h_energy


def run_standard_immune(seed, env_template):
    """运行标准免疫算法"""
    np.random.seed(seed)
    env = NearLinkNetwork(num_nodes=NUM_NODES, area_size=AREA_SIZE,
                          initial_energy=INITIAL_ENERGY)
    env.nodes_pos = env_template.nodes_pos.copy()
    env.nodes_energy = env_template.nodes_energy.copy()

    algo = StandardImmuneAlgorithm(
        pop_size=POP_SIZE, max_iter=MAX_ITER, node_count=NUM_NODES,
        clone_factor=CLONE_FACTOR, p_optimal=P_OPTIMAL, mutation_rate=0.05
    )
    h_fitness, h_alive, h_energy = algo.run(env, verbose=True)
    return h_fitness, h_alive, h_energy


def run_leach(seed, env_template):
    """运行 LEACH 协议"""
    np.random.seed(seed)
    env = NearLinkNetwork(num_nodes=NUM_NODES, area_size=AREA_SIZE,
                          initial_energy=INITIAL_ENERGY)
    env.nodes_pos = env_template.nodes_pos.copy()
    env.nodes_energy = env_template.nodes_energy.copy()

    algo = LEACHAlgorithm(max_iter=MAX_ITER, node_count=NUM_NODES, p=P_OPTIMAL)
    h_fitness, h_alive, h_energy = algo.run(env, verbose=True)
    return h_fitness, h_alive, h_energy


def plot_fitness_convergence(fuzzy_h, standard_h, leach_h):
    """图表 A: 适应度收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    gens = range(len(fuzzy_h))

    ax.plot(gens, fuzzy_h, 'r-', linewidth=2, label='Fuzzy Immune (Ours)')
    ax.plot(gens, standard_h, 'b--', linewidth=1.5, label='Standard Immune')
    ax.plot(gens, leach_h, 'g:', linewidth=1.5, label='LEACH')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Fitness Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(fuzzy_h) - 1)

    path = os.path.join(DATA_DIR, 'fitness_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_alive_nodes(fuzzy_h, standard_h, leach_h):
    """图表 B: 网络生存周期阶梯图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    gens = range(len(fuzzy_h))

    ax.step(gens, fuzzy_h, 'r-', linewidth=2, label='Fuzzy Immune (Ours)', where='post')
    ax.step(gens, standard_h, 'b--', linewidth=1.5, label='Standard Immune', where='post')
    ax.step(gens, leach_h, 'g:', linewidth=1.5, label='LEACH', where='post')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Alive Nodes', fontsize=12)
    ax.set_title('Network Lifetime: Alive Nodes Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(fuzzy_h) - 1)
    ax.set_ylim(0, NUM_NODES + 5)

    path = os.path.join(DATA_DIR, 'alive_nodes_staircase.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_energy_consumption(fuzzy_h, standard_h, leach_h):
    """图表 C: 全网剩余能量衰减曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    gens = range(len(fuzzy_h))

    ax.plot(gens, fuzzy_h, 'r-', linewidth=2, label='Fuzzy Immune (Ours)')
    ax.plot(gens, standard_h, 'b--', linewidth=1.5, label='Standard Immune')
    ax.plot(gens, leach_h, 'g:', linewidth=1.5, label='LEACH')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Total Remaining Energy (J)', fontsize=12)
    ax.set_title('Network Energy Consumption Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(fuzzy_h) - 1)

    path = os.path.join(DATA_DIR, 'energy_consumption.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


def main():
    print("=" * 60)
    print(" NearLink Fuzzy Immune Routing - Comparison Experiment")
    print("=" * 60)
    print(f" Nodes: {NUM_NODES} | Area: {AREA_SIZE}x{AREA_SIZE} | "
          f"Pop: {POP_SIZE} | Iter: {MAX_ITER} | Seed: {SEED}")
    print(f" Initial Energy: {INITIAL_ENERGY}J/node | "
          f"Total: {NUM_NODES * INITIAL_ENERGY}J")
    print("=" * 60)

    # 创建统一的网络模板 (相同拓扑)
    np.random.seed(SEED)
    env_template = NearLinkNetwork(num_nodes=NUM_NODES, area_size=AREA_SIZE,
                                   initial_energy=INITIAL_ENERGY)

    # === 运行三种算法 ===
    print("\n[1/3] Running Fuzzy Immune Algorithm...")
    fuzzy_fit, fuzzy_alive, fuzzy_energy = run_fuzzy_immune(SEED, env_template)

    print("\n[2/3] Running Standard Immune Algorithm...")
    std_fit, std_alive, std_energy = run_standard_immune(SEED, env_template)

    print("\n[3/3] Running LEACH Protocol...")
    leach_fit, leach_alive, leach_energy = run_leach(SEED, env_template)

    # === 输出最终统计 ===
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(f" {'Algorithm':<20} {'Best Fitness':>12} {'Alive Nodes':>12} {'Remaining E(J)':>14}")
    print(f" {'-'*20} {'-'*12} {'-'*12} {'-'*14}")
    print(f" {'Fuzzy Immune':<20} {fuzzy_fit[-1]:>12.4f} {fuzzy_alive[-1]:>12} {fuzzy_energy[-1]:>14.4f}")
    print(f" {'Standard Immune':<20} {std_fit[-1]:>12.4f} {std_alive[-1]:>12} {std_energy[-1]:>14.4f}")
    print(f" {'LEACH':<20} {leach_fit[-1]:>12.4f} {leach_alive[-1]:>12} {leach_energy[-1]:>14.4f}")
    print("=" * 60)

    # === 生成图表 ===
    print("\nGenerating charts...")
    plot_fitness_convergence(fuzzy_fit, std_fit, leach_fit)
    plot_alive_nodes(fuzzy_alive, std_alive, leach_alive)
    plot_energy_consumption(fuzzy_energy, std_energy, leach_energy)

    print("\nAll charts saved to: data/")
    print("Experiment complete.")


if __name__ == "__main__":
    main()
