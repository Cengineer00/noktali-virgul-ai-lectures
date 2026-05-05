"""
bandit.py
──────────────────────────────────────────────────────────────────────────────
Multi-run experiment runner + static plots + animated dashboard.

Bandit logic lives in bandit_core.py; this file only handles
matplotlib visualisations and the experiment entry point.

Usage:
    python bandit.py               # static plots then live animation window
    python bandit.py --save out.mp4
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from bandit_core import BanditEnv, Agent, run_experiment, presimulate


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRUE_MEANS  = [1.0, 1.2, 0.8]
STEPS       = 1000
RUNS        = 500
EPSILON     = 0.1
REWARD_STD  = 1.0
BASE_SEED   = 42

# Animation-specific
ANIM_STEPS  = 500
ANIM_SEED   = 42
PALETTE     = ["#58a6ff", "#f78166", "#3fb950"]   # blue / red / green


# ══════════════════════════════════════════════════════════════════════════════
# STATIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(avg_cumulative_rewards, optimal_action_rate, true_means, epsilon):
    """Two-panel static comparison plot (multi-run averaged results)."""
    steps = len(next(iter(avg_cumulative_rewards.values())))
    x     = np.arange(1, steps + 1)

    plt.figure(figsize=(10, 6))
    for name, values in avg_cumulative_rewards.items():
        plt.plot(x, values, linewidth=2, label=name)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title(
        f"Cumulative Average Reward Comparison\n"
        f"True Means = {true_means}, epsilon = {epsilon}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for name, values in optimal_action_rate.items():
        plt.plot(x, values, linewidth=2, label=name)
    plt.xlabel("Step")
    plt.ylabel("Optimal Action Rate")
    plt.title("How Often Each Agent Chooses the Best Arm")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def animate_experiment(
    true_means,
    steps=500,
    epsilon=0.1,
    reward_std=1.0,
    seed=42,
    interval=25,
    save_path=None,
):
    """
    Side-by-side animated dashboard for all three strategies.
    Top row  : Q-value bar charts (chosen arm highlighted each step)
    Bottom   : Growing optimal-action-rate and cumulative-reward curves
    """
    k            = len(true_means)
    optimal_arm  = int(np.argmax(true_means))
    strategy_map = {
        "Random":         dict(strategy="random"),
        "Greedy":         dict(strategy="greedy"),
        "Epsilon-Greedy": dict(strategy="epsilon_greedy", epsilon=epsilon),
    }

    # ── Pre-simulate every strategy ─────────────────────────────────────────
    all_data = {}
    for idx, (name, cfg) in enumerate(strategy_map.items()):
        hist = presimulate(
            true_means  = true_means,
            reward_std  = reward_std,
            total_steps = steps,
            seed        = seed + idx * 99991,
            **cfg,
        )
        actions = np.array([h["action"] for h in hist])
        rewards = np.array([h["reward"] for h in hist])
        is_opt  = (actions == optimal_arm).astype(float)
        all_data[name] = dict(
            q_hist   = np.array([h["q_after"] for h in hist]),
            act_hist = actions,
            run_opt  = np.cumsum(is_opt) / (np.arange(steps) + 1),
            run_rew  = np.cumsum(rewards) / (np.arange(steps) + 1),
        )

    # ── Figure ──────────────────────────────────────────────────────────────
    C_BG, C_PANEL, C_BORDER = "#0d1117", "#161b22", "#30363d"
    C_TEXT, C_MUTED, C_GOLD = "#e6edf3", "#8b949e", "#ffa657"

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[1.6, 1],
                   hspace=0.52, wspace=0.32,
                   left=0.07, right=0.97, top=0.91, bottom=0.09)

    q_axes  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    opt_ax  = fig.add_subplot(gs[1, :2])
    rew_ax  = fig.add_subplot(gs[1, 2])
    arm_x   = np.arange(k)
    arm_lbl = [f"Arm {i+1}\n(μ={true_means[i]})" for i in range(k)]
    steps_x = np.arange(steps)

    bar_containers = []
    for i, (name, col) in enumerate(zip(strategy_map, PALETTE)):
        ax = q_axes[i]
        ax.set_facecolor(C_PANEL)
        ax.set_title(name, color=col, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-0.6, k - 0.4)
        ax.set_ylim(min(true_means) - 1.4, max(true_means) + 1.4)
        ax.set_xticks(arm_x)
        ax.set_xticklabels(arm_lbl, fontsize=9, color=C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.spines[:].set_color(C_BORDER)
        ax.set_ylabel("Q-value estimate", color=C_MUTED, fontsize=9)
        ax.axhline(0, color=C_BORDER, linewidth=0.8)
        for j, mu in enumerate(true_means):
            ax.hlines(mu, j - 0.3, j + 0.3,
                      colors=C_GOLD, linewidths=2, linestyles="--", alpha=0.9, zorder=3)
        bars = ax.bar(arm_x, np.zeros(k), color=col, alpha=0.55,
                      width=0.55, edgecolor=col, linewidth=1.5, zorder=2)
        bar_containers.append(bars)
        if i == 0:
            ax.legend(handles=[
                Line2D([0], [0], color=C_GOLD, linewidth=2,
                       linestyle="--", label="True mean")
            ], loc="upper left", facecolor=C_PANEL,
               edgecolor=C_BORDER, labelcolor=C_MUTED, fontsize=8)

    for ax in (opt_ax, rew_ax):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.spines[:].set_color(C_BORDER)
        ax.grid(alpha=0.12, color=C_MUTED)

    opt_ax.set_xlim(0, steps); opt_ax.set_ylim(-0.05, 1.08)
    opt_ax.set_xlabel("Step", color=C_MUTED, fontsize=9)
    opt_ax.set_ylabel("Optimal action rate", color=C_MUTED, fontsize=9)
    opt_ax.set_title("Optimal Action Rate", color=C_TEXT, fontsize=11, pad=6)
    opt_ax.axhline(1.0, color=C_MUTED, linewidth=0.5, linestyle=":")

    all_rr = [all_data[n]["run_rew"] for n in strategy_map]
    rew_ax.set_xlim(0, steps)
    rew_ax.set_ylim(min(r.min() for r in all_rr) - 0.15,
                    max(r.max() for r in all_rr) + 0.15)
    rew_ax.set_xlabel("Step", color=C_MUTED, fontsize=9)
    rew_ax.set_ylabel("Avg reward", color=C_MUTED, fontsize=9)
    rew_ax.set_title("Cumulative Avg Reward", color=C_TEXT, fontsize=11, pad=6)

    opt_lines, rew_lines = [], []
    for name, col in zip(strategy_map, PALETTE):
        ln, = opt_ax.plot([], [], color=col, linewidth=2, label=name)
        opt_lines.append(ln)
        ln, = rew_ax.plot([], [], color=col, linewidth=2, label=name)
        rew_lines.append(ln)
    opt_ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
                  labelcolor=C_TEXT, fontsize=8, loc="lower right")
    rew_ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
                  labelcolor=C_TEXT, fontsize=8, loc="lower right")

    fig.text(0.5, 0.975,
             "Multi-Armed Bandit — Live Decision Process",
             ha="center", va="top", color=C_TEXT, fontsize=15, fontweight="bold")
    step_lbl = fig.text(0.5, 0.945, "Step: 0",
                        ha="center", va="top", color=C_MUTED, fontsize=10)

    skip         = max(1, steps // 200)
    frame_idx    = range(0, steps, skip)
    strat_names  = list(strategy_map.keys())

    def _update(t):
        for i, name in enumerate(strat_names):
            col  = PALETTE[i]
            data = all_data[name]
            q_v  = data["q_hist"][t]
            act  = data["act_hist"][t]
            for j, bar in enumerate(bar_containers[i]):
                h = float(q_v[j])
                bar.set_height(abs(h)); bar.set_y(min(0.0, h))
                if j == act:
                    bar.set_alpha(1.0); bar.set_edgecolor("white"); bar.set_linewidth(2.5)
                else:
                    bar.set_alpha(0.45); bar.set_edgecolor(col); bar.set_linewidth(1.0)
            xs = steps_x[:t + 1]
            opt_lines[i].set_data(xs, data["run_opt"][:t + 1])
            rew_lines[i].set_data(xs, data["run_rew"][:t + 1])
        step_lbl.set_text(f"Step: {t + 1} / {steps}")

    anim = FuncAnimation(fig, _update, frames=frame_idx,
                         interval=interval, blit=False)

    if save_path:
        _save(anim, save_path)
    else:
        plt.show()
    return anim


def _save(anim, path):
    import os
    try:
        writer = FFMpegWriter(fps=30, bitrate=2500)
        anim.save(path, writer=writer, dpi=120)
        print(f"[✓] Saved MP4: {path}")
    except Exception as e:
        gif = os.path.splitext(path)[0] + ".gif"
        anim.save(gif, writer="pillow", fps=30, dpi=90)
        print(f"[✓] Saved GIF: {gif}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    save_path = None
    if "--save" in sys.argv:
        save_path = sys.argv[sys.argv.index("--save") + 1]

    print("Running multi-run experiment …")
    avg_rewards, avg_cumulative, optimal_action_rate = run_experiment(
        true_means = TRUE_MEANS,
        steps      = STEPS,
        runs       = RUNS,
        epsilon    = EPSILON,
        reward_std = REWARD_STD,
        base_seed  = BASE_SEED,
    )

    plot_results(
        avg_cumulative_rewards = avg_cumulative,
        optimal_action_rate    = optimal_action_rate,
        true_means             = TRUE_MEANS,
        epsilon                = EPSILON,
    )

    # ── Animated single-run dashboard ────────────────────────────────────────
    # Set save_path via --save out.mp4, or leave None for a live window.
    animate_experiment(
        true_means = TRUE_MEANS,
        steps      = ANIM_STEPS,
        epsilon    = EPSILON,
        reward_std = REWARD_STD,
        seed       = ANIM_SEED,
        interval   = 25,
        save_path  = save_path,
    )