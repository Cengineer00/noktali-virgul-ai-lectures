"""
bandit_scene.py
───────────────────────────────────────────────────────────────────────────────
Educational step-by-step animation of an ε-Greedy Multi-Armed Bandit.

Layout (16 × 9 figure, data-coords 0..16 × 0..9.5):
  Title + subtitle
  [Agent] with dashed connection line to chosen arm
  [Arm 1]         [Arm 2]         [Arm 3]   ← slot machines
  Q(a): …         …               …
  P(%): ██░░      …               …
  Last: …         …               …

Each bandit step is split into 5 animation phases:
  SELECT   (0.00→0.25) — connection line grows; chosen arm glows
  REWARD   (0.25→0.45) — reward text floats up and fades
  UPDATE_Q (0.45→0.65) — Q-value smoothly interpolates to new value
  UPDATE_P (0.65→0.85) — probability bars smoothly interpolate
  PAUSE    (0.85→1.00) — hold final state

Usage:
  python bandit_scene.py                    # interactive window
  python bandit_scene.py --save out.mp4     # MP4 via ffmpeg
  python bandit_scene.py --save out.gif     # GIF via Pillow
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRUE_MEANS      = [1.0, 1.2, 0.8]
EPSILON         = 0.1
REWARD_STD      = 1.0
SEED            = 41

TOTAL_STEPS     = 100          # how many bandit decisions to animate
FRAMES_PER_STEP = 10           # animation frames per decision
FPS             = 30

# ── Policy selection ─────────────────────────────────────────────────────────
# Options: "epsilon_greedy"  |  "greedy"  |  "random"
POLICY = "epsilon_greedy"
# POLICY = "random"
# POLICY = "greedy"

# Display metadata shown in title/subtitle
_POLICY_NAMES = {
    "epsilon_greedy": "ε-Greedy",
    "greedy":         "Greedy",
    "random":         "Random",
}
_POLICY_SUBTITLES = {
    "epsilon_greedy": f"ε = {EPSILON}   |   True means: {TRUE_MEANS}   |   Rewards ~ N(μ, {REWARD_STD})",
    "greedy":         f"Always exploits best Q(a)   |   True means: {TRUE_MEANS}   |   Rewards ~ N(μ, {REWARD_STD})",
    "random":         f"Uniform random selection   |   True means: {TRUE_MEANS}   |   Rewards ~ N(μ, {REWARD_STD})",
}

# Phase boundaries (as fraction of FRAMES_PER_STEP)
PH_SELECT_END   = 0.25
PH_REWARD_END   = 0.45
PH_UPDATEQ_END  = 0.65
PH_UPDATEP_END  = 0.85
# 0.85 → 1.00 = pause

# ── Colours ──────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_BORDER = "#30363d"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"
C_GOLD   = "#ffa657"
C_GREEN  = "#3fb950"
C_RED    = "#f78166"
C_AGENT  = "#a371f7"
ARM_COLS = ["#58a6ff", "#f78166", "#3fb950"]

# ── Layout (data-units) ───────────────────────────────────────────────────────
FIG_W, FIG_H = 16.0, 9.5
M_XC    = [3.0, 8.0, 13.0]   # machine centre-x positions
M_W, M_H = 3.2, 3.2          # machine width / height
M_Y_BOT  = 3.5               # machine bottom y  (lowered to clear agent)
M_Y_TOP  = M_Y_BOT + M_H

AGENT_XC = 8.0
AGENT_YC = 7.5               # lowered so antenna stays below subtitle

# ── Info-panel row y-positions (below machines) ───────────────────────────────
ROW_Q   = M_Y_BOT - 0.55
ROW_P   = ROW_Q   - 0.62
ROW_BAR = ROW_P   - 0.30
ROW_LR  = ROW_BAR - 0.48
BAR_W   = 2.5
BAR_H   = 0.22


# ── Bandit logic lives in bandit_core.py ─────────────────────────────────────
from bandit_core import presimulate as _presimulate

def presimulate():
    """Thin wrapper: calls bandit_core.presimulate with module-level config."""
    return _presimulate(
        true_means  = TRUE_MEANS,
        strategy    = POLICY,
        epsilon     = EPSILON,
        reward_std  = REWARD_STD,
        total_steps = TOTAL_STEPS,
        seed        = SEED,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ease(t):
    """Cubic smooth-step: 0 → 0, 1 → 1, zero derivative at both ends."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def phase_progress(phase, p_start, p_end):
    """Return eased t in [0,1] if phase falls within [p_start, p_end], else None."""
    if phase < p_start or phase > p_end:
        return None
    return ease((phase - p_start) / (p_end - p_start))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE / ARTIST SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_machine(ax, cx, col, label):
    """
    Draw one slot machine and return a dict of *modifiable* artists so the
    animation loop can change colours / alphas without adding new patches.
    """
    x0, y0 = cx - M_W / 2, M_Y_BOT
    arts = {}

    # Three glow layers (hidden by default; revealed when the arm is chosen)
    glows = []
    for alpha_max, extra in [(0.08, 0.9), (0.13, 0.55), (0.22, 0.25)]:
        gp = FancyBboxPatch(
            (x0 - extra, y0 - extra), M_W + 2 * extra, M_H + 2 * extra,
            boxstyle="round,pad=0.2", linewidth=0,
            facecolor=col, alpha=0.0, zorder=1,
        )
        ax.add_patch(gp)
        glows.append((gp, alpha_max))
    arts["glows"] = glows

    # Machine body
    body = FancyBboxPatch(
        (x0, y0), M_W, M_H,
        boxstyle="round,pad=0.12",
        linewidth=1.8, edgecolor=C_BORDER,
        facecolor=C_PANEL, zorder=2,
    )
    ax.add_patch(body)
    arts["body"] = body

    # Coloured top bar
    top = FancyBboxPatch(
        (x0 + 0.18, y0 + M_H - 0.58), M_W - 0.36, 0.46,
        boxstyle="round,pad=0.05", linewidth=1,
        edgecolor=col, facecolor=col, alpha=0.25, zorder=3,
    )
    ax.add_patch(top)
    arts["top"] = top

    # Arm label inside top bar
    ax.text(cx, y0 + M_H - 0.35, label,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, zorder=4)

    # Display window
    win_pad, win_h = 0.28, 1.05
    win_y = y0 + M_H * 0.33
    ax.add_patch(FancyBboxPatch(
        (x0 + win_pad, win_y), M_W - 2 * win_pad, win_h,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor=C_BORDER, facecolor=C_BG, zorder=3,
    ))

    # Symbols inside display window (change when glowing)
    sym = ax.text(cx, win_y + win_h / 2, "· · ·",
                  ha="center", va="center",
                  fontsize=12, color=C_MUTED, zorder=4)
    arts["sym"] = sym

    # Lever (right side) — knob position changes when selected
    lever_x = x0 + M_W + 0.06
    lever_line, = ax.plot(
        [lever_x, lever_x], [y0 + 0.4, y0 + M_H * 0.72],
        color=C_BORDER, linewidth=2.2, zorder=3,
    )
    arts["lever_line"] = lever_line
    knob = Circle((lever_x, y0 + M_H * 0.72), 0.13, color=C_MUTED, zorder=4)
    ax.add_patch(knob)
    arts["knob"] = knob

    # Coin slot at the bottom
    ax.add_patch(FancyBboxPatch(
        (cx - 0.35, y0 + 0.12), 0.70, 0.16,
        boxstyle="round,pad=0.02", linewidth=1,
        edgecolor=C_BORDER, facecolor=C_BG, zorder=3,
    ))

    return arts


def set_machine_glow(arts, col, on: bool):
    """Toggle the glow state of a slot machine artist bundle."""
    for gp, alpha_max in arts["glows"]:
        gp.set_alpha(alpha_max if on else 0.0)
    arts["body"].set_edgecolor(col if on else C_BORDER)
    arts["body"].set_linewidth(2.8 if on else 1.8)
    arts["top"].set_alpha(0.55 if on else 0.25)
    arts["sym"].set_text("★ ★ ★" if on else "· · ·")
    arts["sym"].set_color(col if on else C_MUTED)
    arts["sym"].set_fontsize(13 if on else 12)
    arts["knob"].set_color(col if on else C_MUTED)
    # Lever knob moves down when "pulled"
    x0 = M_XC[0] - M_W / 2   # dummy; we get real x from knob centre
    kx = arts["knob"].center[0]
    ky_base = M_Y_BOT + M_H * 0.72
    arts["knob"].center = (kx, ky_base - (0.35 if on else 0.0))


def build_figure():
    """Create all artists and return (fig, ax, artist_dict)."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ── Static text ───────────────────────────────────────────────────────────
    ax.text(8, 9.2, _POLICY_NAMES[POLICY] + " — Multi-Armed Bandit",
            ha="center", va="center",
            fontsize=20, fontweight="bold", color=C_TEXT)
    ax.text(8, 8.82, _POLICY_SUBTITLES[POLICY],
            ha="center", va="center", fontsize=10.5, color=C_MUTED)

    # Ground line
    ax.axhline(M_Y_BOT - 0.06, xmin=0.08, xmax=0.92,
               color=C_BORDER, linewidth=1.2, alpha=0.6)

    # ── Agent ─────────────────────────────────────────────────────────────────
    cx, cy = AGENT_XC, AGENT_YC
    ax.add_patch(FancyBboxPatch(          # body
        (cx - 0.54, cy - 0.40), 1.08, 0.68,
        boxstyle="round,pad=0.09", linewidth=2,
        edgecolor=C_AGENT, facecolor=C_PANEL, zorder=5,
    ))
    ax.add_patch(FancyBboxPatch(          # head
        (cx - 0.39, cy + 0.28), 0.78, 0.56,
        boxstyle="round,pad=0.07", linewidth=2,
        edgecolor=C_AGENT, facecolor=C_PANEL, zorder=5,
    ))
    for ex in [cx - 0.155, cx + 0.155]:  # eyes
        ax.add_patch(Circle((ex, cy + 0.575), 0.077, color=C_AGENT, zorder=6))
    ax.plot([cx, cx], [cy + 0.84, cy + 1.00],   # antenna (shortened)
            color=C_AGENT, linewidth=2, zorder=5)
    ax.add_patch(Circle((cx, cy + 1.05), 0.068, color=C_GOLD, zorder=6))

    # ── Slot machines ─────────────────────────────────────────────────────────
    machine_arts = []
    for i, (cx_m, col) in enumerate(zip(M_XC, ARM_COLS)):
        machine_arts.append(build_machine(ax, cx_m, col, f"Arm {i + 1}"))

    # ── Info panels ───────────────────────────────────────────────────────────
    k = len(TRUE_MEANS)
    q_texts, pct_texts, bar_data, lr_texts, reward_texts = [], [], [], [], []

    for i, (cx_m, col) in enumerate(zip(M_XC, ARM_COLS)):
        # Q(a)
        ax.text(cx_m, ROW_Q + 0.26, "Q(a)",
                ha="center", va="bottom", fontsize=9, color=C_MUTED)
        qt = ax.text(cx_m, ROW_Q, "0.000",
                     ha="center", va="center",
                     fontsize=16, fontweight="bold", color=col)
        q_texts.append(qt)

        # P(next)
        ax.text(cx_m, ROW_P + 0.28, "P(next)",
                ha="center", va="bottom", fontsize=9, color=C_MUTED)
        pp = ax.text(cx_m, ROW_P, f"{100 / k:.1f}%",
                     ha="center", va="center", fontsize=13, color=C_TEXT)
        pct_texts.append(pp)

        # Probability bar
        bx0 = cx_m - BAR_W / 2
        ax.add_patch(FancyBboxPatch(     # background track
            (bx0, ROW_BAR), BAR_W, BAR_H,
            boxstyle="round,pad=0.03", linewidth=1,
            edgecolor=C_BORDER, facecolor=C_BG, zorder=2,
        ))
        fg = FancyBboxPatch(             # filled portion
            (bx0, ROW_BAR), BAR_W / k, BAR_H,
            boxstyle="round,pad=0.03", linewidth=0,
            facecolor=col, alpha=0.85, zorder=3,
        )
        ax.add_patch(fg)
        bar_data.append((fg, bx0))      # store fg patch + left edge

        # Last reward
        lr = ax.text(cx_m, ROW_LR, "Last:  —",
                     ha="center", va="center", fontsize=9, color=C_MUTED)
        lr_texts.append(lr)

        # Floating reward text (invisible initially)
        rt = ax.text(cx_m, M_Y_TOP + 0.45, "",
                     ha="center", va="bottom",
                     fontsize=18, fontweight="bold",
                     color=C_GOLD, alpha=0.0, zorder=10)
        reward_texts.append(rt)

    # ── Connection line (agent → chosen arm) ──────────────────────────────────
    conn_line, = ax.plot([], [], color=C_AGENT, linewidth=2.5,
                         linestyle="--", alpha=0.0, zorder=4,
                         dash_capstyle="round")

    # ── Step counter ──────────────────────────────────────────────────────────
    step_text = ax.text(0.55, 0.28,
                        f"Step: 0 / {TOTAL_STEPS}",
                        ha="left", va="center", fontsize=11, color=C_MUTED)

    return fig, ax, dict(
        machine_arts = machine_arts,
        q_texts      = q_texts,
        pct_texts    = pct_texts,
        bar_data     = bar_data,
        lr_texts     = lr_texts,
        reward_texts = reward_texts,
        conn_line    = conn_line,
        step_text    = step_text,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION UPDATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_update(history, arts):
    """Return the per-frame update function for FuncAnimation."""
    k = len(TRUE_MEANS)

    def update(frame):
        step_idx      = min(frame // FRAMES_PER_STEP, TOTAL_STEPS - 1)
        frame_in_step = frame % FRAMES_PER_STEP
        phase         = frame_in_step / FRAMES_PER_STEP   # 0.0 → <1.0

        info   = history[step_idx]
        chosen = info["action"]
        reward = info["reward"]
        col    = ARM_COLS[chosen]

        arts["step_text"].set_text(f"Step: {step_idx + 1} / {TOTAL_STEPS}")

        # ── Machine glow: on during SELECT and REWARD phases ────────────────
        for i, marts in enumerate(arts["machine_arts"]):
            glow_on = (i == chosen) and (phase < PH_REWARD_END)
            set_machine_glow(marts, ARM_COLS[i], glow_on)

        # ── PHASE 0 · SELECT ─────────────────────────────────────────────────
        # Connection line grows from agent to chosen arm
        t_sel = phase_progress(phase, 0.0, PH_SELECT_END)
        if t_sel is not None:
            ax_x = AGENT_XC
            ax_y = AGENT_YC - 0.40        # bottom of agent body
            mx_x = M_XC[chosen]
            mx_y = M_Y_TOP
            ex   = lerp(ax_x, mx_x, t_sel)
            ey   = lerp(ax_y, mx_y, t_sel)
            arts["conn_line"].set_data([ax_x, ex], [ax_y, ey])
            arts["conn_line"].set_alpha(0.9 * t_sel)
            for rt in arts["reward_texts"]:
                rt.set_alpha(0.0)
            return

        # ── PHASE 1 · REWARD ─────────────────────────────────────────────────
        # Reward text floats up; connection line fades
        t_rew = phase_progress(phase, PH_SELECT_END, PH_REWARD_END)
        if t_rew is not None:
            arts["conn_line"].set_alpha(0.9 * (1.0 - t_rew))
            sign   = "+" if reward >= 0 else ""
            r_col  = C_GREEN if reward >= 0 else C_RED
            for i, rt in enumerate(arts["reward_texts"]):
                if i == chosen:
                    rt.set_text(f"R = {sign}{reward:.2f}")
                    rt.set_color(r_col)
                    rt.set_position((M_XC[chosen], M_Y_TOP + 0.3 + t_rew * 0.65))
                    rt.set_alpha(t_rew)
                else:
                    rt.set_alpha(0.0)
            return

        # ── PHASE 2 · UPDATE Q ───────────────────────────────────────────────
        # Q-value smoothly interpolates to new value; reward text fades out
        t_q = phase_progress(phase, PH_REWARD_END, PH_UPDATEQ_END)
        if t_q is not None:
            arts["conn_line"].set_alpha(0.0)
            # Fade reward text
            for i, rt in enumerate(arts["reward_texts"]):
                rt.set_alpha(max(0.0, 1.0 - t_q * 2.5) if i == chosen else 0.0)
            # Interpolate Q-values (only chosen arm changes)
            for i, qt in enumerate(arts["q_texts"]):
                q = lerp(info["q_before"][i], info["q_after"][i], t_q) \
                    if i == chosen else info["q_before"][i]
                qt.set_text(f"{q:.3f}")
            # Update last-reward text halfway through
            if t_q > 0.5:
                for i, lr in enumerate(arts["lr_texts"]):
                    lrv = info["lr_after"][i]
                    if lrv is None:
                        lr.set_text("Last:  —")
                    else:
                        s = "+" if lrv >= 0 else ""
                        lr.set_text(f"Last:  {s}{lrv:.2f}")
            return

        # ── PHASE 3 · UPDATE P ───────────────────────────────────────────────
        # Probability bars and percentages smoothly interpolate
        t_p = phase_progress(phase, PH_UPDATEQ_END, PH_UPDATEP_END)
        if t_p is not None:
            arts["conn_line"].set_alpha(0.0)
            for rt in arts["reward_texts"]:
                rt.set_alpha(0.0)
            # Show final Q values
            for i, qt in enumerate(arts["q_texts"]):
                qt.set_text(f"{info['q_after'][i]:.3f}")
            # Interpolate P bars
            best_p = max(info["prob_after"])
            for i, ((fg, bx0), pp) in enumerate(zip(arts["bar_data"], arts["pct_texts"])):
                p = lerp(info["prob_before"][i], info["prob_after"][i], t_p)
                fg.set_width(max(0.01, BAR_W * p))
                pp.set_text(f"{p * 100:.1f}%")
                # Highlight the arm with highest probability
                is_best = abs(info["prob_after"][i] - best_p) < 1e-9
                pp.set_color(ARM_COLS[i] if is_best else C_TEXT)
            return

        # ── PHASE 4 · PAUSE ──────────────────────────────────────────────────
        # Everything at its final state
        arts["conn_line"].set_alpha(0.0)
        for rt in arts["reward_texts"]:
            rt.set_alpha(0.0)
        for i, qt in enumerate(arts["q_texts"]):
            qt.set_text(f"{info['q_after'][i]:.3f}")
        best_p = max(info["prob_after"])
        for i, ((fg, bx0), pp) in enumerate(zip(arts["bar_data"], arts["pct_texts"])):
            p = info["prob_after"][i]
            fg.set_width(max(0.01, BAR_W * p))
            pp.set_text(f"{p * 100:.1f}%")
            is_best = abs(p - best_p) < 1e-9
            pp.set_color(ARM_COLS[i] if is_best else C_TEXT)
        for i, lr in enumerate(arts["lr_texts"]):
            lrv = info["lr_after"][i]
            if lrv is None:
                lr.set_text("Last:  —")
            else:
                s = "+" if lrv >= 0 else ""
                lr.set_text(f"Last:  {s}{lrv:.2f}")

    return update


# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def save_animation(anim, path):
    import os
    try:
        writer = FFMpegWriter(
            fps=FPS, bitrate=3500,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        anim.save(path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": C_BG})
        print(f"[✓] Saved MP4: {path}")
    except Exception as exc:
        print(f"[!] ffmpeg failed ({exc})\nFalling back to GIF …")
        gif = os.path.splitext(path)[0] + ".gif"
        anim.save(gif, writer="pillow", fps=FPS, dpi=90,
                  savefig_kwargs={"facecolor": C_BG})
        print(f"[✓] Saved GIF: {gif}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    save_path = None
    if "--save" in sys.argv:
        idx       = sys.argv.index("--save")
        save_path = sys.argv[idx + 1]

    print("Pre-simulating bandit experiment …")
    history = presimulate()

    print("Building figure …")
    fig, ax, arts = build_figure()

    total_frames = TOTAL_STEPS * FRAMES_PER_STEP
    update_fn    = make_update(history, arts)

    print(f"Starting animation ({total_frames} frames, {FPS} fps) …")
    anim = FuncAnimation(
        fig, update_fn,
        frames=total_frames,
        interval=1000 / FPS,
        blit=False,
    )

    if save_path:
        save_animation(anim, save_path)
    else:
        plt.tight_layout()
        plt.show()
