"""Visualize RA8 training run: adaptive ratio only (no LR dampening)."""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# RA8 eval data extracted from train.log
data = [
    (50, 102400, 0.130, 0.870, "pursuer"),
    (100, 204800, 0.090, 0.910, "pursuer"),
    (150, 307200, 0.080, 0.920, "pursuer"),
    (200, 409600, 0.170, 0.830, "pursuer"),
    (250, 512000, 0.100, 0.900, "pursuer"),
    (300, 614400, 0.100, 0.900, "pursuer"),
    (350, 716800, 0.150, 0.850, "pursuer"),
    (400, 819200, 0.980, 0.020, "evader"),
    (450, 921600, 0.950, 0.050, "evader"),
    (500, 1024000, 0.990, 0.010, "evader"),
    (550, 1126400, 0.980, 0.020, "evader"),
    (600, 1228800, 1.000, 0.000, "evader"),
    (650, 1331200, 0.990, 0.010, "evader"),
    (700, 1433600, 0.980, 0.020, "evader"),
    (750, 1536000, 0.980, 0.020, "evader"),
    (800, 1638400, 0.970, 0.030, "evader"),
    (850, 1740800, 0.970, 0.030, "evader"),
    (900, 1843200, 0.980, 0.020, "evader"),
    (950, 1945600, 0.990, 0.010, "evader"),
    (1000, 2048000, 1.000, 0.000, "evader"),
    (1050, 2150400, 0.990, 0.010, "evader"),
    (1100, 2252800, 1.000, 0.000, "evader"),
    (1150, 2355200, 1.000, 0.000, "evader"),
    (1200, 2457600, 0.970, 0.030, "evader"),
    (1250, 2560000, 1.000, 0.000, "evader"),
    (1300, 2662400, 1.000, 0.000, "evader"),
    (1350, 2764800, 1.000, 0.000, "evader"),
    (1400, 2867200, 0.970, 0.030, "evader"),
    (1450, 2969600, 0.980, 0.020, "evader"),
    (1500, 3072000, 0.990, 0.010, "evader"),
    (1550, 3174400, 0.960, 0.040, "evader"),
    (1600, 3276800, 1.000, 0.000, "evader"),
    (1650, 3379200, 0.990, 0.010, "evader"),
    (1700, 3481600, 1.000, 0.000, "evader"),
    (1750, 3584000, 1.000, 0.000, "evader"),
    (1800, 3686400, 1.000, 0.000, "evader"),
    (1850, 3788800, 0.990, 0.010, "evader"),
    (1900, 3891200, 0.990, 0.010, "evader"),
    (1950, 3993600, 1.000, 0.000, "evader"),
    (2000, 4096000, 0.990, 0.010, "evader"),
    (2050, 4198400, 0.990, 0.010, "evader"),
    (2100, 4300800, 0.970, 0.030, "evader"),
    (2150, 4403200, 0.940, 0.060, "evader"),
    (2200, 4505600, 1.000, 0.000, "evader"),
    (2250, 4608000, 0.980, 0.020, "evader"),
    (2300, 4710400, 0.970, 0.030, "evader"),
    (2350, 4812800, 0.990, 0.010, "evader"),
    (2400, 4915200, 0.970, 0.030, "evader"),
    (2450, 5017600, 0.990, 0.010, "evader"),
    (2500, 5120000, 1.000, 0.000, "evader"),
    (2550, 5222400, 1.000, 0.000, "evader"),
    (2600, 5324800, 0.990, 0.010, "evader"),
    (2650, 5427200, 1.000, 0.000, "evader"),
    (2700, 5529600, 0.970, 0.030, "evader"),
    (2750, 5632000, 1.000, 0.000, "evader"),
    (2800, 5734400, 0.980, 0.020, "evader"),
    (2850, 5836800, 1.000, 0.000, "evader"),
    (2900, 5939200, 0.980, 0.020, "evader"),
    (2950, 6041600, 1.000, 0.000, "evader"),
    (3000, 6144000, 1.000, 0.000, "evader"),
    (3050, 6246400, 0.980, 0.020, "evader"),
    (3100, 6348800, 0.980, 0.020, "evader"),
    (3150, 6451200, 0.990, 0.010, "evader"),
    (3200, 6553600, 0.990, 0.010, "evader"),
    (3250, 6656000, 1.000, 0.000, "evader"),
    (3300, 6758400, 0.990, 0.010, "evader"),
    (3350, 6860800, 0.990, 0.010, "evader"),
    (3400, 6963200, 0.990, 0.010, "evader"),
    (3450, 7065600, 1.000, 0.000, "evader"),
    (3500, 7168000, 1.000, 0.000, "evader"),
    (3550, 7270400, 0.980, 0.020, "evader"),
    (3600, 7372800, 0.940, 0.060, "evader"),
    (3650, 7475200, 0.990, 0.010, "evader"),
    (3700, 7577600, 1.000, 0.000, "evader"),
    (3750, 7680000, 0.980, 0.020, "evader"),
    (3800, 7782400, 0.960, 0.040, "evader"),
    (3850, 7884800, 0.980, 0.020, "evader"),
    (3900, 7987200, 0.990, 0.010, "evader"),
    (3950, 8089600, 0.950, 0.050, "evader"),
    (4000, 8192000, 0.990, 0.010, "evader"),
    (4050, 8294400, 0.990, 0.010, "evader"),
    (4100, 8396800, 0.990, 0.010, "evader"),
    (4150, 8499200, 0.980, 0.020, "evader"),
    (4200, 8601600, 0.980, 0.020, "evader"),
    (4250, 8704000, 0.990, 0.010, "evader"),
    (4300, 8806400, 0.990, 0.010, "evader"),
    (4350, 8908800, 1.000, 0.000, "evader"),
    (4400, 9011200, 0.970, 0.030, "evader"),
    (4450, 9113600, 0.990, 0.010, "evader"),
    (4500, 9216000, 0.980, 0.020, "evader"),
    (4550, 9318400, 0.970, 0.030, "evader"),
    (4600, 9420800, 1.000, 0.000, "evader"),
    (4650, 9523200, 1.000, 0.000, "evader"),
    (4700, 9625600, 0.970, 0.030, "evader"),
    (4750, 9728000, 1.000, 0.000, "evader"),
    (4800, 9830400, 1.000, 0.000, "evader"),
    (4850, 9932800, 1.000, 0.000, "evader"),
]

micro = np.array([d[0] for d in data])
steps = np.array([d[1] for d in data])
sr_p = np.array([d[2] for d in data])
sr_e = np.array([d[3] for d in data])
boost_target = [d[4] for d in data]
gap = np.abs(sr_p - sr_e)
steps_M = steps / 1e6  # millions

# --- Figure: 3 subplots ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.5, 1]})
fig.suptitle("RA8: Adaptive Training Ratio Only (no LR Dampening)\n"
             "10x10 arena, 1.05x evader speed, ±50 terminal, no PBRS, pool=50, seed=47",
             fontsize=13, fontweight="bold")

# --- Panel 1: Success Rates ---
ax1 = axes[0]
ax1.plot(steps_M, sr_p, color="#d62728", linewidth=1.8, label="Pursuer capture rate", zorder=3)
ax1.plot(steps_M, sr_e, color="#1f77b4", linewidth=1.8, label="Evader escape rate", zorder=3)
ax1.fill_between(steps_M, sr_p, sr_e, alpha=0.08, color="gray")

# Mark the flip point
flip_idx = 7  # M400
ax1.axvline(steps_M[flip_idx], color="black", linestyle="--", alpha=0.5, linewidth=1)
ax1.annotate("Flip at M400\n(pursuer boost\noverpowers evader)",
             xy=(steps_M[flip_idx], 0.5), xytext=(steps_M[flip_idx] + 0.8, 0.45),
             fontsize=9, ha="left",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# Equilibrium band
ax1.axhline(0.5, color="green", linestyle=":", alpha=0.4, linewidth=1)
ax1.axhspan(0.45, 0.55, color="green", alpha=0.05)
ax1.text(9.5, 0.52, "NE target", fontsize=8, color="green", alpha=0.7, ha="right")

ax1.set_ylabel("Success Rate", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc="center right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title("Success Rates Over Training", fontsize=11)

# --- Panel 2: NE Gap ---
ax2 = axes[1]
ax2.fill_between(steps_M, gap, color="#ff7f0e", alpha=0.3)
ax2.plot(steps_M, gap, color="#ff7f0e", linewidth=1.5)
ax2.axhline(0.3, color="red", linestyle="--", alpha=0.6, linewidth=1,
            label="Adaptive ratio threshold (0.3)")
ax2.axhline(0.1, color="green", linestyle="--", alpha=0.6, linewidth=1,
            label="Convergence threshold (0.1)")
ax2.set_ylabel("NE Gap", fontsize=11)
ax2.set_ylim(-0.05, 1.1)
ax2.legend(loc="center right", fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Boost Target ---
ax3 = axes[2]
boost_colors = ["#d62728" if b == "pursuer" else "#1f77b4" for b in boost_target]
ax3.bar(steps_M, [1] * len(steps_M), width=0.08, color=boost_colors, alpha=0.7)
ax3.set_ylabel("Boost\nTarget", fontsize=10)
ax3.set_yticks([])
ax3.set_xlabel("Training Steps (millions)", fontsize=11)
ax3.grid(True, alpha=0.3, axis="x")

# Legend for boost
from matplotlib.patches import Patch
ax3.legend(handles=[
    Patch(facecolor="#d62728", alpha=0.7, label="Pursuer boosted"),
    Patch(facecolor="#1f77b4", alpha=0.7, label="Evader boosted"),
], loc="center right", fontsize=9)

# X-axis formatting
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax3.set_xlim(-0.1, 10.3)

plt.tight_layout()
plt.savefig("results/ra8_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: results/ra8_analysis.png")
plt.close()
