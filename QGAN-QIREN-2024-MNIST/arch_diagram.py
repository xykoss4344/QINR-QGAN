"""
Architecture diagrams for Classical CWGAN and Quantum PQWGAN crystal generators.
Saves to results_analysis/architecture_diagram.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette (white background) ─────────────────────────────────────────
BG      = '#ffffff'
PANEL   = '#f6f8fa'
BORDER  = '#d0d7de'
WHITE   = '#1a1a2e'   # text colour (dark on white)
DIM     = '#57606a'
ACCENT1 = '#0969da'   # blue  – classical
ACCENT2 = '#1a7f37'   # green – quantum
PURPLE  = '#8250df'   # quantum circuit
ORANGE  = '#bc4c00'   # output
TEAL    = '#0a7d4b'
YELLOW  = '#9a6700'
RED     = '#cf222e'

def draw_box(ax, x, y, w, h, label, sublabel=None,
             fc='#1f2937', ec=ACCENT1, lw=1.5, fontsize=8.5, radius=0.4):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f'round,pad=0.05,rounding_size={radius}',
                         fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center',
                color=WHITE, fontsize=fontsize, fontweight='bold', zorder=4)
        ax.text(x, y - 0.2, sublabel, ha='center', va='center',
                color=DIM, fontsize=6.5, zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                color=WHITE, fontsize=fontsize, fontweight='bold', zorder=4)

def arrow(ax, x0, y0, x1, y1, color=DIM, lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=5)

def dim_text(ax, x, y, txt, color=DIM, fontsize=6.5, ha='center'):
    ax.text(x, y, txt, ha=ha, va='center', color=color, fontsize=fontsize, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Figure setup — two separate figures
# ══════════════════════════════════════════════════════════════════════════════
fig_cls = plt.figure(figsize=(11, 14), facecolor=BG)
fig_q   = plt.figure(figsize=(11, 14), facecolor=BG)

ax_cls = fig_cls.add_axes([0.03, 0.04, 0.94, 0.88])
ax_q   = fig_q.add_axes([0.03, 0.04, 0.94, 0.88])

for ax in (ax_cls, ax_q):
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.set_aspect('equal')
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(1.5)


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Classical CWGAN Generator
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_cls
CX = 5.0   # centre-x of main column

# Title
ax.text(CX, 17.3, 'Classical CWGAN', ha='center', va='center',
        color=ACCENT1, fontsize=14, fontweight='bold')
ax.text(CX, 16.85, 'Composition-Conditioned Crystal Generator',
        ha='center', va='center', color=DIM, fontsize=8)

# ── Input block ───────────────────────────────────────────────────────────────
# noise
draw_box(ax, 2.5, 16.0, 2.2, 0.65, 'Noise z', 'latent dim = 512',
         fc='#dbeafe', ec=ACCENT1, fontsize=8)
# Composition labels
draw_box(ax, 7.5, 16.0, 2.2, 0.65, 'Composition Labels', 'c₁(8) c₂(8) c₃(12) cₙ(1)',
         fc='#dbeafe', ec=YELLOW, fontsize=7.5)

# concat arrow → Linear
arrow(ax, 2.5, 15.67, 3.9, 15.15, ACCENT1)
arrow(ax, 7.5, 15.67, 6.1, 15.15, YELLOW)
dim_text(ax, CX, 15.35, 'concat  [512 + 8 + 8 + 12 + 1 = 541]')

draw_box(ax, CX, 14.9, 3.8, 0.62, 'Linear', '541 → 128 × 28',
         fc='#dbeafe', ec=ACCENT1)
arrow(ax, CX, 14.59, CX, 14.18)
dim_text(ax, CX, 14.38, 'Reshape  →  (B, 128, 28, 1)')

# ── ConvTranspose tower ───────────────────────────────────────────────────────
conv_steps = [
    ('ConvTranspose2d', '128 ch → 256 ch', 'kernel (1,3)  stride (1,1)',  13.85),
    ('BatchNorm2d + ReLU', '256 ch', '',                                    13.1),
    ('ConvTranspose2d', '256 ch → 512 ch', 'kernel (1,1)',                 12.35),
    ('BatchNorm2d + ReLU', '512 ch', '',                                    11.6),
    ('ConvTranspose2d', '512 ch → 256 ch', 'kernel (1,1)',                 10.85),
    ('BatchNorm2d + ReLU', '256 ch', '',                                    10.1),
    ('ConvTranspose2d', '256 ch → 1 ch',   'kernel (1,1)',                  9.35),
    ('Tanh', 'Feature map  28 × 3', '',                                     8.6),
]
for label, sub1, sub2, cy in conv_steps:
    full_sub = sub1 + ('  ' + sub2 if sub2 else '')
    draw_box(ax, CX, cy, 4.8, 0.58, label, full_sub,
             fc='#dbeafe', ec=ACCENT1, fontsize=8)
    if cy < 13.85:
        arrow(ax, CX, cy + 0.58/2 + 0.58, CX, cy + 0.29 + 0.05)

arrow(ax, CX, 14.18, CX, 14.14)

# ── Split: atom positions + cell map ─────────────────────────────────────────
split_y = 8.29
arrow(ax, CX, split_y, CX, split_y - 0.01)

# Flatten label
dim_text(ax, CX, 8.05, 'Flatten  →  84-dim  (atom positions)')

# atom_pos box
draw_box(ax, 3.0, 7.5, 2.8, 0.62, 'Atom Positions', '84-dim  (28 atoms × 3)',
         fc='#dcfce7', ec=ACCENT2, fontsize=8)

# cellmap branch
draw_box(ax, 7.5, 7.5, 2.8, 0.62, 'Cell Map Linear', '84 → 30 → 6  +  Sigmoid',
         fc='#ffedd5', ec=ORANGE, fontsize=7.5)

arrow(ax, CX, 8.0, 3.0, 7.82, ACCENT2)
arrow(ax, CX, 8.0, 7.5, 7.82, ORANGE)

# outputs
draw_box(ax, 3.0, 6.7, 2.8, 0.58, 'Frac. Coords', '(B, 28, 3)',
         fc='#dcfce7', ec=ACCENT2, fontsize=8)
draw_box(ax, 7.5, 6.7, 2.8, 0.58, 'Lattice Params', '(B, 6)  a,b,c,α,β,γ',
         fc='#ffedd5', ec=ORANGE, fontsize=8)

arrow(ax, 3.0, 7.19, 3.0, 7.0)
arrow(ax, 7.5, 7.19, 7.5, 7.0)

# concat output
arrow(ax, 3.0, 6.41, 4.2, 5.92, ACCENT2)
arrow(ax, 7.5, 6.41, 5.8, 5.92, ORANGE)
dim_text(ax, CX, 6.1, 'concat')

draw_box(ax, CX, 5.62, 4.5, 0.6, 'Crystal Output', '90-dim  (30 × 3)',
         fc='#f3e8ff', ec=ORANGE, fontsize=9, lw=2)

# ── Discriminator (compact, below) ───────────────────────────────────────────
ax.text(CX, 4.85, '─── Discriminator ───', ha='center', color=RED,
        fontsize=8.5, fontweight='bold')
disc_steps = [
    ('Conv2d + LeakyReLU', '1 ch → 512 ch  kernel(1,3)', 4.4),
    ('Conv2d × 2 + LeakyReLU', '512 → 512 → 256 ch  kernel(1,1)', 3.75),
    ('Element AvgPool + Flatten', '→ 1280-dim  (+labels)', 3.1),
    ('Linear + LeakyReLU', '1280 → 1000 → 200 → 10', 2.45),
    ('Linear', 'Validity score  (B,)', 1.8),
]
for label, sub, cy in disc_steps:
    draw_box(ax, CX, cy, 5.2, 0.52, label, sub,
             fc='#fee2e2', ec=RED, fontsize=7.5)
    if cy < 4.4:
        arrow(ax, CX, cy + 0.52/2 + 0.52, CX, cy + 0.26 + 0.04, RED)

arrow(ax, CX, 5.32, CX, 4.68, ORANGE)
arrow(ax, CX, 5.32 - 0.32, CX, 4.68, ORANGE)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc='#dbeafe', ec=ACCENT1, label='Generator layers'),
    mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Atom output branch'),
    mpatches.Patch(fc='#ffedd5', ec=ORANGE,  label='Cell output branch'),
    mpatches.Patch(fc='#fee2e2', ec=RED,     label='Discriminator layers'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=7,
          facecolor='#ffffff', edgecolor=BORDER, labelcolor=WHITE,
          bbox_to_anchor=(0.98, 0.01))


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Quantum PQWGAN Generator
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_q
QX = 5.0

ax.text(QX, 17.3, 'Quantum PQWGAN', ha='center', va='center',
        color=ACCENT2, fontsize=14, fontweight='bold')
ax.text(QX, 16.85, 'Hybrid Quantum-Classical Crystal Generator (8 qubits)',
        ha='center', va='center', color=DIM, fontsize=8)

# ── Inputs ────────────────────────────────────────────────────────────────────
draw_box(ax, 2.5, 16.0, 2.2, 0.65, 'Noise z', 'dim = 64',
         fc='#dbeafe', ec=ACCENT2, fontsize=8)
draw_box(ax, 7.5, 16.0, 2.2, 0.65, 'Composition Labels', '28-dim  (one-hot)',
         fc='#dbeafe', ec=YELLOW, fontsize=7.5)

arrow(ax, 2.5, 15.67, 3.9, 15.15, ACCENT2)
arrow(ax, 7.5, 15.67, 6.1, 15.15, YELLOW)
dim_text(ax, QX, 15.35, 'concat  [64 + 28 = 92-dim]')

draw_box(ax, QX, 14.9, 3.8, 0.62, 'Linear', '92 → 8',
         fc='#dcfce7', ec=ACCENT2)
arrow(ax, QX, 14.59, QX, 14.14)

# ── HybridLayer × 4 ──────────────────────────────────────────────────────────
hybrid_top = 13.85
for i in range(4):
    cy = hybrid_top - i * 1.8
    # Outer hybrid block
    outer = FancyBboxPatch((QX - 3.2, cy - 0.95), 6.4, 1.6,
                           boxstyle='round,pad=0.05,rounding_size=0.3',
                           fc='#f0fdf4', ec=ACCENT2, lw=1.8, zorder=2,
                           linestyle='--' if i > 0 else '-')
    ax.add_patch(outer)
    ax.text(QX - 2.9, cy + 0.55, f'HybridLayer {i+1}', color=ACCENT2,
            fontsize=7, fontweight='bold', zorder=4)

    # Linear sub-box
    draw_box(ax, QX - 1.5, cy + 0.22, 1.8, 0.52, 'Linear',
             '8 → 8', fc='#dbeafe', ec=ACCENT1, fontsize=7)
    # BatchNorm
    draw_box(ax, QX + 0.5, cy + 0.22, 1.8, 0.52, 'BatchNorm1d',
             '+ Tanh', fc='#dbeafe', ec=ACCENT1, fontsize=7)
    # QuantumLayer
    draw_box(ax, QX, cy - 0.42, 5.0, 0.62, 'QuantumLayer  (PennyLane)',
             '8 qubits · RZ re-upload · StronglyEntanglingLayers(L=2) · PauliZ',
             fc='#ede9fe', ec=PURPLE, fontsize=7.5)

    arrow(ax, QX - 0.6, cy + 0.22, QX + 0.3 - 0.9, cy + 0.22, ACCENT1)
    arrow(ax, QX + 1.2, cy - 0.07, QX + 1.2, cy - 0.11, ACCENT1)
    arrow(ax, QX - 1.5, cy - 0.07, QX - 1.5, cy - 0.11, ACCENT1)

    if i > 0:
        arrow(ax, QX, cy + 0.95 + 0.85, QX, cy + 0.75 + 0.05, ACCENT2)

# arrow from last layer down
arrow(ax, QX, hybrid_top - 3 * 1.8 - 0.73, QX, hybrid_top - 3 * 1.8 - 1.0, ACCENT2)

# ── Quantum circuit detail inset ──────────────────────────────────────────────
circ_y_top = hybrid_top - 3 * 1.8 - 1.15   # ~6.58
circ_h = 1.85
circ_box = FancyBboxPatch((0.3, circ_y_top - circ_h), 9.4, circ_h,
                           boxstyle='round,pad=0.05,rounding_size=0.25',
                           fc='#faf5ff', ec=PURPLE, lw=1.2, zorder=2)
ax.add_patch(circ_box)
ax.text(QX, circ_y_top - 0.17, 'Quantum Circuit Detail  (1 layer shown)',
        ha='center', color=PURPLE, fontsize=7.5, fontweight='bold', zorder=4)

# Draw qubit lines (3 representative)
q_colors = [ACCENT2, ACCENT1, YELLOW]
for qi in range(3):
    qy = circ_y_top - 0.52 - qi * 0.42
    ax.plot([0.6, 9.1], [qy, qy], color='#94a3b8', lw=0.8, zorder=3)
    ax.text(0.55, qy, f'q{qi}', ha='right', va='center',
            color=DIM, fontsize=6, zorder=4)
    # RZ gate
    rz = FancyBboxPatch((0.75, qy - 0.14), 0.55, 0.28,
                         boxstyle='round,pad=0.02', fc='#ede9fe', ec=PURPLE,
                         lw=0.9, zorder=4)
    ax.add_patch(rz)
    ax.text(1.025, qy, 'RZ', ha='center', va='center',
            color=PURPLE, fontsize=5.5, fontweight='bold', zorder=5)
    # Hadamard
    h_gate = FancyBboxPatch((1.6, qy - 0.14), 0.45, 0.28,
                             boxstyle='round,pad=0.02', fc='#ccfbf1', ec=TEAL,
                             lw=0.9, zorder=4)
    ax.add_patch(h_gate)
    ax.text(1.825, qy, 'H', ha='center', va='center',
            color=TEAL, fontsize=5.5, fontweight='bold', zorder=5)
    # Entangle block marker
    ent = FancyBboxPatch((2.3, qy - 0.14), 4.5, 0.28,
                          boxstyle='round,pad=0.02', fc='#ede9fe', ec=PURPLE,
                          lw=0.9, zorder=4)
    ax.add_patch(ent)
    ax.text(4.55, qy, 'StronglyEntanglingLayers (CZ + Rot)',
            ha='center', va='center', color=PURPLE, fontsize=5.5, zorder=5)
    # PauliZ measure
    mbox = FancyBboxPatch((7.0, qy - 0.14), 0.7, 0.28,
                           boxstyle='round,pad=0.02', fc='#ffedd5', ec=ORANGE,
                           lw=0.9, zorder=4)
    ax.add_patch(mbox)
    ax.text(7.35, qy, '⟨Z⟩', ha='center', va='center',
            color=ORANGE, fontsize=6, fontweight='bold', zorder=5)

ax.text(4.55, circ_y_top - circ_h + 0.12, '⋮  ×8 qubits total  ⋮',
        ha='center', color=DIM, fontsize=6, zorder=4)

# ── Post-quantum trunk ────────────────────────────────────────────────────────
trunk_y = circ_y_top - circ_h - 0.12
arrow(ax, QX, trunk_y, QX, trunk_y - 0.25, ACCENT2)
dim_text(ax, QX, trunk_y - 0.12, '8-dim PauliZ expectations')

trunk_steps = [
    ('Linear + LeakyReLU', '8 → 128', trunk_y - 0.38),
    ('Linear + LeakyReLU', '128 → 256', trunk_y - 0.38 - 0.65),
]
for lbl, sub, cy in trunk_steps:
    draw_box(ax, QX, cy, 4.0, 0.52, lbl, sub,
             fc='#dcfce7', ec=ACCENT2, fontsize=8)

arrow(ax, QX, trunk_y - 0.38 + 0.26, QX, trunk_y - 0.38 - 0.39 + 0.26 - 0.26, ACCENT2)

split2_y = trunk_steps[-1][2] - 0.38
arrow(ax, QX, split2_y, QX, split2_y - 0.04, ACCENT2)
dim_text(ax, QX, split2_y - 0.12, '256-dim trunk features')

# atom head
draw_box(ax, 2.5, split2_y - 0.55, 3.2, 0.55,
         'Atom Head', '256→512→256→84  +  Sigmoid',
         fc='#dcfce7', ec=ACCENT2, fontsize=7.5)
# cell head
draw_box(ax, 7.5, split2_y - 0.55, 3.2, 0.55,
         'Cell Head', '256→64→6  +  Sigmoid',
         fc='#ffedd5', ec=ORANGE, fontsize=7.5)

arrow(ax, QX, split2_y - 0.16, 2.5, split2_y - 0.28, ACCENT2)
arrow(ax, QX, split2_y - 0.16, 7.5, split2_y - 0.28, ORANGE)

out_y = split2_y - 0.55 - 0.44
draw_box(ax, 2.5, out_y, 3.2, 0.52, 'Frac. Coords', '84-dim  (28 atoms × 3)',
         fc='#dcfce7', ec=ACCENT2, fontsize=8)
draw_box(ax, 7.5, out_y, 3.2, 0.52, 'Lattice Params', '6-dim  a,b,c,α,β,γ',
         fc='#ffedd5', ec=ORANGE, fontsize=8)

arrow(ax, 2.5, split2_y - 0.55 - 0.27, 2.5, out_y + 0.26, ACCENT2)
arrow(ax, 7.5, split2_y - 0.55 - 0.27, 7.5, out_y + 0.26, ORANGE)

concat2_y = out_y - 0.42
arrow(ax, 2.5, out_y - 0.26, 4.2, concat2_y + 0.15, ACCENT2)
arrow(ax, 7.5, out_y - 0.26, 5.8, concat2_y + 0.15, ORANGE)
dim_text(ax, QX, concat2_y + 0.28, 'concat')

draw_box(ax, QX, concat2_y, 4.5, 0.55, 'Crystal Output', '90-dim  (30 × 3)',
         fc='#f3e8ff', ec=ORANGE, fontsize=9, lw=2)

# ── Critic (quantum) ──────────────────────────────────────────────────────────
critic_top = concat2_y - 0.42
ax.text(QX, critic_top - 0.05, '─── Critic (Wasserstein) ───', ha='center',
        color=RED, fontsize=8.5, fontweight='bold')

crit_steps = [
    ('concat [crystal + labels]', '90 + 28 = 118-dim', critic_top - 0.45),
    ('Linear + LeakyReLU', '118 → 512 → 256', critic_top - 1.05),
    ('Linear', 'Wasserstein score  (B,)', critic_top - 1.65),
]
for lbl, sub, cy in crit_steps:
    draw_box(ax, QX, cy, 5.2, 0.48, lbl, sub,
             fc='#fee2e2', ec=RED, fontsize=7.5)
    if cy < critic_top - 0.45:
        arrow(ax, QX, cy + 0.48/2 + 0.48, QX, cy + 0.24 + 0.04, RED)

arrow(ax, QX, concat2_y - 0.28, QX, critic_top - 0.21, ORANGE)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Quantum Generator layers'),
    mpatches.Patch(fc='#ede9fe', ec=PURPLE,  label='Quantum circuit (PennyLane)'),
    mpatches.Patch(fc='#dcfce7', ec=ACCENT2, label='Atom output branch'),
    mpatches.Patch(fc='#ffedd5', ec=ORANGE,  label='Cell output branch'),
    mpatches.Patch(fc='#fee2e2', ec=RED,     label='Critic (Wasserstein) layers'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=7,
          facecolor='#ffffff', edgecolor=BORDER, labelcolor=WHITE,
          bbox_to_anchor=(0.98, 0.01))


# ── Titles + watermarks ───────────────────────────────────────────────────────
fig_cls.text(0.5, 0.985, 'Classical CWGAN — Crystal Generator Architecture',
             ha='center', va='top', color=ACCENT1, fontsize=15, fontweight='bold')
fig_cls.text(0.5, 0.965, 'Mg-Mn-O Crystal Generation  ·  90-dim output (30×3)',
             ha='center', va='top', color=DIM, fontsize=9)
fig_cls.text(0.98, 0.005, 'QGAN-QIREN 2024', ha='right', color='#94a3b8', fontsize=7)

fig_q.text(0.5, 0.985, 'Quantum PQWGAN — Crystal Generator Architecture',
           ha='center', va='top', color=ACCENT2, fontsize=15, fontweight='bold')
fig_q.text(0.5, 0.965, 'Mg-Mn-O Crystal Generation  ·  90-dim output (30×3)  ·  8 qubits',
           ha='center', va='top', color=DIM, fontsize=9)
fig_q.text(0.98, 0.005, 'QGAN-QIREN 2024', ha='right', color='#94a3b8', fontsize=7)

# ── Save ──────────────────────────────────────────────────────────────────────
cls_path = os.path.join(OUT_DIR, 'architecture_classical.png')
q_path   = os.path.join(OUT_DIR, 'architecture_quantum.png')

fig_cls.savefig(cls_path, dpi=180, bbox_inches='tight', facecolor=BG)
fig_q.savefig(q_path,   dpi=180, bbox_inches='tight', facecolor=BG)
plt.close('all')
print(f'Saved: {cls_path}')
print(f'Saved: {q_path}')
