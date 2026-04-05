"""
QINR-QGAN — clean, IEEE-style architecture diagram.
Layout mirrors the style of the reference GAN figure:
  Top row:  Z / C_gen  → Generator → Generated Feature  ↘
                                                          ✕ → Critic → Wasserstein Distance
  Bot row:  Real Crystal DB          → Real Feature      ↗
                                                          ↘ Classifier → Ĉ_gen / Ĉ_real

A single gradient-feedback dashed arc closes the training loop.
No crossing decorative lines, no sub-boxes, no clutter.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
W, H = 15, 8.0
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colour palette ────────────────────────────────────────────────────────────
ORANGE  = '#E67E22'   # Generator
BLUE    = '#2980B9'   # Critic
GREY    = '#7F8C8D'   # Classifier / Q-Head
GREEN_L = '#A9DFBF'   # Generated feature (light)
GREEN_D = '#1E8449'   # Real feature (dark)
PURPLE  = '#8E44AD'   # gradient loop
RED     = '#E74C3C'   # re-uploading
TEXT    = '#1A252F'

# ── Helper: rounded box ───────────────────────────────────────────────────────
def rbox(cx, cy, w, h, title, subtitle='', color='#3498DB', tcolor='white', fs=11, fss=8.5):
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0,rounding_size=0.15',
                       fc=color, ec='none', zorder=3)
    ax.add_patch(p)
    if subtitle:
        ax.text(cx, cy + h*0.16, title, fontsize=fs, fontweight='bold',
                color=tcolor, ha='center', va='center', zorder=4)
        ax.text(cx, cy - h*0.22, subtitle, fontsize=fss, color=tcolor,
                ha='center', va='center', zorder=4, alpha=0.90)
    else:
        ax.text(cx, cy, title, fontsize=fs, fontweight='bold',
                color=tcolor, ha='center', va='center', zorder=4)

# ── Helper: straight arrow ────────────────────────────────────────────────────
def arr(x0, y0, x1, y1, col=TEXT, lw=2.0, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                linestyle=ls, mutation_scale=14), zorder=5)

def line(pts, col=TEXT, lw=1.8, ls='-'):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=5, solid_capstyle='round')

def arrowhead(x, y, d='right', col=TEXT, lw=2.0):
    dx = dy = 0
    if d == 'right': dx = 0.001
    elif d == 'left': dx = -0.001
    elif d == 'down': dy = -0.001
    elif d == 'up':   dy = 0.001
    ax.annotate('', xy=(x, y), xytext=(x-dx, y-dy),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw, mutation_scale=14), zorder=5)

def label(x, y, txt, fs=8.5, col='#555555', ha='center', va='center', italic=True):
    ax.text(x, y, txt, fontsize=fs, color=col, ha=ha, va=va,
            style='italic' if italic else 'normal', zorder=6)

# =============================================================================
# TITLE
# =============================================================================
ax.text(W/2, H - 0.28,
        'QINR-QGAN: Quantum-Hybrid GAN for Crystal Structure Generation',
        fontsize=12, fontweight='bold', color=TEXT, ha='center', va='center')
ax.plot([0.4, W-0.4], [H-0.52, H-0.52], color='#BDC3C7', lw=0.8)

# =============================================================================
# TOP ROW  (y-centre = 5.40)
# =============================================================================
TY = 5.50     # top row y
# Generator box:  cx=3.30, w=2.60  → left edge x=2.00, right edge x=4.60
GEN_CX   = 3.30
GEN_W    = 2.60
GEN_LEFT = GEN_CX - GEN_W/2   # 2.00

# ── Inputs (x=0.55, well left of generator) ──────────────────────────────────
ax.text(0.65, TY + 0.40, 'Z',         fontsize=12, fontweight='bold', color=TEXT, ha='center')
ax.text(0.65, TY - 0.40, '$C_{gen}$', fontsize=12, fontweight='bold', color=TEXT, ha='center')
arr(0.95, TY + 0.40, GEN_LEFT, TY + 0.30)
arr(0.95, TY - 0.40, GEN_LEFT, TY - 0.30)

# ── Generator ─────────────────────────────────────────────────────────────────
rbox(GEN_CX, TY, GEN_W, 1.80,
     'Generator',
     'Linear → BN → QuantumCircuit\n(×3 HybridLayer) → MLP → Sigmoid',
     color=ORANGE)

# ── Generated Feature ─────────────────────────────────────────────────────────
arr(GEN_CX + GEN_W/2, TY, 5.40, TY)
rbox(6.20, TY, 1.90, 1.10,
     'Generated\nFeature',
     '($\\tilde{x}$, $C_{gen}$)',
     color=GREEN_L, tcolor=TEXT)

# ── Data re-uploading: small curved arrow ON TOP of generator (above box) ─────
# Arc starts and ends at top-centre of generator, so it never overlaps text
ax.annotate('', xy=(GEN_CX + 0.55, TY + 0.90), xytext=(GEN_CX - 0.55, TY + 0.90),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                            connectionstyle='arc3,rad=-0.55', linestyle='--'), zorder=4)
label(GEN_CX, TY + 1.32, 'data re-uploading', fs=7, col=RED)

# =============================================================================
# BOTTOM ROW  (y-centre = 2.10)
# =============================================================================
BY = 2.20

# ── Real data ─────────────────────────────────────────────────────────────────
rbox(GEN_CX, BY, GEN_W, 1.10,
     'Real Crystal',
     'Mg-Mn-O  (90-d coords + 28-d label)',
     color='#AED6F1', tcolor=TEXT)

# ── Real Feature ──────────────────────────────────────────────────────────────
arr(GEN_CX + GEN_W/2, BY, 5.40, BY)
rbox(6.20, BY, 1.90, 1.10,
     'Real\nFeature',
     '($x$, $C_{real}$)',
     color=GREEN_D)

# =============================================================================
# CROSS ARROWS — carefully routed through a clear corridor
# Feature right edges at x=7.15 (6.20+0.95)
# Network left edges  at x=8.50 (9.50-1.00)
# Cross corridor: x = 7.15 to 8.50  (1.35 wide, nothing there)
# =============================================================================
GFR = 6.20 + 0.95   # 7.15  generated feature right edge
RFR = 6.20 + 0.95   # 7.15  real feature right edge
NETLEFT = 9.50 - 1.00  # 8.50  network left edge

CX = 9.50   # Critic   centre-x
QX = 9.50   # Q-Head   centre-x

# Generated → Critic (straight horizontal)  — no diagonal needed
arr(GFR, TY, NETLEFT, TY, col=TEXT)

# Real → Critic (diagonal: goes from RFR,BY up to NETLEFT,TY)
# This crosses the Generated→Classifier line — that X is intentional
line([(RFR, BY), (NETLEFT, TY)], col=TEXT)
arrowhead(NETLEFT, TY, 'right', col=TEXT)

# Generated → Classifier (diagonal: GFR,TY down to NETLEFT,BY)
line([(GFR, TY), (NETLEFT, BY)], col=TEXT)
arrowhead(NETLEFT, BY, 'right', col=TEXT)

# Real → Classifier (straight horizontal)
arr(RFR, BY, NETLEFT, BY, col=TEXT)

# ── Critic ────────────────────────────────────────────────────────────────────
rbox(CX, TY, 2.10, 1.80,
     'Critic  D',
     'FC(118→512)\nFC(512→256)\nFC(256→1)',
     color=BLUE)

# ── Classifier / Q-Head ───────────────────────────────────────────────────────
rbox(QX, BY, 2.10, 1.80,
     'Q-Head  Q',
     'Slot: BCE loss\n(Mg/Mn/O occupancy)\nCount: CE loss',
     color=GREY)

# =============================================================================
# OUTPUT LABELS (right side)
# =============================================================================
# Output labels start after critic/qhead right edge: 9.50+1.05 = 10.55
OX = 10.55

# Wasserstein Distance
arr(CX + 1.05, TY, OX, TY)
ax.text(OX + 0.15, TY + 0.30, 'Wasserstein Distance',
        fontsize=9.0, fontweight='bold', color=TEXT, ha='left', va='center')
ax.text(OX + 0.15, TY - 0.20, '$D(\\tilde{x}) - D(x)$',
        fontsize=9.0, color=TEXT, ha='left', va='center')

# Composition predictions
arr(QX + 1.05, BY, OX, BY)
ax.text(OX + 0.15, BY + 0.32, '$\\hat{C}_{gen}$  (for $\\tilde{x}$)',
        fontsize=9.0, fontweight='bold', color=TEXT, ha='left', va='center')
ax.text(OX + 0.15, BY - 0.22, '$\\hat{C}_{real}$  (for $x$)',
        fontsize=9.0, fontweight='bold', color=TEXT, ha='left', va='center')

# =============================================================================
# GRADIENT FEEDBACK LOOP
# Route: bottom of Critic/QHead area → right margin → bottom margin → left
# margin → up to Generator input.  All segments are in CLEAR space.
# Key constraint: Generator left edge = GEN_LEFT = 2.00
#                 Loop left vertical at x = 0.45  (safely left of everything)
# =============================================================================
LOOP_X_R = 14.50          # right margin (outside all boxes)
LOOP_Y_B = 0.45           # bottom margin (well below all boxes whose min-y ≈ 1.65)
LOOP_X_L = 0.45           # left margin  (Generator left edge = 2.00, so 0.45 is clear)
LOOP_ENTRY_Y = TY         # enter Generator at its vertical centre

# 1. Horizontal from below-outputs rightward to right margin
line([(OX, BY - 0.90), (LOOP_X_R, BY - 0.90)], col=PURPLE, lw=1.4, ls='--')
# 2. Down to bottom margin
line([(LOOP_X_R, BY - 0.90), (LOOP_X_R, LOOP_Y_B)], col=PURPLE, lw=1.4, ls='--')
# 3. Left along bottom
line([(LOOP_X_R, LOOP_Y_B), (LOOP_X_L, LOOP_Y_B)], col=PURPLE, lw=1.4, ls='--')
# 4. Up left margin to Generator entry height
line([(LOOP_X_L, LOOP_Y_B), (LOOP_X_L, LOOP_ENTRY_Y)], col=PURPLE, lw=1.4, ls='--')
# 5. Arrow right into Generator left edge
line([(LOOP_X_L, LOOP_ENTRY_Y), (GEN_LEFT, LOOP_ENTRY_Y)], col=PURPLE, lw=1.4, ls='--')
arrowhead(GEN_LEFT, LOOP_ENTRY_Y, 'right', col=PURPLE, lw=1.4)

# Label sits in the clear bottom strip
label(7.50, LOOP_Y_B + 0.22,
      'Backprop  (Adam, n_critic = 5,  LR decay ×0.99 / 10 ep)',
      fs=7.5, col=PURPLE)

# =============================================================================
# LEGEND  (bottom-left)
# =============================================================================
items = [
    (ORANGE,  'Hybrid Quantum Generator'),
    (BLUE,    'Classical Critic'),
    (GREY,    'Q-Head Classifier'),
    (GREEN_L, 'Generated Feature'),
    (GREEN_D, 'Real Feature'),
]
LX0, LY0 = 0.25, 2.80
ax.text(LX0, LY0 + 0.30, 'Legend', fontsize=8, fontweight='bold', color=TEXT)
for k, (col, lbl) in enumerate(items):
    cy = LY0 - k * 0.42
    p = FancyBboxPatch((LX0, cy - 0.14), 0.30, 0.28,
                       boxstyle='round,pad=0.02',
                       fc=col, ec='none', zorder=7)
    ax.add_patch(p)
    ax.text(LX0 + 0.40, cy, lbl, fontsize=7.0, color=TEXT, va='center', zorder=8)

# Dashed legend entries
for k, (col, lbl) in enumerate([(RED, 'Data re-uploading loop'), (PURPLE, 'Gradient update')]):
    cy = LY0 - (len(items) + k) * 0.42
    ax.plot([LX0, LX0+0.30], [cy, cy], color=col, lw=1.5, ls='--')
    ax.text(LX0 + 0.40, cy, lbl, fontsize=7.0, color=TEXT, va='center')

# =============================================================================
# SAVE
# =============================================================================
out = ('C:/Users/Adminb/OneDrive/Documents/Projects/qgan/'
       'QINR-QGAN/workflow_diagram.png')
plt.tight_layout(pad=0)
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out}')
