import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "data/opening_analysis.json"
OUTPUT_DIR = "poster_figures"
DPI = 300
FONT_SIZE = 10
USE_LATEX = False   # set to True if you have LaTeX + dvipng installed

# Subset selection: None for all, or a list, or "top_N_by_metric" / "bottom_N_by_metric"
SUBSET = None   # e.g., ["Ruy Lopez", "Sicilian Defense"] or "top_10_by_mean_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
with open(INPUT_FILE) as f:
    data = json.load(f)

openings = {name: values for name, values in data.items() if not name.startswith("_")}
ALL_NAMES = list(openings.keys())
NUM_OPENINGS = len(openings)

# =============================================================================
# STYLING – Research-grade colours, no seaborn style
# =============================================================================
# Use LaTeX if available
plt.rcParams['text.usetex'] = USE_LATEX
plt.rcParams['font.family'] = 'sans-serif' if not USE_LATEX else 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']
if USE_LATEX:
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = FONT_SIZE

# Clear matplotlib style (no seaborn)
plt.style.use('default')
# We'll use a clean white background with subtle grid lines where needed

# Professional colour palette (ColorBrewer / viridis inspired)
MAIN_COLOR = '#2E4A62'        # dark slate blue
SECONDARY_COLOR = '#1B9E77'   # teal
ACCENT_COLOR = '#7570B3'      # purple
SOFT_ORANGE = "#02B9D9"       # warm orange (can be replaced if disliked)
LIGHT_BLUE = '#AEC7E8'        # light blue for top/bottom

# For qualitative multiple categories (piece importance, grouped bars)
QUAL_COLORS = ['#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B',
               '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', '#FF7F0E']

# Set default color cycle for matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[MAIN_COLOR, SECONDARY_COLOR,
                                                    ACCENT_COLOR, SOFT_ORANGE,
                                                    '#8C564B', '#E377C2'])

# Default colormap for heatmaps
plt.rcParams['image.cmap'] = 'viridis'

# DPI
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI

# =============================================================================
# METRIC DEFINITIONS
# =============================================================================
METRICS = {
    'mean_shapley_entropy': ('Shapley Entropy', 'Entropy'),
    'mean_banzhaf_entropy': ('Banzhaf Entropy', 'Entropy'),
    'mean_eval': ('Evaluation', 'Centipawns'),
    'mean_mobility': ('Mobility', 'Legal Moves'),
    'mean_top5_shapley': ('Top-5 Shapley Concentration', 'Fraction'),
    'mean_top5_banzhaf': ('Top-5 Banzhaf Concentration', 'Fraction'),
}

# =============================================================================
# SUBSET SELECTION
# =============================================================================
def select_openings(subset_spec):
    if subset_spec is None:
        return ALL_NAMES
    if isinstance(subset_spec, list):
        return [name for name in subset_spec if name in openings]
    if isinstance(subset_spec, str) and subset_spec.startswith(('top_', 'bottom_')):
        parts = subset_spec.split('_')
        if len(parts) >= 3:
            try:
                n = int(parts[1])
                metric_key = '_'.join(parts[3:])
                if metric_key not in METRICS:
                    raise ValueError(f"Unknown metric: {metric_key}")
                sorted_names = sorted(ALL_NAMES, key=lambda x: openings[x].get(metric_key, 0))
                if subset_spec.startswith('top_'):
                    return sorted_names[-n:][::-1]
                else:
                    return sorted_names[:n]
            except (ValueError, IndexError):
                pass
    return ALL_NAMES

SELECTED_NAMES = select_openings(SUBSET)
SELECTED_DATA = {name: openings[name] for name in SELECTED_NAMES}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def opening_figure(height_scale=0.45, min_height=6):
    n = len(SELECTED_NAMES)
    height = max(min_height, height_scale * n)
    return plt.figure(figsize=(12, height))

def add_regression_line(x, y, ax=None, label_prefix="", color=SOFT_ORANGE):
    if ax is None:
        ax = plt.gca()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_range = np.linspace(min(x), max(x), 100)
    ax.plot(x_range, intercept + slope * x_range, color=color, lw=2,
            label=f'{label_prefix}$y={slope:.2f}x+{intercept:.2f}$\n$R^2={r_value**2:.3f}$, $p={p_value:.3f}$')
    return r_value**2, p_value

# -----------------------------------------------------------------------------
# 1. Bar chart for any metric
# -----------------------------------------------------------------------------
def plot_bar_metric(metric_key, title=None, xlabel=None, ascending=True,
                    color=MAIN_COLOR, filename=None):
    names = sorted(SELECTED_NAMES, key=lambda x: SELECTED_DATA[x].get(metric_key, 0),
                   reverse=not ascending)
    values = [SELECTED_DATA[name].get(metric_key, 0) for name in names]
    fig = opening_figure()
    ax = fig.add_subplot(111)
    ax.barh(names, values, color=color)
    ax.set_xlabel(xlabel or metric_key.replace('_', ' ').title())
    ax.set_title(title or f"{metric_key.replace('_', ' ').title()} by Opening")
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout(pad=1.5)
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 2. Scatter with regression
# -----------------------------------------------------------------------------
def plot_scatter_regression(x_metric, y_metric, xlabel=None, ylabel=None,
                            title=None, filename=None):
    x_vals = [SELECTED_DATA[name].get(x_metric, 0) for name in SELECTED_NAMES]
    y_vals = [SELECTED_DATA[name].get(y_metric, 0) for name in SELECTED_NAMES]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x_vals, y_vals, alpha=0.7, s=60, c=SECONDARY_COLOR)
    r2, p = add_regression_line(x_vals, y_vals, ax, color=SOFT_ORANGE)
    ax.set_xlabel(xlabel or x_metric.replace('_', ' ').title())
    ax.set_ylabel(ylabel or y_metric.replace('_', ' ').title())
    ax.set_title(title or f"{ylabel} vs {xlabel}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 3. Chess-board heatmap (improved)
# -----------------------------------------------------------------------------
def plot_chess_heatmap(metric='square_heatmap_shapley', 
                       title="Average Shapley Importance by Square",
                       filename="chess_heatmap.png", power_scale=1.0):
    """Heatmap with chessboard background and clear color mapping."""
    heatmap = np.zeros(64)
    count = 0
    for name in SELECTED_NAMES:
        arr = SELECTED_DATA[name].get(metric)
        if arr is not None and len(arr) == 64:
            heatmap += np.array(arr)
            count += 1
    if count == 0:
        print("Warning: No square heatmap data found. Skipping.")
        return
    heatmap /= count

    if power_scale != 1.0:
        heatmap = np.sign(heatmap) * np.abs(heatmap) ** power_scale

    heatmap = heatmap.reshape(8, 8)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Chessboard background
    for i in range(8):
        for j in range(8):
            color = '#F0D9B5' if (i + j) % 2 == 0 else '#B58863'
            rect = Rectangle((j, 7 - i), 1, 1, facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # Overlay heatmap with full opacity
    im = ax.imshow(heatmap[::-1], cmap='viridis', alpha=1.0, 
                   extent=[0, 8, 0, 8], interpolation='bilinear')
    # Numeric annotations
    vmin, vmax = heatmap.min(), heatmap.max()
    norm = plt.Normalize(vmin, vmax)
    for i in range(8):
        for j in range(8):
            val = heatmap[7 - i, j]
            color = 'black' if norm(val) < 0.6 else 'white'
            ax.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_xticklabels(list('abcdefgh'))
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(range(8, 0, -1))
    ax.set_xlabel('File')
    ax.set_ylabel('Rank')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Mean Shapley Value')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 4. Piece importance (corrected per‑piece averaging)
# -----------------------------------------------------------------------------
def plot_piece_importance(filename="piece_importance.png"):
    piece_config = {
        'white_pawn': 8, 'black_pawn': 8,
        'white_knight': 2, 'black_knight': 2,
        'white_bishop': 2, 'black_bishop': 2,
        'white_rook': 2, 'black_rook': 2,
        'white_queen': 1, 'black_queen': 1,
    }
    piece_names = list(piece_config.keys())
    display_names = ['W Pawn', 'B Pawn', 'W Knight', 'B Knight',
                     'W Bishop', 'B Bishop', 'W Rook', 'B Rook',
                     'W Queen', 'B Queen']

    means = []
    for piece in piece_names:
        vals = []
        for name in SELECTED_NAMES:
            total = SELECTED_DATA[name].get('piece_totals_shapley', {}).get(piece, 0)
            per_piece = total / piece_config[piece] if piece_config[piece] > 0 else 0
            vals.append(per_piece)
        means.append(np.mean(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [MAIN_COLOR if 'white' in p else SECONDARY_COLOR for p in piece_names]
    ax.bar(display_names, means, color=colors)
    ax.set_ylabel('Mean Shapley value per piece')
    ax.set_title('Average Piece Importance (per piece, over selected openings)')
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 5. Parallel coordinates (replaces noisy radar)
# -----------------------------------------------------------------------------
def plot_parallel_coordinates(metric_keys=None, top_n=20, filename="parallel.png"):
    if metric_keys is None:
        metric_keys = ['mean_shapley_entropy', 'mean_banzhaf_entropy',
                       'mean_eval', 'mean_mobility', 'mean_top5_shapley']
    top_names = sorted(SELECTED_NAMES, key=lambda x: SELECTED_DATA[x].get('mean_shapley_entropy', 0), reverse=True)[:top_n]
    df = pd.DataFrame({name: [SELECTED_DATA[name].get(k, 0) for k in metric_keys]
                       for name in top_names}).T
    df.columns = metric_keys
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-9)
    df_norm['opening'] = df_norm.index

    from pandas.plotting import parallel_coordinates
    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(df_norm, 'opening', ax=ax, colormap='viridis', alpha=0.7)
    ax.set_xticklabels([METRICS.get(k, (k,))[0] for k in metric_keys], rotation=45, ha='right')
    ax.set_title(f'Parallel Coordinates for Top {top_n} Openings (by Shapley Entropy)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 6. Correlation matrix
# -----------------------------------------------------------------------------
def plot_correlation_matrix(metric_keys=None, filename="correlation_matrix.png"):
    if metric_keys is None:
        metric_keys = list(METRICS.keys())
    df = pd.DataFrame({name: [SELECTED_DATA[name].get(k, 0) for k in metric_keys]
                       for name in SELECTED_NAMES}).T
    df.columns = metric_keys
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Use a custom diverging colormap (RdBu_r) with a softer look
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels([METRICS.get(k, (k,))[0] for k in corr.columns], rotation=45, ha='right')
    ax.set_yticklabels([METRICS.get(k, (k,))[0] for k in corr.columns], rotation=0)
    ax.set_title('Correlation Matrix of Opening Metrics')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 7. Top & bottom N (improved with two shades)
# -----------------------------------------------------------------------------
def plot_top_bottom(metric_key, n=10, filename=None):
    sorted_names = sorted(SELECTED_NAMES, key=lambda x: SELECTED_DATA[x].get(metric_key, 0))
    bottom = sorted_names[:n]
    top = sorted_names[-n:][::-1]
    selected = bottom + top
    # Use dark blue for top, light blue for bottom
    colors = [LIGHT_BLUE] * n + [MAIN_COLOR] * n
    values = [SELECTED_DATA[name].get(metric_key, 0) for name in selected]
    display_names = [f"{name[:30]}\n({i+1})" for i, name in enumerate(selected)]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(selected))))
    ax.barh(display_names, values, color=colors)
    ax.set_xlabel(METRICS.get(metric_key, (metric_key, ''))[1])
    ax.set_title(f'Top {n} (dark) and Bottom {n} (light) by {METRICS.get(metric_key, (metric_key,))[0]}')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 8. Summary panel (2x2)
# -----------------------------------------------------------------------------
def plot_summary_panel(metrics=None, filename="summary_panel.png"):
    if metrics is None:
        metrics = ['mean_shapley_entropy', 'mean_eval', 'mean_mobility', 'mean_top5_shapley']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        names = sorted(SELECTED_NAMES, key=lambda x: SELECTED_DATA[x].get(metric, 0), reverse=True)
        values = [SELECTED_DATA[name].get(metric, 0) for name in names]
        ax = axes[idx]
        ax.barh(names, values, color=MAIN_COLOR)
        ax.set_title(METRICS.get(metric, (metric, ''))[0])
        ax.set_xlabel(METRICS.get(metric, (metric, ''))[1])
        ax.grid(True, alpha=0.3, axis='x')
        if len(names) > 30:
            ax.set_yticklabels(names, fontsize=6)
    plt.suptitle('Opening Metrics Summary', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 9. Grouped piece importance for top N openings
# -----------------------------------------------------------------------------
def plot_grouped_piece_importance(top_n=5, filename="grouped_piece_importance.png"):
    top_names = sorted(SELECTED_NAMES, key=lambda x: SELECTED_DATA[x].get('mean_shapley_entropy', 0), reverse=True)[:top_n]
    piece_config = {
        'white_pawn': 8, 'black_pawn': 8,
        'white_knight': 2, 'black_knight': 2,
        'white_bishop': 2, 'black_bishop': 2,
        'white_rook': 2, 'black_rook': 2,
        'white_queen': 1, 'black_queen': 1,
    }
    piece_names = list(piece_config.keys())
    display_names = ['W Pawn', 'B Pawn', 'W Knight', 'B Knight',
                     'W Bishop', 'B Bishop', 'W Rook', 'B Rook',
                     'W Queen', 'B Queen']
    df = pd.DataFrame(index=top_names, columns=display_names)
    for name in top_names:
        for piece, disp in zip(piece_names, display_names):
            total = SELECTED_DATA[name].get('piece_totals_shapley', {}).get(piece, 0)
            df.loc[name, disp] = total / piece_config[piece] if piece_config[piece] > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, color=QUAL_COLORS[:len(display_names)])
    ax.set_title(f'Piece Importance (per piece) for Top {top_n} Openings (by Shapley Entropy)')
    ax.set_xlabel('Opening')
    ax.set_ylabel('Shapley Value')
    ax.legend(loc='best', ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 10. Histograms of each metric
# -----------------------------------------------------------------------------
def plot_histograms(metrics=None, filename="histograms.png"):
    if metrics is None:
        metrics = list(METRICS.keys())
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n))
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, metrics):
        vals = [SELECTED_DATA[name].get(key, 0) for name in SELECTED_NAMES]
        ax.hist(vals, bins=20, edgecolor='black', alpha=0.7, color=MAIN_COLOR)
        ax.set_title(METRICS.get(key, (key,))[0])
        ax.set_xlabel(METRICS.get(key, (key, ''))[1])
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Distribution of Opening Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 11. Boxplots
# -----------------------------------------------------------------------------
def plot_boxplots(metrics=None, filename="boxplots.png"):
    if metrics is None:
        metrics = list(METRICS.keys())
    data = {METRICS.get(k, (k,))[0]: [SELECTED_DATA[name].get(k, 0) for name in SELECTED_NAMES]
            for k in metrics}
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use our main color for boxes
    bp = df.boxplot(ax=ax, rot=45, grid=True, patch_artist=True,
                    boxprops=dict(facecolor=MAIN_COLOR, color=MAIN_COLOR),
                    whiskerprops=dict(color=MAIN_COLOR),
                    capprops=dict(color=MAIN_COLOR),
                    medianprops=dict(color='white'))
    ax.set_ylabel('Value')
    ax.set_title('Boxplots of Opening Metrics')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 12. Scatter matrix (pairplot) – using seaborn but with our style
# -----------------------------------------------------------------------------
def plot_pairplot(metrics=None, filename="pairplot.png"):
    if metrics is None:
        metrics = ['mean_shapley_entropy', 'mean_eval', 'mean_mobility', 'mean_top5_shapley']
    df = pd.DataFrame({name: [SELECTED_DATA[name].get(k, 0) for k in metrics]
                       for name in SELECTED_NAMES}).T
    df.columns = [METRICS.get(k, (k,))[0] for k in metrics]
    # Temporarily set seaborn style to 'white' (clean) and use our colors
    with sns.axes_style("white"):
        g = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.6, 'color': SECONDARY_COLOR},
                         diag_kws={'color': MAIN_COLOR})
    g.fig.suptitle('Pairwise Relationships between Metrics', y=1.02)
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(g.fig)

# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================
print(f"Generating figures for {len(SELECTED_NAMES)} openings (out of {NUM_OPENINGS})")

# 1. Bar charts
for key, (label, unit) in METRICS.items():
    plot_bar_metric(key, title=label, xlabel=unit, filename=f"bar_{key}.png")

# 2. Scatter plots
plot_scatter_regression('mean_shapley_entropy', 'mean_eval',
                        xlabel='Shapley Entropy', ylabel='Evaluation (cp)',
                        title='Entropy vs Evaluation', filename='scatter_entropy_eval.png')
plot_scatter_regression('mean_mobility', 'mean_shapley_entropy',
                        xlabel='Mobility', ylabel='Shapley Entropy',
                        title='Mobility vs Entropy', filename='scatter_mobility_entropy.png')
plot_scatter_regression('mean_top5_shapley', 'mean_shapley_entropy',
                        xlabel='Top-5 Concentration', ylabel='Shapley Entropy',
                        title='Concentration vs Entropy', filename='scatter_concentration_entropy.png')

# 3. Chess heatmap (power_scale=0.5 to enhance contrast if needed)
plot_chess_heatmap(power_scale=1.0)   # try 0.5 for stronger contrast

# 4. Piece importance
plot_piece_importance()

# 5. Parallel coordinates
plot_parallel_coordinates(top_n=20)

# 6. Correlation matrix
plot_correlation_matrix()

# 7. Top / bottom
plot_top_bottom('mean_shapley_entropy', n=10, filename='top_bottom_entropy.png')
plot_top_bottom('mean_eval', n=10, filename='top_bottom_eval.png')

# 8. Summary panel
plot_summary_panel()

# 9. Grouped piece importance
plot_grouped_piece_importance(top_n=5)

# 10. Histograms
plot_histograms()

# 11. Boxplots
plot_boxplots()

# 12. Pairplot
plot_pairplot()

print(f"All figures saved to {OUTPUT_DIR}")