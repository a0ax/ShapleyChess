import json
import os

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "data/opening_analysis.json"

OUTPUT_DIR = "poster_figures"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

with open(INPUT_FILE) as f:
    data = json.load(f)


# ---------------------------------------------------------
# Remove metadata
# ---------------------------------------------------------

openings = {

    name: values

    for name, values in data.items()

    if not name.startswith("_")

}

NUM_OPENINGS = len(openings)


def opening_figure():

    plt.figure(

        figsize=(12, max(8, 0.45 * NUM_OPENINGS))

    )


#
# ---------------------------------------------------------
# Figure 1
# Shapley entropy by opening
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_shapley_entropy"
        ]

)

values = [

    openings[name][
        "mean_shapley_entropy"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Entropy"

)

plt.title(

    "Shapley Entropy by Opening"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "shapley_entropy_by_opening.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 2
# Banzhaf entropy by opening
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_banzhaf_entropy"
        ]

)

values = [

    openings[name][
        "mean_banzhaf_entropy"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Entropy"

)

plt.title(

    "Banzhaf Entropy by Opening"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "banzhaf_entropy_by_opening.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 3
# Mean evaluation by opening
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_eval"
        ]

)

values = [

    openings[name][
        "mean_eval"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Centipawns"

)

plt.title(

    "Average LC0 Evaluation"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "eval_by_opening.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 4
# Mobility by opening
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_mobility"
        ]

)

values = [

    openings[name][
        "mean_mobility"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Legal moves"

)

plt.title(

    "Average Mobility"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "mobility_by_opening.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 5
# Top-5 Shapley concentration
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_top5_shapley"
        ]

)

values = [

    openings[name][
        "mean_top5_shapley"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Fraction"

)

plt.title(

    "Top-5 Shapley Concentration"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "top5_shapley.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 6
# Top-5 Banzhaf concentration
# ---------------------------------------------------------
#

names = sorted(

    openings,

    key=lambda x:

        openings[x][
            "mean_top5_banzhaf"
        ]

)

values = [

    openings[name][
        "mean_top5_banzhaf"
    ]

    for name in names

]

opening_figure()

plt.barh(

    names,

    values,

)

plt.xlabel(

    "Fraction"

)

plt.title(

    "Top-5 Banzhaf Concentration"

)

plt.tight_layout(
    pad=1.5
)

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "top5_banzhaf.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 7
# Piece importance
# ---------------------------------------------------------
#

piece_names = [

    "white_pawn",
    "black_pawn",

    "white_knight",
    "black_knight",

    "white_bishop",
    "black_bishop",

    "white_rook",
    "black_rook",

    "white_queen",
    "black_queen",

]

display_names = [

    "White Pawn",
    "Black Pawn",

    "White Knight",
    "Black Knight",

    "White Bishop",
    "Black Bishop",

    "White Rook",
    "Black Rook",

    "White Queen",
    "Black Queen",

]

means = []

for piece in piece_names:

    vals = [

        openings[opening]

        .get(
            "piece_totals_shapley",
            {},
        )

        .get(
            piece,
            0,
        )

        for opening in openings

    ]

    means.append(

        np.mean(vals)

    )

plt.figure(

    figsize=(12, 5)

)

plt.bar(

    display_names,

    means,

)

plt.xticks(

    rotation=30,
    ha="right",

)

plt.ylabel(

    "Mean Shapley value"

)

plt.title(

    "Average Piece Importance"

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "piece_importance.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 8
# Square heatmap
# ---------------------------------------------------------
#

heatmap = np.zeros(64)

for opening in openings:

    heatmap += np.array(

        openings[opening][
            "square_heatmap_shapley"
        ]

    )

heatmap /= NUM_OPENINGS

heatmap = heatmap.reshape(

    8,
    8,

)

plt.figure(

    figsize=(8, 8)

)

plt.imshow(

    heatmap[::-1]

)

plt.colorbar(

    label="Mean Shapley value"

)

plt.xticks(

    range(8),

    list("abcdefgh"),

)

plt.yticks(

    range(8),

    range(8, 0, -1),

)

plt.title(

    "Average Shapley Importance by Square"

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "square_heatmap.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 9
# Entropy vs evaluation
# ---------------------------------------------------------
#

x = [

    openings[name][
        "mean_shapley_entropy"
    ]

    for name in openings

]

y = [

    openings[name][
        "mean_eval"
    ]

    for name in openings

]

plt.figure(

    figsize=(8, 6)

)

plt.scatter(

    x,
    y,

)

plt.xlabel(

    "Shapley entropy"

)

plt.ylabel(

    "Evaluation (cp)"

)

plt.title(

    "Entropy vs Evaluation"

)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "entropy_vs_eval.png",

    )

)

plt.close()


#
# ---------------------------------------------------------
# Figure 10
# Mobility vs entropy
# ---------------------------------------------------------
#

x = [

    openings[name][
        "mean_mobility"
    ]

    for name in openings

]

y = [

    openings[name][
        "mean_shapley_entropy"
    ]

    for name in openings

]

plt.figure(

    figsize=(8, 6)

)

plt.scatter(

    x,
    y,

)

plt.xlabel(

    "Mobility"

)

plt.ylabel(

    "Shapley entropy"

)

plt.title(

    "Mobility vs Entropy"

)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "mobility_vs_entropy.png",

    )

)

plt.close()


print()

print(

    f"Saved 10 figures to {OUTPUT_DIR}"

)