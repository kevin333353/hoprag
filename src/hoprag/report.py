import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def markdown_table(reports: dict) -> str:
    cols = ["em", "f1", "support_recall", "avg_claude_calls"]
    lines = ["| variant | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, r in reports.items():
        cells = [f"{r[c]:.2f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cost_curve_png(reports: dict, out_path: str) -> None:
    xs = [r["avg_claude_calls"] for r in reports.values()]
    ys = [r["f1"] for r in reports.values()]
    names = list(reports.keys())
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y))
    ax.set_xlabel("avg Claude calls / question")
    ax.set_ylabel("answer F1")
    ax.set_title("Accuracy vs cost")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
