import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def markdown_table(reports: dict) -> str:
    cols = ["em", "f1", "support_recall", "support_precision",
            "avg_claude_calls", "avg_retrievals"]
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


def traces_markdown(report: dict, k: int = 3) -> str:
    """Render up to k per-question hop traces as markdown (a qualitative demo artifact)."""
    lines = []
    for pq in report.get("per_question", [])[:k]:
        lines.append(f"### {pq['qid']} — answer: {pq.get('answer', '')!r}")
        for i, step in enumerate(pq.get("trace", [])):
            lines.append(
                f"- hop {i}: query={step['query']!r} "
                f"sufficient={step['sufficient']} retrieved={step['retrieved_ids']}"
            )
            if step.get("reasoning"):
                lines.append(f"    - reasoning: {step['reasoning']}")
        lines.append("")
    return "\n".join(lines)
