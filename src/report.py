"""Render a research brief as Markdown, with gaps stated, not hidden."""


def render_markdown(brief: dict) -> str:
    lines = [
        f"# Research brief: {brief['query']}",
        "",
        brief["summary"],
        "",
        f"Confidence: {brief['confidence']} | Rounds: {brief['rounds']}",
        "",
        "## Findings",
        "",
    ]
    for finding in brief["findings"]:
        lines.append(f"### {finding['question']}")
        lines.append("")
        lines.append(finding["answer"])
        lines.append("")
        lines.append(f"_Source: [{finding['source_id']}] {finding['source_title']}_")
        lines.append("")
    if brief["gaps"]:
        lines.append("## Gaps (no evidence found)")
        lines.append("")
        for gap in brief["gaps"]:
            lines.append(f"- {gap}")
        lines.append("")
    lines.append("## Citations")
    lines.append("")
    lines.append(", ".join(brief["citations"]) if brief["citations"] else "None")
    lines.append("")
    return "\n".join(lines)
