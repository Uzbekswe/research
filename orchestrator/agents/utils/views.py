_STATUS_ICONS: dict[str, str] = {
    "started":   "🔄",
    "complete":  "✅",
    "reviewing": "🔍",
    "revising":  "✏️",
    "approved":  "✓",
    "failed":    "❌",
}

_SEPARATOR = "=" * 60


def print_agent_output(output: str, agent_name: str = "AGENT") -> None:
    """Print formatted agent output to console."""
    print(f"\n{_SEPARATOR}")
    print(f"🤖 {agent_name.upper()}")
    print(_SEPARATOR)
    print(output)
    print(_SEPARATOR)


def log_research_progress(section: str, status: str, details: str = "") -> None:
    """Log section-level research progress with a status icon."""
    icon = _STATUS_ICONS.get(status, "•")
    msg = f"  {icon} [{section[:40]}] {status}"
    if details:
        msg += f" — {details}"
    print(msg)


def format_sections_table(
    sections: list[str],
    statuses: list[str] | None = None,
) -> str:
    """Return a formatted table of sections and their completion status."""
    lines = ["\n📋 Research Outline:"]
    for i, section in enumerate(sections):
        status = statuses[i] if statuses else "pending"
        icon = "✅" if status == "complete" else "⏳"
        lines.append(f"  {i + 1}. {icon} {section}")
    return "\n".join(lines)
