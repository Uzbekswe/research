import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from orchestrator import ChiefEditorAgent  # noqa: E402


def load_task(task_path: str = "task.json") -> dict:
    path = Path(task_path)
    if not path.exists():
        print(f"❌ task.json not found at {task_path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


async def main() -> dict:
    task_path = sys.argv[1] if len(sys.argv) > 1 else "task.json"
    task = load_task(task_path)

    chief_editor = ChiefEditorAgent(task=task, output_dir="./outputs")
    final_state = await chief_editor.run_research_task()

    print("\n📁 Report saved to ./outputs/")
    return final_state


if __name__ == "__main__":
    asyncio.run(main())
