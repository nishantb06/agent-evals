uv run python -c '
from pathlib import Path
from mcp_server import index_document

kb = Path("sandbox/kb")
for p in sorted(kb.rglob("*.md")):
    rel = p.relative_to("sandbox").as_posix()  # e.g. "kb/doc1.md"
    print(rel, index_document(rel))
'