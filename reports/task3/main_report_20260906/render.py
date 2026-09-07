"""Execute the saved-results notebook and build its local HTML preview."""

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
from bs4 import BeautifulSoup
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOK = ROOT / "notebooks/04_task3_gender_usage.ipynb"
OUTPUT = Path(__file__).parent / "main_report.html"

notebook = nbformat.read(NOTEBOOK, as_version=4)
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [
    str(ROOT / ".venv/bin/python"),
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
try:
    NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}
    ).execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
nbformat.write(notebook, NOTEBOOK)
html, _ = HTMLExporter().from_notebook_node(notebook)
soup = BeautifulSoup(html, "html.parser")
for link in soup.find_all("a", href=True):
    url = urlsplit(link["href"])
    if url.scheme or url.netloc or not url.path:
        continue
    target = (NOTEBOOK.parent / unquote(url.path)).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Broken report link: {link['href']}")
    link["href"] = os.path.relpath(target, OUTPUT.parent)
    if url.fragment:
        link["href"] += "#" + url.fragment
OUTPUT.write_text(str(soup), encoding="utf-8")
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells.")
print(f"Preview: {OUTPUT}")
