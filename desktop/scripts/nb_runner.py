"""Execute a notebook with the research kernel and print what it produced.

Invoked by the terminal's notebook runtime with the research venv's python:
executes in place (outputs are written back into the .ipynb, so the artifact
keeps its results), then prints each code cell's outputs so the run is
readable in the visible terminal lane.
"""

import sys

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=90, kernel_name="python3")
try:
    client.execute()
finally:
    nbformat.write(nb, path)

failed = False
cell_no = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    cell_no += 1
    print(f"--- cell {cell_no} ---")
    for out in cell.get("outputs", []):
        if out["output_type"] == "stream":
            print(out["text"], end="")
        elif out["output_type"] == "execute_result":
            print("".join(out["data"].get("text/plain", [])))
        elif out["output_type"] == "error":
            failed = True
            print(f"ERROR: {out['ename']}: {out['evalue']}")

sys.exit(1 if failed else 0)
