"""Split a .py on `# %%` markers (or blank-line paragraphs) into a Kaggle-ready .ipynb."""
import json, sys, re, pathlib

src = pathlib.Path(sys.argv[1])
text = src.read_text(encoding="utf-8")

if "# %%" in text:
    chunks = [c.strip("\n") for c in text.split("# %%") if c.strip()]
else:
    head, _, body = text.partition('"""')
    doc, _, rest = body.partition('"""')
    chunks = [f'"""{doc}"""', rest.strip("\n")]

nb = {
    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
               "outputs": [], "source": (c + "\n").splitlines(keepends=True)} for c in chunks],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = src.with_suffix(".ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
