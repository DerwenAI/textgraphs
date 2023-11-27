TODO:

  * debug len(E) == 0
  * debug multiple sentences
  * why aren't there multiple relaions in the lemma graph?
  * make rel infer optional
  * refactor: structure in lemma graph, style in `pyvis`
  * StrEnum for node categories
  * defined styles for `pyvis`
  * add `nx.neighbors()` to hypervolume df
  * render graph in Streamlit

  * `pytest` unit tests
  * setup/config for PyPi
  * add to conda
  * `mkrefs` docs


---

local install:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
pre-commit install --hook-type pre-commit
```

local tests:

```bash
python3 demo.py
```

```bash
./venv/bin/jupyter lab
```

```bash
streamlit run app.py
```
