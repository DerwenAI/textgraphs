---
title: TextGraphs
emoji: ✴
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.28.2
app_file: app.py
pinned: false
license: mit
---

# TextGraphs

[![DOI](https://zenodo.org/badge/735568863.svg)](https://zenodo.org/doi/10.5281/zenodo.10431783)
![Licence](https://img.shields.io/github/license/DerwenAI/textgraphs)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![CI](https://github.com/DerwenAI/textgraphs/workflows/CI/badge.svg)
<br/>
![Repo size](https://img.shields.io/github/repo-size/DerwenAI/textgraphs)
![downloads](https://img.shields.io/pypi/dm/textgraphs)
![sponsor](https://img.shields.io/github/sponsors/ceteri)

<img
 alt="TextGraphs logo"
 src="https://raw.githubusercontent.com/DerwenAI/textgraphs/main/docs/assets/logo.png"
 width="231"
/>


## project info

Project home: <https://huggingface.co/spaces/DerwenAI/textgraphs>

Full documentation: <https://derwen.ai/docs/txg/>

Sample code is provided in `demo.py`


## requirements

  * Python 3.10+


## deploy library from PyPi

To install from [PyPi](https://pypi.python.org/pypi/textgraphs):

```bash
python3 -m pip install -u textgraphs
python3 -m pip install git+https://github.com/thunlp/OpenNRE
```

NB: both the spaCy and PyPi teams induce packaging errors
since they have "opinionated" views which conflict against
each other and also don't quite follow the Python packaging
standards.


## run demos locally

```bash
python3 demo.py
```

```bash
./venv/bin/jupyter-lab
```

```bash
streamlit run app.py
```


## install library from a source code repo locally

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/thunlp/OpenNRE
```

to run the Streamlit or JupyterLab demos, also install:

```bash
python3 -m pip install -r requirements-dev.txt
```


## license and copyright

Source code for **TextGraphs** plus its logo, documentation, and
examples have an [MIT license](https://spdx.org/licenses/MIT.html)
which is succinct and simplifies use in commercial applications.

All materials herein are Copyright &copy; 2023-2024 Derwen, Inc.


## attribution

Please use the following BibTeX entry for citing **TextGraphs** if you
use it in your research or software:
```bibtex
@software{TextGraphs,
  author = {Paco Nathan},
  title = {{TextGraphs + LLMs + graph ML for entity extraction, linking, ranking, and constructing a lemma graph}},
  year = 2023,
  publisher = {Derwen},
  doi = {10.5281/zenodo.10431783},
  url = {https://github.com/DerwenAI/textgraphs}
}
```


## star history

[![Star History Chart](https://api.star-history.com/svg?repos=derwenai/textgraphs&type=Date)](https://star-history.com/#derwenai/textgraphs&Date)
