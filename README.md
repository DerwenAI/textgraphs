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

## project info:

see <https://huggingface.co/spaces/DerwenAI/textgraphs>


## requirements:

  * Python 3.10+


## deploy library from PyPi:

```bash
python3 -m pip install -u textgraphs
python3 -m pip install git+https://github.com/thunlp/OpenNRE
```

NB: both the spaCy and PyPi teams induce packaging errors
because they have "opinionated" views which conflict against
each other and also don't quite follow the Python packaging
standards.


## run demos locally:

```bash
python3 demo.py
```

```bash
./venv/bin/jupyter lab
```

```bash
streamlit run app.py
```


## install library locally:

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
```

to run the Streamlit or JupyterLab demos, or prior to working
locally on the development and testing of this repo:

```bash
python3 -m pip install -r requirements-dev.txt
pre-commit install --hook-type pre-commit
```

## test library locally:

```bash
python3 -m pytest
```


## update code on GitHub:

```bash
git remote set-url origin https://github.com/DerwenAI/textgraphs.git
```


## update code on Hugging Face:

```bash
git remote set-url origin https://huggingface.co/spaces/DerwenAI/textgraphs
```


## publish library:

```bash
rm dist/*
python3 -m build
twine check dist/*
twine upload ./dist/* --verbose
```


## administrivia

<details>
  <summary>License and Copyright</summary>

Source code for **TextGraphs** plus its logo, documentation, and
examples have an [MIT license](https://spdx.org/licenses/MIT.html)
which is succinct and simplifies use in commercial applications.

All materials herein are Copyright &copy; 2023 Derwen, Inc.
</details>
