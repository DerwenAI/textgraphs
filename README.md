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

  * Python 3.9+


## deploy library from PyPi:

```bash
python3 -m pip install -u textgraphs
python3 -m spacy download en_core_web_sm
python3 -m pip install git+https://github.com/thunlp/OpenNRE
```

NB: both the spaCy and PyPi teams cause packagine errors
because they have "opinionated" views against following
the Python packaging standards.


## install locally:

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

## test locally:

```bash
python3 -m pytest
```


## run locally:

```bash
python3 demo.py
```

```bash
./venv/bin/jupyter lab
```

```bash
streamlit run app.py
```

## to publish:

```bash
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
