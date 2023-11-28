---
title: TextGraph
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

see <https://huggingface.co/spaces/DerwenAI/textgraph>


## install locally:

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt

pre-commit install --hook-type pre-commit
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
