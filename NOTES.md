local install:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
pre-commit install --hook-type pre-commit
```

local test:

```bash
streamlit run app.py
```