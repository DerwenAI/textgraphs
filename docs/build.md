# Build Instructions

!!! note
    In most cases you won't need to build this package locally.

Unless you're doing development work on the **textgraphs** library itself,
simply install based on the instructions in
["Getting Started"](https://derwen.ai/docs/txg/start/).


## Setup

To set up the build environment locally:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip wheel setuptools

python3 -m pip install -e .
python3 -m pip install -r requirements-dev.txt
```

We use *pre-commit hooks* based on [`pre-commit`](https://pre-commit.com/)
and to configure that locally:
```
pre-commit install --hook-type pre-commit
```


## Test Coverage

This project uses
[`pytest`](https://docs.pytest.org/)
for *unit test* coverage.
Source for unit tests is in the
[`tests`](https://github.com/DerwenAI/textgraphs/tree/main/tests)
subdirectory.

To run the unit tests:
```
python3 -m pytest
```

Note that these tests run as part of the CI workflow
whenever code is updated on the GitHub repo.


## Online Documentation

To generate documentation pages, you will also need to download
[`ChromeDriver`](https://googlechromelabs.github.io/chrome-for-testing/)
for your version of the `Chrome` browser, saved as `chromedriver` in
this directory.

Source for the documentation is in the
[`docs`](https://github.com/DerwenAI/textgraphs/tree/main/docs)
subdirectory.

To build the documentation:
```
./bin/nb_md.sh
./pkg_doc.py docs/ref.md
mkdocs build
```

Then run `./bin/preview.py` and load <http://127.0.0.1:8000/docs/>
in your browser to preview the generated microsite locally.

To package the generated microsite for deployment on a
web server:
```
tar cvzf txg.tgz site/
```


## Remote Repo Updates

To update source code repo on GitHub:

```
git remote set-url origin https://github.com/DerwenAI/textgraphs.git
git push
```

Create new releases on GitHub then run `git pull` locally prior to
updating Hugging Face or making a new package release.

To update source code repo+demo on Hugging Face:

```
git remote set-url origin https://huggingface.co/spaces/DerwenAI/textgraphs
git push
```


## Package Release

To update the [release on PyPi](https://pypi.org/project/textgraphs/):
```
./bin/push_pypi.sh
```


## Packaging

Both the spaCy and PyPi teams induce packaging errors since they
have "opinionated" views which conflict against each other and also
don't quite follow the [Python packaging standards](https://peps.python.org/pep-0621/).

Moreover, the various dependencies here use a wide range of approaches
for model downloads: quite appropriately, the spaCy team does not want
to package their language models on PyPi.
However, they don't use more contemporary means of model download,
such as HF transformers, either -- and that triggers logging problems.
Overall, logging approaches used by the dependencies here for errors/warnings
are mostly ad-hoc.

These three issues (packaging, model downloads, logging) pose a small nightmare
for managing Python library packaging downstream.
To that point, this project implements several workarounds so that
applications can download from PyPi.

Meanwhile keep watch on developments of the following dependencies,
if they introduce breaking changes or move toward more standard
packaging practices:

  * `spaCy` -- model downloads, logging
  * `OpenNRE` -- PyPi packaging, logging
  * HF `transformers` and `tokenizers` -- logging
  * WikiMedia APIs -- SSL certificate expiry
