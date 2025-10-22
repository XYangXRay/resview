#!/usr/bin/env python3
"""
Scaffold a ResView-style Python app (PyPI-ready, optional conda recipe, Pixi, VS Code tasks).
Single entry point: `run(...)` — can be called from Python or via CLI (main() just forwards).

Examples
--------
CLI:
  python scaffold_resview.py --dir ./resview \
    --pkg resview --display-name "ResView" --version 0.1.0 \
    --author "Xiaogang Yang" --email "yangxg@bnl.gov" \
    --license "BSD-3-Clause" --repo "https://github.com/your-org/resview" \
    --with-pixi --with-conda --with-gh --force

Python:
  from scaffold_resview import run
  run(dir="./resview", pkg="resview", display_name="ResView", version="0.1.0",
      author="Xiaogang Yang", email="yangxg@bnl.gov", license="BSD-3-Clause",
      repo="https://github.com/your-org/resview", with_pixi=True,
      with_conda=True, with_gh=True, force=True)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
from string import Template
import sys
from textwrap import dedent

# ---------- Tiny templating helpers ----------
def T(s: str, **kw) -> str:
    return Template(dedent(s).lstrip("\n")).substitute(**kw)

def write_file(path: Path, content: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"SKIP (exists): {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"WRITE: {path}")

# ---------- Templates ----------
def render_pyproject(**k) -> str:
    return T(
        r"""
        [build-system]
        requires = ["hatchling>=1.24", "hatch-vcs>=0.4"]
        build-backend = "hatchling.build"

        [project]
        name = "${pkg}"
        description = "${display_name} — Reciprocal-Space Viewer & RSM workflow"
        readme = "README.md"
        license = { file = "LICENSE" }
        authors = [{ name = "${author}", email = "${email}" }]
        requires-python = ">=3.9"
        dynamic = ["version"]
        dependencies = [
          "numpy>=1.23",
          "pyyaml>=6",
          "qtpy>=2.4",
          "magicgui>=0.9",
          "napari>=0.5",
          "xrayutilities>=1.7",
          "rsm3d>=0.1.0",
        ]

        [project.optional-dependencies]
        pyside6 = ["pyside6>=6.5"]
        pyqt6   = ["pyqt6>=6.5"]

        [project.urls]
        Homepage = "${repo}"
        Issues   = "${repo}/issues"

        [project.scripts]
        ${pkg} = "${pkg}.gui:main"

        [tool.hatch.version]
        source = "vcs"

        [tool.hatch.build.targets.sdist]
        include = ["src", "README.md", "LICENSE"]

        [tool.hatch.build.targets.wheel]
        packages = ["src/${pkg}"]

        [tool.ruff]
        line-length = 100
        lint.select = ["E", "F", "I", "B"]
        lint.ignore = ["E501"]
        """,
        **k,
    )

def render_readme(**k) -> str:
    return T(
        """
        # ${display_name}

        ${display_name} is a Reciprocal-Space Viewer (ResView) and RSM workflow GUI.

        ## Install
        **pip**
        ```bash
        pip install ${pkg}
        ${pkg} --version
        ```

        **Run**
        ```bash
        ${pkg}
        ```

        ## Dev quickstart
        ```bash
        python -m pip install -U pip build
        python -m pip install -e .  # editable dev
        python -m build             # sdist + wheel
        ```

        ## Environment variable
        Set a defaults YAML for your experiment setup:
        ```bash
        export RSM3D_DEFAULTS_YAML=/path/to/rsm3d_defaults.yaml
        ```
        """,
        **k,
    )

def render_license(**k) -> str:
    return T(
        """
        BSD 3-Clause License

        Copyright (c) ${year}, ${author}
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice,
           this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
           notice, this list of conditions and the following disclaimer in the
           documentation and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
           contributors may be used to endorse or promote products derived from
           this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """,
        **k,
    )

def render_init(**k) -> str:
    return T(
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("${pkg}")
        except PackageNotFoundError:
            __version__ = "0.0.0"
        """,
        **k,
    )

def render_main(**k) -> str:
    return T(
        r"""
        from .gui import main

        if __name__ == "__main__":
            raise SystemExit(main())
        """,
        **k,
    )

def render_gui(**k) -> str:
    return T(
        r'''
        import sys
        from qtpy import QtWidgets
        from . import __version__

        APP_QSS = """
        QMainWindow { background: #fafafa; }
        QLabel.title { font-size: 20px; font-weight: 700; padding: 8px; }
        QPushButton { padding: 6px 10px; border-radius: 8px; }
        """

        def main(argv=None):
            argv = sys.argv[1:] if argv is None else argv
            if "--version" in argv or "-V" in argv:
                print(__version__)
                return 0

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            app.setStyleSheet(APP_QSS)

            win = QtWidgets.QMainWindow()
            win.setWindowTitle("ResView")
            central = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(central)
            title = QtWidgets.QLabel("ResView — Reciprocal-Space Viewer")
            title.setObjectName("title"); title.setProperty("class", "title")
            btn = QtWidgets.QPushButton("Launch GUI (replace stub with your app)")
            lay.addWidget(title)
            lay.addWidget(btn)
            lay.addStretch(1)
            win.setCentralWidget(central)
            win.resize(900, 600)
            win.show()

            exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
            return exec_fn()
        '''
    )

def render_tests(**k) -> str:
    return T(
        """
        def test_import():
            import ${pkg}
            assert hasattr(${pkg}, "__version__")
        """,
        **k,
    )

def render_gitignore() -> str:
    return dedent(
        """
        __pycache__/
        .pytest_cache/
        .ruff_cache/
        .venv/
        .pixi/
        build/
        dist/
        *.egg-info/
        *.pyc
        .DS_Store
        rsm3d_defaults.yaml
        """
    ).lstrip("\n")

def render_actions(**k) -> str:
    return T(
        """
        name: Publish to PyPI
        on:
          push:
            tags: ["v*.*.*"]
        jobs:
          build-publish:
            runs-on: ubuntu-latest
            permissions: { id-token: write, contents: read }
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with: { python-version: "3.11" }
              - run: python -m pip install -U pip build
              - run: python -m build
              - uses: pypa/gh-action-pypi-publish@release/v1
        """,
        **k,
    )

def render_meta(**k) -> str:
    return T(
        """
        {% set name = "${pkg}" %}
        {% set version = "${version}" %}

        package:
          name: {{ name|lower }}
          version: {{ version }}

        source:
          url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
          sha256:

        build:
          number: 0
          noarch: python
          script: {{ PYTHON }} -m pip install . -vv --no-deps

        requirements:
          host:
            - python >=3.9
            - pip
            - hatchling
          run:
            - python >=3.9
            - numpy >=1.23
            - pyyaml >=6
            - qtpy >=2.4
            - magicgui >=0.9
            - napari >=0.5
            - xrayutilities >=1.7
            - rsm3d >=0.1.0
            - pyqt  # or pyside6

        test:
          commands:
            - ${pkg} --version
          imports:
            - ${pkg}

        about:
          home: ${repo}
          license: BSD-3-Clause
          license_file: LICENSE
          summary: Reciprocal-space viewer and RSM workflow GUI

        extra:
          recipe-maintainers:
            - your-github-handle
        """,
        **k,
    )

def render_pixi(**k) -> str:
    return T(
        """
        [project]
        name = "${pkg}"
        version = "${version}"
        channels = ["conda-forge"]

        [dependencies]
        python = ">=3.11,<3.14"
        numpy = "*"
        pyyaml = "*"
        qtpy = "*"
        magicgui = "*"
        napari = "*"
        xrayutilities = "*"
        pyqt = "*"
        rsm3d = "*"

        [pypi-dependencies]
        ${pkg} = { path = ".", editable = true }

        [tasks]
        run = "${pkg}"
        test = "pytest -q"
        """
    )


# ---------- The single entry point ----------
def run(
    *,
    dir: str,
    pkg: str = "resview",
    display_name: str = "ResView",
    version: str = "0.1.0",
    author: str = "Your Name",
    email: str = "you@example.com",
    license: str = "BSD-3-Clause",
    repo: str = "https://github.com/your-org/resview",
    with_pixi: bool = False,
    with_conda: bool = False,
    with_gh: bool = False,
    force: bool = False,
) -> None:
    """Scaffold the project files and folders."""
    year = _dt.datetime.now().year

    root = Path(dir).resolve()
    pkg_dir = root / "src" / pkg
    tests_dir = root / "tests"

    # Core files
    write_file(
        root / "pyproject.toml",
        render_pyproject(pkg=pkg, display_name=display_name, author=author, email=email, repo=repo),
        overwrite=force,
    )
    write_file(root / "README.md", render_readme(pkg=pkg, display_name=display_name), overwrite=force)
    write_file(root / "LICENSE", render_license(year=year, author=author), overwrite=force)
    write_file(root / ".gitignore", render_gitignore(), overwrite=force)

    # Package
    write_file(pkg_dir / "__init__.py", render_init(pkg=pkg), overwrite=force)
    write_file(pkg_dir / "__main__.py", render_main(pkg=pkg), overwrite=force)
    write_file(pkg_dir / "gui.py", render_gui(pkg=pkg), overwrite=force)

    # Tests
    write_file(tests_dir / "test_import.py", render_tests(pkg=pkg), overwrite=force)


    # Optional: GitHub Actions
    if with_gh:
        write_file(root / ".github" / "workflows" / "release-pypi.yml", render_actions(), overwrite=force)

    # Optional: conda recipe skeleton
    if with_conda:
        write_file(root / "recipe" / "meta.yaml", render_meta(pkg=pkg, version=version, repo=repo), overwrite=force)

    # Optional: Pixi
    if with_pixi:
        write_file(root / "pixi.toml", render_pixi(pkg=pkg, version=version), overwrite=force)

    print("\nDone.")
    print(f"Next steps:\n  1) cd {root}\n  2) python -m pip install -e .\n  3) {pkg}  # run GUI\n")

# ---------- CLI wrapper (for convenience) ----------
def parse_args():
    p = argparse.ArgumentParser(description="Scaffold a ResView project (single entry point via run()).")
    p.add_argument("--dir", required=True, help="~/pyprojects/resview")
    p.add_argument("--pkg", default="resview", help="Package / project name (PyPI/conda)")
    p.add_argument("--display-name", default="ResView", help="Human-friendly app name")
    p.add_argument("--version", default="0.1.0", help="Initial version (used in recipe/pixi)")
    p.add_argument("--author", default="Xiaogang Yang", help="Author name")
    p.add_argument("--email", default="yangxg@bnl.gov", help="Author email")
    p.add_argument("--license", default="BSD-3-Clause", help="License id (stored in LICENSE file)")
    p.add_argument("--repo", default="https://github.com/your-org/resview", help="Homepage/Repo URL")
    p.add_argument("--with-pixi", action="store_true", help="Include pixi.toml")
    p.add_argument("--with-conda", action="store_true", help="Include conda recipe skeleton")
    p.add_argument("--with-gh", action="store_true", help="Include GitHub Actions publish workflow")
    p.add_argument("--force", action="store_true", help="Overwrite files if they exist")
    return p.parse_args()

def main():
    a = parse_args()
    # Forward everything to the single entry point:
    run(
        dir=a.dir,
        pkg=a.pkg,
        display_name=a.display_name,
        version=a.version,
        author=a.author,
        email=a.email,
        license=a.license,
        repo=a.repo,
        with_pixi=a.with_pixi,
        with_conda=a.with_conda,
        with_gh=a.with_gh,
        force=a.force,
    )

if __name__ == "__main__":
    main()