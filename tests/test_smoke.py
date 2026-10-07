import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_readme_exists():
    assert (ROOT / "README.md").is_file()


def test_requirements_exists():
    assert (ROOT / "requirements.txt").is_file()


def test_has_python_source():
    assert list(ROOT.glob("**/*.py"))


def test_has_no_empty_python_files():
    for path in ROOT.glob("**/*.py"):
        if path.stat().st_size > 0:
            assert path.stat().st_size > 20, f"{path} looks empty"


def test_src_directory_exists():
    assert (ROOT / "src").is_dir()
