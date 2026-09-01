from pathlib import Path


def pinned_requirements(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("-r ")
    ]


def test_pyproject_declares_all_pinned_runtime_and_dev_dependencies() -> None:
    text = Path("pyproject.toml").read_text()
    for requirement in pinned_requirements("requirements.txt"):
        assert f'"{requirement}"' in text
    for requirement in pinned_requirements("requirements-dev.txt"):
        assert f'"{requirement}"' in text
