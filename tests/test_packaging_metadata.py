from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_turbojpeg_extra_is_dedicated_and_included_in_all():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        optional_dependencies = tomllib.load(handle)["project"][
            "optional-dependencies"
        ]

    assert optional_dependencies["turbojpeg"] == ["PyTurboJPEG"]
    assert "PyTurboJPEG" in optional_dependencies["all"]
