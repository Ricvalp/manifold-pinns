from manifold_pinns.pipeline.cli import main
import pytest


def test_cli_help_paths_import():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
