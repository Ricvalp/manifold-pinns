from manifold_pinns.pipeline.cli import _parse_overrides, main
import pytest


def test_cli_help_paths_import():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_parse_overrides_keeps_commas_inside_literals():
    overrides = _parse_overrides(
        "wandb.use=False,idxs=[0,475,871],run_id='pca,baseline'"
    )
    assert overrides["wandb"]["use"] is False
    assert overrides["idxs"] == [0, 475, 871]
    assert overrides["run_id"] == "pca,baseline"
