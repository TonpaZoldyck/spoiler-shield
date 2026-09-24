from spoiler_shield import __version__
from spoiler_shield.cli import main


def test_version_command_prints_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
