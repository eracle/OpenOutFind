# tests/core/test_logging.py
"""stdout is result-only: everything that is not a row goes to stderr."""
import logging

from openoutfind.core.logging import configure_logging, print_banner


def test_logs_go_to_stderr_not_stdout(capsys):
    """A log line on stdout would corrupt a redirected CSV or a piped JSON document."""
    try:
        configure_logging(level=logging.INFO)
        logging.getLogger("openoutfind.test").info("a log line")
    finally:
        logging.getLogger().handlers.clear()

    captured = capsys.readouterr()
    assert "a log line" in captured.err
    assert captured.out == ""


def test_the_banner_is_decoration_and_shares_stderr(capsys):
    print_banner()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "___" in captured.err  # the ASCII banner
