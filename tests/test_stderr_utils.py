import os

from hs2p.stderr_utils import run_with_filtered_stderr


def test_run_with_filtered_stderr_suppresses_known_cucim_noise(capfd):
    def _write_noise():
        os.write(2, b"cuInit Failed, error CUDA_ERROR_NO_DEVICE\n")
        os.write(2, b"cuFile initialization failed\n")
        os.write(2, b"kept line\n")
        return 123

    result = run_with_filtered_stderr(_write_noise)
    captured = capfd.readouterr()

    assert result == 123
    assert captured.err == "kept line\n"


def test_run_with_filtered_stderr_forwards_unrelated_stderr(capfd):
    def _write_noise():
        os.write(2, b"unrelated warning\n")
        return 7

    result = run_with_filtered_stderr(_write_noise)
    captured = capfd.readouterr()

    assert result == 7
    assert captured.err == "unrelated warning\n"
