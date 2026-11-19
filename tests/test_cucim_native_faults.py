"""cuCIM's native faults: the cuFile config hs2p sets, and crashes that stay visible (#278).

None of these tests needs cuCIM or a GPU: cuCIM is replaced by a fake module.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import hs2p
import hs2p.wsi.backends.cucim as cucim_reader_mod

CUFILE_ENV = "CUFILE_ENV_PATH_JSON"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SLIDE = REPO_ROOT / "tests" / "fixtures" / "input" / "test-wsi.tif"

# fake_crash.py from #278, verbatim: the cuCIM opener dies inside
# ``_suppress_native_stderr`` as cuCIM's native crashes do.
FAKE_CRASH = """
import os
import signal
import sys

from hs2p.wsi import reader
from hs2p.wsi.backends.cucim import _suppress_native_stderr


def failing_opener(*args, **kwargs):
    if sys.argv[2] == "raise":
        raise RuntimeError("cuCIM cannot open this slide")
    with _suppress_native_stderr():  # CuCIMReader wraps its import and reads like this
        os.write(2, b"native assertion failed\\n")
        os.kill(os.getpid(), signal.SIGSEGV)


cucim = reader._BACKENDS["cucim"]
reader._BACKENDS["cucim"] = reader._BackendSpec(failing_opener, cucim.supports_path)
with reader.open_slide(sys.argv[1], backend="auto") as slide:
    print("selected", slide.backend_name)
"""


def _run_child(
    script: str,
    *args: str,
    interpreter_flags: tuple[str, ...] = (),
    child_env: dict[str, str] | None = None,
):
    # Either variable turns faulthandler on in the child (dev mode does), whatever the test asks.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONFAULTHANDLER", "PYTHONDEVMODE"}
    }
    env.update(child_env or {})
    return subprocess.run(
        [sys.executable, *interpreter_flags, "-c", script, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_cucim() -> SimpleNamespace:
    slide = SimpleNamespace(
        metadata={
            "openslide": {"MPP": 0.5},
            "cucim": {
                "resolutions": {
                    "level_dimensions": [[64, 64]],
                    "level_downsamples": [1.0],
                }
            },
        }
    )
    return SimpleNamespace(CuImage=MagicMock(return_value=slide))


def _open_reader_on_fake_cucim(monkeypatch) -> str | None:
    """Open a ``CuCIMReader`` on a fake cuCIM; return ``CUFILE_ENV_PATH_JSON`` as its import saw it."""
    fake_cucim = _fake_cucim()
    seen_at_import: dict[str, str | None] = {}
    original_import_module = cucim_reader_mod.importlib.import_module

    def _import_module(name):
        if name != "cucim":
            return original_import_module(name)
        seen_at_import["value"] = os.environ.get(CUFILE_ENV)
        return fake_cucim

    monkeypatch.setattr(cucim_reader_mod.importlib, "import_module", _import_module)
    cucim_reader_mod.CuCIMReader("fake.svs")
    return seen_at_import["value"]


def test_cucim_import_sees_the_shipped_cufile_config_when_the_variable_is_unset(
    monkeypatch,
):
    monkeypatch.delenv(CUFILE_ENV, raising=False)

    seen = _open_reader_on_fake_cucim(monkeypatch)

    assert seen is not None
    config_path = Path(seen)
    assert config_path.is_file()
    assert config_path.resolve().is_relative_to(Path(hs2p.__file__).parent.resolve())
    assert json.loads(config_path.read_text()) == {"execution": {"max_io_threads": 0}}


def test_cucim_import_keeps_a_cufile_config_the_user_set(monkeypatch, tmp_path):
    user_config = str(tmp_path / "my-cufile.json")
    monkeypatch.setenv(CUFILE_ENV, user_config)

    seen = _open_reader_on_fake_cucim(monkeypatch)

    assert seen == user_config
    assert os.environ[CUFILE_ENV] == user_config


class _CucimWithoutClara:
    """cuCIM 26.06 whose native ``clara`` module failed to load.

    Its ``__init__`` swallows that ``ImportError``, so ``import cucim`` succeeds and
    the error resurfaces when ``cucim.CuImage`` is looked up.
    """

    def __getattr__(self, name):
        raise ImportError("libcucim.so.26: cannot open shared object file")


def _fail_cucim_import(monkeypatch, failure: str) -> None:
    original_import_module = cucim_reader_mod.importlib.import_module

    def _import_module(name):
        if name != "cucim":
            return original_import_module(name)
        if failure == "at CuImage lookup":
            return _CucimWithoutClara()
        raise ImportError("No module named 'cucim'")

    monkeypatch.setattr(cucim_reader_mod.importlib, "import_module", _import_module)


CUCIM_IMPORT_FAILURES = pytest.mark.parametrize(
    "failure", ["at import", "at CuImage lookup"]
)


@CUCIM_IMPORT_FAILURES
def test_a_failed_cucim_import_unsets_the_cufile_config_hs2p_set(
    monkeypatch, failure
):
    monkeypatch.delenv(CUFILE_ENV, raising=False)
    _fail_cucim_import(monkeypatch, failure)

    with pytest.raises(ImportError, match="cucim is required"):
        cucim_reader_mod.CuCIMReader("fake.svs")

    assert CUFILE_ENV not in os.environ


@CUCIM_IMPORT_FAILURES
def test_a_failed_cucim_import_keeps_a_cufile_config_the_user_set(
    monkeypatch, tmp_path, failure
):
    user_config = str(tmp_path / "my-cufile.json")
    monkeypatch.setenv(CUFILE_ENV, user_config)
    _fail_cucim_import(monkeypatch, failure)

    with pytest.raises(ImportError, match="cucim is required"):
        cucim_reader_mod.CuCIMReader("fake.svs")

    assert os.environ[CUFILE_ENV] == user_config


# cuCIM 26.06 guards its ``import cupy`` with ``except ImportError`` only, so another
# error there, such as a NumPy ABI mismatch, escapes ``import cucim``.
@pytest.mark.parametrize("user_set", [False, True], ids=["unset", "user-set"])
def test_a_cucim_import_failing_with_another_error_raises_it_unchanged(
    monkeypatch, tmp_path, user_set
):
    user_config = str(tmp_path / "my-cufile.json")
    if user_set:
        monkeypatch.setenv(CUFILE_ENV, user_config)
    else:
        monkeypatch.delenv(CUFILE_ENV, raising=False)
    abi_mismatch = ValueError("numpy.dtype size changed, may indicate binary incompatibility")
    original_import_module = cucim_reader_mod.importlib.import_module

    def _import_module(name):
        if name != "cucim":
            return original_import_module(name)
        raise abi_mismatch

    monkeypatch.setattr(cucim_reader_mod.importlib, "import_module", _import_module)

    with pytest.raises(ValueError) as raised:
        cucim_reader_mod.CuCIMReader("fake.svs")

    assert raised.value is abi_mismatch
    assert os.environ.get(CUFILE_ENV) == (user_config if user_set else None)


# Two threads open their first reader at once. Thread A set the variable; thread B came in
# while A was importing cuCIM. A's import then dies of a failure that does not repeat.
def test_a_cucim_import_failing_in_another_thread_leaves_this_one_the_cufile_config(
    monkeypatch,
):
    timeout = 10.0
    a_in_import = threading.Event()
    b_in_import = threading.Event()
    a_raised = threading.Event()
    raised: dict[str, BaseException] = {}
    seen_by_b: dict[str, str | None] = {}
    fake_cucim = _fake_cucim()
    original_import_module = cucim_reader_mod.importlib.import_module

    def _import_module(name):
        if name != "cucim":
            return original_import_module(name)
        if threading.current_thread() is thread_a:
            a_in_import.set()
            b_in_import.wait(0.5)  # B gets here only if nothing keeps it out
            raise MemoryError
        b_in_import.set()
        a_raised.wait(timeout)
        seen_by_b["value"] = os.environ.get(CUFILE_ENV)
        return fake_cucim

    def _open(name: str) -> None:
        try:
            cucim_reader_mod.CuCIMReader(f"{name}.svs")
        except BaseException as exc:
            raised[name] = exc

    def _open_a() -> None:
        _open("a")
        a_raised.set()

    def _open_b() -> None:
        a_in_import.wait(timeout)
        _open("b")

    monkeypatch.setattr(cucim_reader_mod.importlib, "import_module", _import_module)
    found = os.environ.pop(CUFILE_ENV, None)
    try:
        thread_a = threading.Thread(target=_open_a, daemon=True)
        thread_b = threading.Thread(target=_open_b, daemon=True)
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout)
        thread_b.join(timeout)
    finally:
        if found is None:
            os.environ.pop(CUFILE_ENV, None)
        else:
            os.environ[CUFILE_ENV] = found

    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert isinstance(raised.pop("a", None), MemoryError)
    assert raised == {}
    assert seen_by_b == {"value": str(cucim_reader_mod.CUFILE_CONFIG_PATH)}


# A thread holds one of the reader's two locks while the main thread forks. The forked
# child, where that thread does not exist, opens a reader on a fake cuCIM and reads.
FORK_WHILE_A_THREAD_HOLDS_A_LOCK = """
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

import hs2p.wsi.backends.cucim as m

TIMEOUT = 10.0


class FakeCuImage:
    metadata = {
        "openslide": {"MPP": 0.5},
        "cucim": {"resolutions": {"level_dimensions": [[64, 64]], "level_downsamples": [1.0]}},
    }

    def __init__(self, path):
        pass

    def read_region(self, *, location, size, level, num_workers, device):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _import_module(name, _original=m.importlib.import_module):
    if name != "cucim":
        return _original(name)
    return SimpleNamespace(CuImage=FakeCuImage)


m.importlib.import_module = _import_module
lock = {
    "cufile config": m._CUFILE_CONFIG_LOCK,
    "stderr suppression": m._suppress_native_stderr._lock,
}[sys.argv[1]]
held, forked = threading.Event(), threading.Event()


def _hold():
    with lock:
        held.set()
        forked.wait(TIMEOUT)


threading.Thread(target=_hold, daemon=True).start()
assert held.wait(TIMEOUT)
pid = os.fork()
if pid == 0:
    m.CuCIMReader("fake.svs").read_region((0, 0), 0, (16, 16))
    os._exit(0)
forked.set()
deadline = time.monotonic() + TIMEOUT
while not (waited := os.waitpid(pid, os.WNOHANG))[0] and time.monotonic() < deadline:
    time.sleep(0.01)
if waited[0]:
    print("child exited", os.waitstatus_to_exitcode(waited[1]))
else:
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    print("child hung")
"""


@pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(os, "register_at_fork"),
    reason="needs os.fork and os.register_at_fork",
)
@pytest.mark.parametrize("lock", ["cufile config", "stderr suppression"])
def test_a_child_forked_while_another_thread_holds_a_reader_lock_opens_and_reads(lock):
    result = _run_child(FORK_WHILE_A_THREAD_HOLDS_A_LOCK, lock)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "child exited 0\n", result.stderr


@pytest.mark.parametrize(
    ("interpreter_flags", "child_env"),
    [
        ((), {}),
        (("-X", "faulthandler"), {}),
        ((), {"PYTHONFAULTHANDLER": "1"}),
        (("-X", "dev"), {}),
    ],
    ids=["faulthandler-off", "faulthandler-on", "PYTHONFAULTHANDLER", "dev-mode"],
)
def test_a_native_crash_inside_suppressed_stderr_names_the_signal_on_stderr(
    interpreter_flags, child_env
):
    result = _run_child(
        FAKE_CRASH,
        str(FIXTURE_SLIDE),
        "crash",
        interpreter_flags=interpreter_flags,
        child_env=child_env,
    )

    assert result.returncode == -signal.SIGSEGV
    assert "Segmentation fault" in result.stderr


# A caller that aimed faulthandler at its own file, as pytest's plugin and an application's
# crash log do, dies of a signal inside a suppression or after one has ended. Started with
# no faulthandler option, it may set PYTHONFAULTHANDLER afterwards for its child processes.
CRASH_WITH_OWN_FAULTHANDLER = """
import faulthandler
import os
import signal
import sys

from hs2p.wsi.backends.cucim import _suppress_native_stderr

crash_log = open(sys.argv[1], "w")
faulthandler.enable(file=crash_log)
if sys.argv[3] == "sets PYTHONFAULTHANDLER":
    os.environ["PYTHONFAULTHANDLER"] = "1"
with _suppress_native_stderr():
    if sys.argv[2] == "inside":
        os.kill(os.getpid(), signal.SIGSEGV)
os.kill(os.getpid(), signal.SIGSEGV)
"""


@pytest.mark.parametrize("environment", ["leaves the environment", "sets PYTHONFAULTHANDLER"])
@pytest.mark.parametrize("crash", ["inside", "after"])
def test_a_native_crash_is_dumped_where_the_caller_aimed_faulthandler(
    crash, environment, tmp_path
):
    crash_log = tmp_path / "crash.log"

    result = _run_child(CRASH_WITH_OWN_FAULTHANDLER, str(crash_log), crash, environment)

    assert result.returncode == -signal.SIGSEGV
    assert "Fatal Python error: Segmentation fault" in crash_log.read_text()


# A fake cuCIM whose import aborts the way cuFile's assertion does, opened through either
# entry point that imports cuCIM: ``open_slide`` and the batched streaming reads, which
# wrap the reader's own suppressed import in a second suppression.
ABORTING_CUCIM_IMPORT = """
import os
import sys
from pathlib import Path

import hs2p.wsi.backends.cucim as m


def _import_module(name, _original=m.importlib.import_module):
    if name != "cucim":
        return _original(name)
    os.write(2, b"cufile_worker_thread.h:57: Assertion `0' failed.\\n")
    os.abort()


m.importlib.import_module = _import_module
if sys.argv[1] == "open_slide":
    from hs2p.wsi.reader import open_slide

    open_slide("fake.svs", backend="cucim")
else:
    from hs2p.wsi.streaming.batched import (
        BatchedReadRequest,
        iter_cucim_batched_read_regions,
    )

    iter_cucim_batched_read_regions(
        image_path=Path("fake.svs"),
        requests=[BatchedReadRequest(location=(0, 0), size=(16, 16))],
        level=0,
        num_workers=1,
    )
"""


@pytest.mark.parametrize("entry_point", ["open_slide", "batched_reads"])
def test_an_abort_at_cucim_import_names_the_signal_on_stderr(entry_point):
    result = _run_child(ABORTING_CUCIM_IMPORT, entry_point)

    assert result.returncode == -signal.SIGABRT
    assert "Aborted" in result.stderr


# Batched streaming reads on a fake cuCIM that prints cuInit/cuFile noise at import, at
# open, at the read call and while its batch is iterated, which happens after the
# reader's own suppression has ended but inside the batched reads' outer one.
NOISY_BATCHED_READS = """
import faulthandler
import os
from pathlib import Path

import numpy as np

import hs2p.wsi.backends.cucim as m
from hs2p.wsi.streaming.batched import BatchedReadRequest, iter_cucim_batched_read_regions

NOISE = b"cuInit Failed, error CUDA_ERROR_NO_DEVICE\\ncuFile initialization failed\\n"


class FakeCuImage:
    metadata = {
        "openslide": {"MPP": 0.5},
        "cucim": {"resolutions": {"level_dimensions": [[64, 64]], "level_downsamples": [1.0]}},
    }

    def __init__(self, path):
        os.write(2, NOISE)

    def read_region(self, *, location, size, level, num_workers, device):
        os.write(2, NOISE)

        def _decode():  # cuCIM decodes a batch as it is iterated
            for _ in location:
                os.write(2, NOISE)
                yield np.zeros((size[1], size[0], 3), dtype=np.uint8)

        return _decode()


class FakeCuCIM:
    CuImage = FakeCuImage


def _import_module(name, _original=m.importlib.import_module):
    if name != "cucim":
        return _original(name)
    os.write(2, NOISE)
    return FakeCuCIM


m.importlib.import_module = _import_module
regions = list(
    iter_cucim_batched_read_regions(
        image_path=Path("fake.svs"),
        requests=[
            BatchedReadRequest(location=(0, 0), size=(16, 16)),
            BatchedReadRequest(location=(16, 0), size=(16, 16)),
        ],
        level=0,
        num_workers=2,
    )
)
print(len(regions), faulthandler.is_enabled())
os.write(2, b"stderr is back\\n")
"""


@pytest.mark.parametrize(
    ("interpreter_flags", "faulthandler_enabled"),
    [((), False), (("-X", "faulthandler"), True)],
    ids=["faulthandler-off", "faulthandler-on"],
)
def test_batched_reads_hide_cuinit_and_cufile_noise_and_restore_stderr(
    interpreter_flags, faulthandler_enabled
):
    result = _run_child(NOISY_BATCHED_READS, interpreter_flags=interpreter_flags)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"2 {faulthandler_enabled}\n"
    assert result.stderr == "stderr is back\n"
