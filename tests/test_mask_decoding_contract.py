import pytest

import hs2p.mask as source_mask_mod
from hs2p.mask import Mask, TissueLabels


def test_tissue_backend_exception_fails_without_an_alternate_read(monkeypatch):
    opened_backends = []

    def fail_open(path, backend=None, **kwargs):
        opened_backends.append(backend)
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(source_mask_mod, "open_slide", fail_open)

    with pytest.raises(RuntimeError) as excinfo:
        Mask(
            path="/masks/decode-error.tif",
            labels=TissueLabels(background=0, tissue=1),
            backend="cucim",
        )

    assert opened_backends == ["cucim"]
    message = str(excinfo.value)
    assert "/masks/decode-error.tif" in message
    assert "backend=cucim" in message
    assert "codec unavailable" in message
