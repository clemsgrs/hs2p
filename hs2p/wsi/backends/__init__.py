from hs2p.wsi.backends.asap import ASAPReader
from hs2p.wsi.backends.cucim import (
    CUCIM_SUPPORTED_SUFFIXES,
    CuCIMReader,
    supports_cucim_path,
)
from hs2p.wsi.backends.openslide import OpenSlideReader
from hs2p.wsi.backends.vips import VIPSReader, supports_vips_path

__all__ = [
    "ASAPReader",
    "CUCIM_SUPPORTED_SUFFIXES",
    "CuCIMReader",
    "OpenSlideReader",
    "VIPSReader",
    "supports_cucim_path",
    "supports_vips_path",
]
