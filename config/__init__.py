from pathlib import Path
from omegaconf import OmegaConf


def load_config(dirname: str):
    config_filename = Path(dirname, "default.yaml")
    return OmegaConf.load(Path(__file__).parent.resolve() / config_filename)


default_extraction_config = load_config("extraction")
default_sampling_config = load_config("sampling")