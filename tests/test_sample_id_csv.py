import pandas as pd
import pytest

from hs2p.fileops import read_csv_keyed_by_sample_id


@pytest.mark.parametrize("sample_id", ["001", "NA", "nan", "0", "slide-1"])
def test_sample_ids_are_read_back_verbatim(tmp_path, sample_id):
    path = tmp_path / "rows.csv"
    pd.DataFrame(
        [{"sample_id": sample_id, "mask_path": None, "num_tiles": 4}]
    ).to_csv(path, index=False)

    df = read_csv_keyed_by_sample_id(path)

    assert df["sample_id"].tolist() == [sample_id]
    # other columns keep pandas' inference and missing-value handling
    assert df["mask_path"].isna().all()
    assert df["num_tiles"].tolist() == [4]
