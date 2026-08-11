from pathlib import Path

from scripts.run_unseen_defect import get_defect_type


class DummySample:

    def __init__(self, image_path):
        self.image_path = image_path


def test_get_defect_type():

    sample = DummySample(
        Path(
            "data/raw/mvtec_anomaly_detection/"
            "bottle/test/broken_small/001.png"
        )
    )

    assert get_defect_type(sample) == "broken_small"