import eval
import pipeline


def test_packages_importable() -> None:
    assert pipeline.__version__ == "0.1.0"
    assert eval.__version__ == "0.1.0"
