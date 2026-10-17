import app


def test_app_importable() -> None:
    assert app.__version__ == "0.1.0"
