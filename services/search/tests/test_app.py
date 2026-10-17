import importlib


def test_app_importable() -> None:
    assert importlib.import_module("app").__doc__
