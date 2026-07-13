"""Import-compatibility smoke: the new package imports, and legacy root imports
still resolve during migration."""


def test_mobiletransformers_imports():
    import mobiletransformers

    assert mobiletransformers.__version__ == "0.1.0"


def test_legacy_parser_config_imports():
    # Root package must keep resolving throughout migration (shim added in config plan).
    from tools.parser_config import ARTIFACT_CONFIG, INFERENCE_CONFIG, TRAIN_CONFIG

    assert TRAIN_CONFIG == "TRAIN_BUILDER"
    assert ARTIFACT_CONFIG == "ARTIFACT_BUILDER"
    assert INFERENCE_CONFIG == "INFERENCE_BUILDER"
