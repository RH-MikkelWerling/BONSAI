from opera.compat import bonsai


def test_bonsai_compat_exports_core_symbols():
    assert bonsai.BonsaiEncoder is not None
    assert bonsai.BiGRU is not None
    assert callable(bonsai.dynamic_padding)
    assert callable(bonsai.binarize_outcomes)

