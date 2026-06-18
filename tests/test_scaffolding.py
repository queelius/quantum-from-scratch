def test_package_imports():
    import qfs

    assert isinstance(qfs.__version__, str)
    assert qfs.__version__
