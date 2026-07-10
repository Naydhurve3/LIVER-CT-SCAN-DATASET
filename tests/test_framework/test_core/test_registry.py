import pytest
from src.framework.core.registry import Registry, MODELS, LOSSES, METRICS
from src.framework.core.exceptions import RegistryError


def test_registry_register_and_get():
    r = Registry("test")
    @r.register("my_item")
    class MyItem:
        pass
    assert r.get("my_item") is MyItem


def test_registry_list():
    r = Registry("test")
    @r.register("a")
    class A:
        pass
    @r.register("b")
    class B:
        pass
    assert "a" in r.list()
    assert "b" in r.list()


def test_registry_duplicate_raises():
    r = Registry("test")
    @r.register("dup")
    class A:
        pass
    with pytest.raises(RegistryError):
        @r.register("dup")
        class B:
            pass


def test_registry_get_unknown_raises():
    r = Registry("test")
    with pytest.raises(RegistryError):
        r.get("nonexistent")


def test_registry_contains():
    r = Registry("test")
    @r.register("present")
    class A:
        pass
    assert "present" in r
    assert "absent" not in r


def test_global_registries_exist():
    assert MODELS.list() is not None
    assert LOSSES.list() is not None
    assert METRICS.list() is not None
