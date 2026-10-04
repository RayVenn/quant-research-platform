import pytest

from pairlab.registry.store import Registry, RegistryError


def _art(version, passed=True):
    return {"name": "s", "version": version, "created_at": "now",
            "validation": {"passed": passed, "oos": {"sharpe": 1.0}}}


def test_promotion_flow(tmp_path):
    reg = Registry(tmp_path)
    reg.register(_art("v1"))
    with pytest.raises(RegistryError, match="next stage"):
        reg.promote("s", "v1", "production")
    reg.promote("s", "v1", "staging")
    reg.promote("s", "v1", "production")
    assert reg.get("s", stage="production")["version"] == "v1"
    assert reg.stage_of("s", "v1") == "production"
    assert [h["action"] for h in reg.history("s")] == ["register", "promote", "promote"]


def test_failed_validation_blocks_promotion_unless_forced(tmp_path):
    reg = Registry(tmp_path)
    reg.register(_art("v2", passed=False))
    with pytest.raises(RegistryError, match="failed validation"):
        reg.promote("s", "v2", "staging")
    reg.promote("s", "v2", "staging", force=True, note="manual override")
    assert reg.history("s")[-1]["forced"] is True
