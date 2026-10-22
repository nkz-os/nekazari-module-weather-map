import asyncio
import app.zone_accumulator as za


def test_zone_main_uses_discovery(monkeypatch):
    seen = []
    monkeypatch.setattr(za, "discover_tenants", lambda var: ["t1", "t2"], raising=False)

    async def _fake_run(tenant_id):
        seen.append(tenant_id)

    monkeypatch.setattr(za, "run_for_tenant", _fake_run)
    asyncio.run(za.main())
    assert seen == ["t1", "t2"]
