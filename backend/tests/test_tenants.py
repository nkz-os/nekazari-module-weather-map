from app import tenants


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("MONITORED_TENANTS", "a, b ,c")
    assert tenants.discover_tenants("MONITORED_TENANTS") == ["a", "b", "c"]


def test_db_discovery_when_no_override(monkeypatch):
    monkeypatch.delenv("MONITORED_TENANTS", raising=False)
    monkeypatch.setattr(tenants, "_discover_from_db", lambda: ["t1", "t2"])
    assert tenants.discover_tenants("MONITORED_TENANTS") == ["t1", "t2"]


def test_empty_fallback_never_default(monkeypatch):
    monkeypatch.delenv("COG_TENANTS", raising=False)
    monkeypatch.setattr(tenants, "_discover_from_db", lambda: [])
    assert tenants.discover_tenants("COG_TENANTS") == []


def test_postgres_url_built_from_parts(monkeypatch):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "h")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    assert tenants._postgres_url() == "postgresql://u:p@h:5432/d"


def test_postgres_url_empty_without_password(monkeypatch):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    assert tenants._postgres_url() == ""


def test_db_discovery_reads_the_installations_view(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    conn = MagicMock()
    cur = conn.cursor.return_value
    cur.fetchall.return_value = [("tenant-a",)]
    fake_psycopg2 = MagicMock()
    fake_psycopg2.connect.return_value = conn
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@db:5432/x")

    assert tenants._discover_from_db() == ["tenant-a"]
    assert "FROM module_installations" in cur.execute.call_args.args[0]
