"""Test that cog_generator does NOT call fetch_agri_soil with an empty parcel id."""
import inspect
import app.cog_generator as cog


def test_cog_generator_does_not_call_fetch_agri_soil_with_empty_parcel():
    src = inspect.getsource(cog)
    assert 'fetch_agri_soil(tenant_id, "")' not in src
    assert "fetch_agri_soil(tenant_id, '')" not in src
