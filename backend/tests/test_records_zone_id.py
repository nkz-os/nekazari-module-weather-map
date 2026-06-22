"""build_agri_parcel_zone must derive a DISTINCT entity id from the zone's id.

zone_accumulator builds zone dicts with key "id"; a regression here makes every
zone collapse to ...:unknown (single entity + batch duplicate errors).
"""
from app.records import build_agri_parcel_zone

_GEOM = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}


def _build(zone):
    return build_agri_parcel_zone(
        tenant_id="montiko",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone=zone,
        geometry=_GEOM,
        metrics={},
        observed_at="2026-06-22T00:00:00Z",
    )


def test_zone_id_from_id_key():
    e = _build({"id": "zABC-e3-N"})
    assert e["id"] == "urn:ngsi-ld:AgriParcelZone:montiko:p1:zABC-e3-N"
    assert e["nkz:zoneId"]["value"] == "zABC-e3-N"


def test_zone_id_from_zone_id_key():
    e = _build({"zone_id": "zXYZ-e1-S"})
    assert e["id"].endswith(":zXYZ-e1-S")


def test_distinct_zones_distinct_ids():
    ids = {_build({"id": f"z-e{b}-N"})["id"] for b in range(5)}
    assert len(ids) == 5  # no collapse to :unknown
