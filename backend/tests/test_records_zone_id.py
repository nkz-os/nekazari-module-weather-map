"""build_agri_parcel_zone must derive a DISTINCT entity id from the zone's id.

zone_accumulator builds zone dicts with key "id"; a regression here makes every
zone collapse to ...:unknown (single entity + batch duplicate errors).
"""
from app.records import build_agri_parcel_zone

_GEOM = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}


def _build(zone):
    return build_agri_parcel_zone(
        tenant_id="acmefarm",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone=zone,
        geometry=_GEOM,
        metrics={},
        observed_at="2026-06-22T00:00:00Z",
    )


def test_zone_id_from_id_key():
    e = _build({"id": "zABC-e3-N"})
    assert e["id"] == "urn:ngsi-ld:AgriParcelZone:acmefarm:p1:zABC-e3-N"
    assert e["nkz:zoneId"]["value"] == "zABC-e3-N"


def test_zone_id_from_zone_id_key():
    e = _build({"zone_id": "zXYZ-e1-S"})
    assert e["id"].endswith(":zXYZ-e1-S")


def test_distinct_zones_distinct_ids():
    ids = {_build({"id": f"z-e{b}-N"})["id"] for b in range(5)}
    assert len(ids) == 5  # no collapse to :unknown


# --- zone content (type, version, rank, area, centroid) ---------------------
import json  # noqa: E402

from app.records import _geometry_centroid, zone_version  # noqa: E402


def test_zone_version_stable_and_sensitive():
    g1 = {"type": "Polygon", "coordinates": [[[0.1, 0.2], [0.1, 1.2], [1.1, 1.2], [0.1, 0.2]]]}
    g2 = json.loads(json.dumps({"coordinates": g1["coordinates"], "type": "Polygon"}))
    g3 = {"type": "Polygon", "coordinates": [[[0.10000001, 0.2], [0.1, 1.2], [1.1, 1.2], [0.10000001, 0.2]]]}
    g4 = {"type": "Polygon", "coordinates": [[[0.1, 0.2], [0.1, 1.3], [1.1, 1.2], [0.1, 0.2]]]}
    v = zone_version(g1)
    assert len(v) == 8 and int(v, 16) >= 0
    assert zone_version(g2) == v
    assert zone_version(g3) == v
    assert zone_version(g4) != v


def test_zone_content_attributes():
    e = build_agri_parcel_zone(
        tenant_id="acme", parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone={"id": "z1", "pixelCount": 250}, geometry=_GEOM, metrics={},
        observed_at="2026-06-22T00:00:00Z", zone_rank=3,
    )
    assert e["nkz:zoneType"]["value"] == "management"
    assert e["nkz:zoneVersion"]["value"] == zone_version(_GEOM)
    assert e["nkz:zoneRank"]["value"] == 3
    assert e["nkz:areaHa"]["value"] == 2.5
    assert "nkz:sensorNearby" not in e and "nkz:sensorDistanceM" not in e


def test_multipolygon_centroid_inside_a_part():
    mp = {"type": "MultiPolygon", "coordinates": [
        [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        [[[10, 10], [10, 11], [11, 11], [11, 10], [10, 10]]],
    ]}
    lon, lat = _geometry_centroid(mp)
    in_a = 0 <= lon <= 1 and 0 <= lat <= 1
    in_b = 10 <= lon <= 11 and 10 <= lat <= 11
    assert in_a or in_b


def test_zone_version_int_equals_float():
    a = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]}
    b = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}
    assert zone_version(a) == zone_version(b)
