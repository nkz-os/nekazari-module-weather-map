from app.records import build_agri_parcel_record, build_agri_parcel_zone

METRICS = {
    "solar_radiation": 21.5, "soil_moisture": 0.31, "eto": 4.2,
    "water_balance": -1.1, "frost_risk": 0.0, "temperature_avg": 18.3,
}


def test_record_is_agri_parcel_record_with_hasagriparcel():
    rec = build_agri_parcel_record(
        tenant_id="asociacion-allotarra",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        metrics=METRICS,
        observed_at="2026-06-10T10:00:00Z",
    )
    assert rec["type"] == "AgriParcelRecord"
    assert rec["id"].startswith("urn:ngsi-ld:AgriParcelRecord:weather-asociacion-allotarra-")
    assert rec["hasAgriParcel"]["object"] == "urn:ngsi-ld:AgriParcel:p1"
    assert rec["location"]["type"] == "GeoProperty"


def test_all_historized_metrics_are_flat_scalars():
    rec = build_agri_parcel_record(
        tenant_id="t", parcel_id="urn:ngsi-ld:AgriParcel:p1",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        metrics=METRICS, observed_at="2026-06-10T10:00:00Z",
    )
    for key in ("solarRadiation", "soilMoistureVwc", "eto", "waterBalance",
                "frostRisk", "airTemperatureAvg"):
        assert rec[key]["type"] == "Property"
        assert isinstance(rec[key]["value"], (int, float)), f"{key} must be scalar"
    assert "weatherStats" not in rec


SAMPLE_ZONE = {
    "zone_id": "zone-a",
    "elevation_mean": 150.0,
    "elevation_min": 120.0,
    "elevation_max": 180.0,
    "aspect_sector": "S",
    "pixel_count": 500,
}


SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [10.0, 21.0], [10.0, 20.0]]],
}


def test_build_agri_parcel_zone_minimal():
    entity = build_agri_parcel_zone(
        tenant_id="asociacion-allotarra",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone=SAMPLE_ZONE,
        geometry=SAMPLE_GEOMETRY,
        metrics={"tMin": 12.5, "tMax": 28.3, "eto": 4.2},
        observed_at="2026-06-20T10:00:00Z",
        area_ha=2.5,
    )
    # Correct type
    assert entity["type"] == "AgriParcelZone"

    # STATIC ID — NO timestamp component (contrast with AgriParcelRecord which has ts_compact)
    assert entity["id"] == (
        "urn:ngsi-ld:AgriParcelZone:asociacion-allotarra:p1:zone-a"
    ), "ID must be static, no timestamp"

    # Relationship uses hasAgriParcel (not refAgriParcel)
    assert entity["hasAgriParcel"]["object"] == "urn:ngsi-ld:AgriParcel:p1"
    assert entity["hasAgriParcel"]["type"] == "Relationship"
    assert "refAgriParcel" not in entity

    # Location GeoProperty
    assert entity["location"]["type"] == "GeoProperty"
    assert entity["location"]["value"] == SAMPLE_GEOMETRY

    # Zone metadata
    assert entity["nkz:zoneId"]["value"] == "zone-a"
    # Centroid of the rectangle (closing point excluded) = (10.5, 20.5)
    assert entity["nkz:centroid"]["value"] == [10.5, 20.5]
    assert entity["nkz:elevationMean"]["value"] == 150.0
    assert entity["nkz:elevationMin"]["value"] == 120.0
    assert entity["nkz:elevationMax"]["value"] == 180.0
    assert entity["nkz:aspectSector"]["value"] == "S"
    assert entity["nkz:pixelCount"]["value"] == 500

    # nkz:areaHa present when area_ha is provided
    assert entity["nkz:areaHa"]["value"] == 2.5

    # Metrics as flat scalar Properties
    assert entity["tMin"]["value"] == 12.5
    assert entity["tMax"]["value"] == 28.3
    assert entity["eto"]["value"] == 4.2
    for key in ("tMin", "tMax", "eto"):
        assert entity[key]["type"] == "Property"
        assert isinstance(entity[key]["value"], float)

    # dateObserved as DateTime
    assert entity["dateObserved"]["value"]["@value"] == "2026-06-20T10:00:00Z"

    # No sensor fields when not provided
    assert "nkz:sensorNearby" not in entity
    assert "nkz:sensorDistanceM" not in entity


def test_build_agri_parcel_zone_with_sensor():
    entity = build_agri_parcel_zone(
        tenant_id="t",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone=SAMPLE_ZONE,
        geometry=SAMPLE_GEOMETRY,
        metrics={"tMin": 10.0},
        observed_at="2026-06-20T10:00:00Z",
        sensor_nearby=True,
        sensor_distance_m=45.0,
    )
    assert entity["nkz:sensorNearby"]["value"] is True
    assert entity["nkz:sensorDistanceM"]["value"] == 45.0
    assert "nkz:areaHa" not in entity


def test_build_agri_parcel_zone_no_metrics():
    entity = build_agri_parcel_zone(
        tenant_id="t",
        parcel_id="urn:ngsi-ld:AgriParcel:p1",
        zone=SAMPLE_ZONE,
        geometry=SAMPLE_GEOMETRY,
        metrics={},
        observed_at="2026-06-20T10:00:00Z",
    )
    # Zone metadata still present
    assert entity["nkz:zoneId"]["value"] == "zone-a"
    assert entity["nkz:centroid"]["value"] == [10.5, 20.5]
    # No metric attributes
    for key in ("tMin", "tMax", "eto", "waterBalance"):
        assert key not in entity
    # nkz:areaHa absent when not provided
    assert "nkz:areaHa" not in entity


def test_zone_metrics_filter_non_scalar():
    entity = build_agri_parcel_zone(
        tenant_id="montiko",
        parcel_id="urn:ngsi-ld:AgriParcel:montiko:1",
        zone=SAMPLE_ZONE,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        metrics={"tMin": None, "tMax": [1, 2, 3], "eto": 4.2},
        observed_at="2026-06-20T00:00:00Z",
    )
    assert "tMin" not in entity
    assert "tMax" not in entity
    assert entity["eto"]["value"] == 4.2
