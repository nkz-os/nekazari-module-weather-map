from app.records import build_agri_parcel_record

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
