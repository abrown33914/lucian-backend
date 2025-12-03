import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Tuple

import requests
import azure.functions as func
from azure.identity import ManagedIdentityCredential
from azure.digitaltwins.core import DigitalTwinsClient
from azure.core.exceptions import HttpResponseError
import math

from azure.storage.blob import BlobServiceClient  # NEW
import joblib
import numpy as np
from collections import defaultdict

app = func.FunctionApp()

MODEL_CACHE = {
    "model": None,
    "loaded": False,
}


def _get_env(name: str, default: str = "") -> str:
    """Get an app setting with a simple default."""
    v = os.getenv(name)
    return v if v else default

# -------------------- Blob listing utilities for API -------------------- #
def _blob_service() -> BlobServiceClient:
    conn = os.getenv("AzureWebJobsStorage", "").strip()
    if not conn:
        raise RuntimeError("AzureWebJobsStorage is not configured")
    return BlobServiceClient.from_connection_string(conn)

def _list_json_blobs(container: str, prefix: str = "") -> List[str]:
    svc = _blob_service()
    cc = svc.get_container_client(container)
    names = []
    for b in cc.list_blobs(name_starts_with=prefix):
        if b.name.endswith(".json"):
            names.append(b.name)
    return names

def _read_blob_json(container: str, name: str) -> dict:
    svc = _blob_service()
    bc = svc.get_blob_client(container=container, blob=name)
    data = bc.download_blob().readall()
    return json.loads(data.decode("utf-8"))

def _summarize_payload(payload: dict) -> dict:
    items = payload.get("items", [])
    timestamp = payload.get("timestamp")
    count = len(items)
    jams = []
    delays = []
    for it in items:
        fsd = (it.get("data") or {}).get("flowSegmentData", {})
        jf = fsd.get("jamFactor")
        dr = fsd.get("delayRatio")
        if isinstance(jf, (int, float)):
            jams.append(float(jf))
        if isinstance(dr, (int, float)):
            delays.append(float(dr))
    return {
        "timestamp": timestamp,
        "count": count,
        "jamFactorAvg": (sum(jams) / len(jams)) if jams else None,
        "delayRatioAvg": (sum(delays) / len(delays)) if delays else None,
    }


def build_query_url(lat: float, lon: float, subscription_key: str, zoom: int) -> str:
    """Build Azure Maps Traffic Flow Segment URL for a given point."""
    base = "https://atlas.microsoft.com/traffic/flow/segment/json"
    return (
        f"{base}?api-version=1.0"
        f"&style=absolute"
        f"&zoom={zoom}"
        f"&query={lat},{lon}"
        f"&subscription-key={subscription_key}"
    )


def build_incidents_url(lat: float, lon: float, radius_km: float, subscription_key: str, severity: str = "") -> str:
    """Build Azure Maps Traffic Incident URL around a point within a radius."""
    base = "https://atlas.microsoft.com/traffic/incident/json"
    # bbox can be used; for simplicity use `query` with radius and optional severity
    params = [
        "api-version=1.0",
        f"query={lat},{lon}",
        f"radius={int(radius_km * 1000)}",
        f"subscription-key={subscription_key}",
    ]
    if severity:
        params.append(f"severity={severity}")
    return base + "?" + "&".join(params)


def build_route_url(origin: Tuple[float, float], destination: Tuple[float, float], subscription_key: str, traffic: bool = True) -> str:
    """Build Azure Maps Route Directions URL between two points."""
    base = "https://atlas.microsoft.com/route/directions/json"
    traffic_flag = "true" if traffic else "false"
    return (
        f"{base}?api-version=1.0&query={origin[0]},{origin[1]}:{destination[0]},{destination[1]}"
        f"&traffic={traffic_flag}&subscription-key={subscription_key}"
    )


def fetch_with_retries(url: str, max_retries: int = 4, backoff: float = 0.8) -> dict:
    """HTTP GET with basic retry/backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            else:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            last_err = e
        time.sleep(backoff * (attempt + 1))
    raise last_err if last_err else RuntimeError("Unknown fetch error")


def snap_to_road(lat: float, lon: float, subscription_key: str) -> Tuple[float, float]:
    """
    Snap a point to the nearest road segment using Azure Maps NearestRoad API.
    If no nearby road is found (404), just return the original point quietly.
    """
    url = "https://atlas.microsoft.com/route/nearestRoad/json"
    params = {
        "api-version": "1.0",
        "query": f"{lat},{lon}",
        "subscription-key": subscription_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "addresses" in data and data["addresses"]:
                pos = data["addresses"][0]["position"]
                return float(pos["lat"]), float(pos["lon"])
            # 200 but no addresses → fall through to original point
        elif resp.status_code != 404:
            # Only warn on non-404 errors
            logging.warning(
                f"NearestRoad {resp.status_code} for {lat},{lon}: {resp.text[:150]}"
            )
    except Exception as e:
        logging.warning(f"Snap to road failed for {lat},{lon}: {e}")

    # Fallback: keep original if snapping fails or 404/no addresses
    return lat, lon


def generate_fgcu_points() -> List[Tuple[float, float]]:
    """
    Generate sample points around FGCU using known road intersections as anchors,
    plus small offsets around each anchor. This keeps us tightly on/near roads.
    """
    # Core anchors around FGCU & Ben Hill Griffin
    anchors = [
        (26.4666, -81.7726),  # Ben Hill Griffin Pkwy & FGCU Blvd
        (26.4886, -81.7803),  # Ben Hill Griffin Pkwy & Alico Rd
        (26.4419, -81.7706),  # Ben Hill Griffin Pkwy & Corkscrew Rd
        (26.4638, -81.7725),  # FGCU Blvd entrance
        (26.4609, -81.7787),  # Miromar / University Village
    ]

    # Small offsets (~60–70m grid) around each anchor
    offsets = [
        (0.0, 0.0),
        (0.0006, 0.0), (-0.0006, 0.0),
        (0.0, 0.0006), (0.0, -0.0006),
        (0.0006, 0.0006), (0.0006, -0.0006),
        (-0.0006, 0.0006), (-0.0006, -0.0006),
    ]

    pts = set()
    for alat, alon in anchors:
        for dlat, dlon in offsets:
            pts.add((round(alat + dlat, 6), round(alon + dlon, 6)))

    return sorted(list(pts))


def km_to_deg_lat(km: float) -> float:
    """Approximate conversion near FGCU for latitude degrees."""
    return km / 111.32


def km_to_deg_lon(km: float, lat: float) -> float:
    """Longitude degrees scale by cos(latitude)."""
    return km / (111.32 * math.cos(math.radians(lat)))


def outward_ring_points(center: Tuple[float, float], rings: int, points_per_ring: int, ring_step_km: float) -> List[Tuple[float, float]]:
    """Generate points in concentric rings around center.

    Each ring has points_per_ring evenly spaced around the circle.
    """
    clat, clon = center
    pts: List[Tuple[float, float]] = []
    for r in range(1, rings + 1):
        radius_km = r * ring_step_km
        dlat = km_to_deg_lat(radius_km)
        dlon = km_to_deg_lon(radius_km, clat)
        for k in range(points_per_ring):
            theta = 2 * math.pi * (k / points_per_ring)
            lat = clat + dlat * math.sin(theta)
            lon = clon + dlon * math.cos(theta)
            pts.append((round(lat, 6), round(lon, 6)))
    return pts


@app.schedule(schedule="0 */5 * * * *", arg_name="timer")
@app.blob_output(
    arg_name="outblob",
    path="%TRAFFIC_CONTAINER%/fortmyers/{datetime}.json",
    connection="AzureWebJobsStorage",
)
def collect_fort_myers(timer: func.TimerRequest, outblob: func.Out[str]) -> None:
    """
    Timer-triggered function: every 5 minutes, sample Azure Maps traffic
    around FGCU, snap to roads when possible, and write results into Blob Storage
    as a timestamped JSON blob.
    """
    logging.info("Starting Fort Myers traffic collection run")

    subscription_key = _get_env("AZURE_MAPS_SUBSCRIPTION_KEY")
    if not subscription_key:
        logging.error("AZURE_MAPS_SUBSCRIPTION_KEY is not configured")
        return

    zoom = int(_get_env("FORT_MYERS_ZOOM", "14"))  # higher zoom = more local detail

    # Generate base sample points near FGCU and outward rings to expand coverage
    base_points = generate_fgcu_points()
    center_lat = float(_get_env("FGCU_CENTER_LAT", "26.4666"))
    center_lon = float(_get_env("FGCU_CENTER_LON", "-81.7726"))
    rings = int(_get_env("FGCU_RINGS", "6"))
    points_per_ring = int(_get_env("FGCU_POINTS_PER_RING", "16"))
    ring_step_km = float(_get_env("FGCU_RING_STEP_KM", "0.5"))
    ring_points = outward_ring_points((center_lat, center_lon), rings, points_per_ring, ring_step_km)
    candidate_points = base_points + ring_points
    logging.info(f"Generated {len(base_points)} anchor-offset points and {len(ring_points)} ring points (total {len(candidate_points)})")

    results = []
    failures = 0
    seen_snapped = set()  # avoid duplicate segments for the same snapped coordinate
    seen_segments = set()  # deduplicate by segment identifier when available
    target_segments = int(_get_env("FGCU_TARGET_SEGMENTS", "50"))

    for (la, lo) in candidate_points:
        if len(seen_segments) >= target_segments:
            break
        # 1) Snap to nearest road
        snap_lat, snap_lon = snap_to_road(la, lo, subscription_key)

        # Round for dedup key
        key = (round(snap_lat, 5), round(snap_lon, 5))
        if key in seen_snapped:
            # Already hit this snapped position; skip duplicate segment
            continue
        seen_snapped.add(key)

        # 2) Query traffic flow segment at the snapped position
        url = build_query_url(snap_lat, snap_lon, subscription_key, zoom)
        try:
            data = fetch_with_retries(url)

            # NEW: extract segmentId / roadName from flowSegmentData for storage & later analytics
            fsd = data.get("flowSegmentData") or {}
            segment_id = fsd.get("segmentId")
            road_name = fsd.get("street") or fsd.get("roadName") or fsd.get("description") or ""

            results.append(
                {
                    "lat": snap_lat,
                    "lon": snap_lon,
                    "original": [la, lo],
                    "url": url,
                    "segmentId": segment_id,      # NEW
                    "roadName": road_name,        # NEW
                    "data": data,
                }
            )
            # Try to extract a segment identifier for dedup
            try:
                sid = None
                fsd = data.get("flowSegmentData")
                if isinstance(fsd, dict):
                    # Prefer explicit segmentId if present; else fall back to FRC or road class combo
                    sid = fsd.get("segmentId") or fsd.get("frc") or fsd.get("functionalRoadClass")
                if sid is None:
                    # Fallback: short hash of the payload to avoid exact duplicates
                    sid = (json.dumps(fsd or data)[:64])
                sid = str(sid)
                if sid not in seen_segments:
                    seen_segments.add(sid)
                    logging.info(f"Added segment {sid} ({len(seen_segments)}/{target_segments})")
            except Exception as e:
                logging.debug(f"Segment id extraction failed: {e}")
        except Exception as e:
            failures += 1
            logging.warning(
                f"Failed to fetch traffic for snapped {snap_lat},{snap_lon} "
                f"(original {la},{lo}): {e}"
            )

    now = datetime.now(timezone.utc)

    # For metadata, we can treat center as FGCU center (configurable)
    center_lat = float(_get_env("FORT_MYERS_LAT", "26.4635"))
    center_lon = float(_get_env("FORT_MYERS_LON", "-81.7728"))

    # Fetch traffic incidents around center within radius (configurable)
    incidents_radius_km = float(_get_env("FGCU_INCIDENTS_RADIUS_KM", "3"))
    incidents_severity = _get_env("FGCU_INCIDENTS_SEVERITY", "")  # e.g., 'minor','moderate','major'
    incidents_data = []
    try:
        inc_url = build_incidents_url(center_lat, center_lon, incidents_radius_km, subscription_key, incidents_severity)
        inc_json = fetch_with_retries(inc_url)
        incidents_data = inc_json.get("results") or inc_json.get("incidents") or []
    except Exception as e:
        logging.warning(f"Traffic incidents fetch failed: {e}")

    # Sample a few OD pairs for route delay ratios
    route_pairs = [
        ((26.4666, -81.7726), (26.4886, -81.7803)),  # FGCU -> Alico
        ((26.4666, -81.7726), (26.4419, -81.7706)),  # FGCU -> Corkscrew
        ((26.4638, -81.7725), (26.4682, -81.7677)),  # Campus -> I-75
    ]
    route_metrics = []
    for o, d in route_pairs:
        try:
            rt_url_live = build_route_url(o, d, subscription_key, traffic=True)
            rt_url_free = build_route_url(o, d, subscription_key, traffic=False)
            live = fetch_with_retries(rt_url_live)
            free = fetch_with_retries(rt_url_free)
            # Extract travel time in seconds; Azure Maps uses summary for routes
            def _duration_s(rjson):
                try:
                    return (
                        rjson["routes"][0]["summary"]["travelTimeInSeconds"]
                    )
                except Exception:
                    return None
            t_live = _duration_s(live)
            t_free = _duration_s(free)
            ratio = (t_live / t_free) if (t_live and t_free and t_free > 0) else None
            route_metrics.append({
                "origin": o,
                "destination": d,
                "liveSeconds": t_live,
                "freeflowSeconds": t_free,
                "delayRatio": ratio,
                "liveUrl": rt_url_live,
                "freeUrl": rt_url_free,
            })
        except Exception as e:
            logging.warning(f"Route sampling failed for {o}->{d}: {e}")

    payload = {
        "timestamp": now.isoformat(),
        "center": {"lat": center_lat, "lon": center_lon},
        "zoom": zoom,
        "source": "azure-maps-traffic-flow-segment",
        "area": "fgcu_corridor",
        "count": len(results),
        "uniqueSegmentCount": len(seen_segments),
        "targetSegments": target_segments,
        "rings": rings,
        "pointsPerRing": points_per_ring,
        "ringStepKm": ring_step_km,
        "failures": failures,
        "items": results,
        "incidentsRadiusKm": incidents_radius_km,
        "incidentsSeverity": incidents_severity,
        "incidents": incidents_data,
        "routes": route_metrics,
    }

    outblob.set(json.dumps(payload))
    logging.info(
        f"Wrote blob with {len(results)} unique snapped items (failures={failures})"
    )

    # Inline ADT upsert as a fallback to ensure twins are created even if blob trigger is misconfigured
    try:
        client = get_adt_client()
        timestamp_str = payload.get("timestamp")
        updated = 0
        errors = 0
        for item in results:
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
                flow = (item.get("data") or {}).get("flowSegmentData", {})
                _adt_upsert_road_segment(client, lat, lon, flow, timestamp_str)
                updated += 1
            except Exception as e:
                errors += 1
                logging.error(f"Inline ADT upsert failed for item {item}: {e}")
        logging.info(f"Inline ADT upsert completed: updated={updated}, errors={errors}")
    except Exception as e:
        logging.error(f"Inline ADT upsert skipped due to client error: {e}")

def get_ml_model() -> object:
    """
    Load the delay ratio forecast model from Blob Storage once and cache it.
    Expects:
      - AzureWebJobsStorage connection string
      - ML_MODELS_CONTAINER app setting (default 'ml-models')
      - ML_MODEL_BLOB_NAME app setting (default 'delayratio_forecast_rf.joblib')
    """
    if MODEL_CACHE["loaded"] and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"]

    conn_str = os.getenv("AzureWebJobsStorage", "").strip()
    container_name = _get_env("ML_MODELS_CONTAINER", "ml-models")
    blob_name = _get_env("ML_MODEL_BLOB_NAME", "delayratio_forecast_rf.joblib")

    if not conn_str:
        raise RuntimeError("AzureWebJobsStorage is not configured.")

    try:
        service = BlobServiceClient.from_connection_string(conn_str)
        container = service.get_container_client(container_name)
        blob_client = container.get_blob_client(blob_name)

        import io
        buf = io.BytesIO(blob_client.download_blob().readall())
        model = joblib.load(buf)

        MODEL_CACHE["model"] = model
        MODEL_CACHE["loaded"] = True
        logging.info(f"ML model loaded from blob: container={container_name}, blob={blob_name}")
        return model
    except Exception as e:
        logging.error(f"Failed to load ML model from blob: {e}")
        raise


_pavement_model_cache = None


def get_pavement_model():
    """
    Load the pavement model from Blob Storage (cached in memory).
    Uses:
      - AzureWebJobsStorage
      - PAVEMENT_MODEL_CONTAINER
      - PAVEMENT_MODEL_BLOB
    """
    global _pavement_model_cache
    if _pavement_model_cache is not None:
        return _pavement_model_cache

    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        raise RuntimeError("AzureWebJobsStorage is not configured.")

    container_name = os.getenv("PAVEMENT_MODEL_CONTAINER", "models")
    blob_name = os.getenv("PAVEMENT_MODEL_BLOB", "pavement_model.joblib")

    logging.info(
        f"Loading pavement model from container='{container_name}', blob='{blob_name}'"
    )

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(container_name)
    blob_data = container.download_blob(blob_name).readall()

    import io
    _pavement_model_cache = joblib.load(io.BytesIO(blob_data))
    return _pavement_model_cache


# -------------------- Digital Twins helper -------------------- #
def get_adt_client() -> DigitalTwinsClient:
    """Create a DigitalTwinsClient using the Function App's managed identity."""
    adt_url = os.getenv("ADT_SERVICE_URL", "").strip()
    if not adt_url:
        raise RuntimeError("ADT_SERVICE_URL is not configured.")
    cred = ManagedIdentityCredential()
    return DigitalTwinsClient(adt_url, cred)


def _adt_upsert_road_segment(client: DigitalTwinsClient, lat: float, lon: float, flow: dict, timestamp: str) -> None:
    """Map Azure Maps flow segment to RoadSegment twin and upsert in ADT."""
    current_speed = flow.get("currentSpeed")
    free_flow_speed = flow.get("freeFlowSpeed")
    confidence = flow.get("confidence")
    jam_factor = flow.get("jamFactor")
    current_tt = flow.get("currentTravelTime")
    freeflow_tt = flow.get("freeFlowTravelTime")
    frc = flow.get("frc")
    closure = flow.get("roadClosure")

    # NEW: extract segmentId and roadName from flow data (when present)
    segment_id = flow.get("segmentId")
    road_name = flow.get("street") or flow.get("roadName") or flow.get("description") or ""

    coords = (flow.get("coordinates") or {}).get("coordinate") or []
    polyline = [[c.get("longitude"), c.get("latitude")] for c in coords if isinstance(c, dict)]
    geojson_line = None
    try:
        if polyline:
            # Downsample coordinates to keep GeoJSON under ADT 4096-byte string limit
            coords_ds = polyline[:]
            # Keep endpoints always; iteratively reduce by taking every 2nd point
            def build(coords_list):
                return json.dumps({"type": "LineString", "coordinates": coords_list})
            gj = build(coords_ds)
            while len(gj.encode("utf-8")) > 4000 and len(coords_ds) > 10:
                coords_ds = [coords_ds[0]] + coords_ds[1:-1:2] + [coords_ds[-1]]
                gj = build(coords_ds)
            geojson_line = gj if len(gj.encode("utf-8")) <= 4096 else None
    except Exception:
        geojson_line = None
    delay_ratio = None
    try:
        if current_tt and freeflow_tt and freeflow_tt > 0:
            delay_ratio = float(current_tt) / float(freeflow_tt)
    except Exception:
        delay_ratio = None

    twin_id = f"road_{round(float(lat), 5)}_{round(float(lon), 5)}"
    twin_body = {
        "$metadata": {"$model": "dtmi:fgcu:traffic:RoadSegment;2"},
        "name": twin_id,
        "latitude": float(lat),
        "longitude": float(lon),
        "segmentId": str(segment_id) if segment_id is not None else "",  # NEW
        "roadName": road_name,  # NEW
        "currentSpeed": float(current_speed) if current_speed is not None else 0.0,
        "freeFlowSpeed": float(free_flow_speed) if free_flow_speed is not None else 0.0,
        "jamFactor": float(jam_factor) if jam_factor is not None else 0.0,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "currentTravelTime": int(current_tt) if current_tt is not None else 0,
        "freeFlowTravelTime": int(freeflow_tt) if freeflow_tt is not None else 0,
        "delayRatio": float(delay_ratio) if delay_ratio is not None else 0.0,
        "frc": str(frc) if frc is not None else "",
        "roadClosure": bool(closure) if closure is not None else False,
        "coordinatesGeoJson": geojson_line if geojson_line is not None else "",
        "lastUpdatedUtc": timestamp,
    }
    client.upsert_digital_twin(twin_id, twin_body)


# -------------------- Blob-trigger → ADT function -------------------- #
@app.blob_trigger(
    arg_name="inblob",
    path="%TRAFFIC_CONTAINER%/fortmyers/{name}",
    connection="AzureWebJobsStorage",
)
def process_fort_myers_blob(inblob: func.InputStream):
    """
    Triggered whenever a new Fort Myers traffic blob is created.
    Reads the latest snapshot and upserts RoadSegment twins in Azure Digital Twins.
    """
    logging.info(
        f"process_fort_myers_blob triggered, name={inblob.name}, size={inblob.length} bytes"
    )
    # 1) Parse JSON from blob
    try:
        raw = inblob.read()
        payload = json.loads(raw)
    except Exception as e:
        logging.error(f"Failed to parse traffic blob as JSON: {e}")
        return
    items = payload.get("items", [])
    timestamp = payload.get("timestamp")
    if not items:
        logging.warning("Traffic blob has no items; nothing to update.")
        return
    # 2) Build ADT client
    try:
        client = get_adt_client()
    except Exception as e:
        logging.error(f"Could not create ADT client: {e}")
        return
    updated = 0
    errors = 0
    # 3) Loop through each traffic item and upsert a twin
    for item in items:
        try:
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            flow = item.get("data", {}).get("flowSegmentData", {})
            _adt_upsert_road_segment(client, lat, lon, flow, timestamp)
            updated += 1
        except Exception as e:
            errors += 1
            if isinstance(e, HttpResponseError):
                logging.error(
                    f"ADT upsert failed (status={getattr(e, 'status_code', 'unknown')}): {e.message}"
                )
            else:
                logging.error(f"Failed to upsert twin for item {item}: {e}")
    logging.info(
        f"process_fort_myers_blob finished: updated={updated}, errors={errors}"
    )


# -------------------- Diagnosed upsert (validates then upserts) -------------------- #
@app.route(route="traffic/adt/upsert-diagnosed", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def http_upsert_diagnosed(req: func.HttpRequest) -> func.HttpResponse:
    """Validate ADT connectivity/model like diagnose, then upsert items from collector payload."""
    try:
        payload = req.get_json()
    except Exception:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    items = payload.get("items", [])
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    if not isinstance(items, list) or not items:
        return func.HttpResponse("Payload missing items", status_code=400)

    # Validate ADT connectivity and model presence
    try:
        client = get_adt_client()
        client.get_model("dtmi:fgcu:traffic:RoadSegment;2")
    except Exception as e:
        return func.HttpResponse(f"ADT validation failed: {e}", status_code=500)

    updated = 0
    errors = 0
    for item in items:
        try:
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            flow = item.get("data", {}).get("flowSegmentData", {})
            _adt_upsert_road_segment(client, lat, lon, flow, timestamp)
            updated += 1
        except Exception as e:
            errors += 1
            if isinstance(e, HttpResponseError):
                logging.error(
                    f"ADT upsert failed (status={getattr(e, 'status_code', 'unknown')}): {e.message}"
                )
            else:
                logging.error(f"Failed to upsert twin for item {item}: {e}")

    return func.HttpResponse(json.dumps({"updated": updated, "errors": errors}), mimetype="application/json", status_code=200)


# -------------------- ADT diagnostics -------------------- #
@app.route(route="traffic/adt/diagnose", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_adt_diagnose(req: func.HttpRequest) -> func.HttpResponse:
    """Diagnose ADT connectivity, model presence, and permissions.

    - Checks `ADT_SERVICE_URL` env is set
    - Connects with managed identity
    - Verifies model `dtmi:fgcu:traffic:RoadSegment;2` exists
    - Attempts a test twin upsert and delete
    Returns a JSON report with statuses and errors.
    """
    report = {
        "adtUrl": os.getenv("ADT_SERVICE_URL", ""),
        "modelId": "dtmi:fgcu:traffic:RoadSegment;2",
        "envPresent": bool(os.getenv("ADT_SERVICE_URL")),
        "connected": False,
        "modelExists": False,
        "upsertTest": False,
        "deleteTest": False,
        "error": None,
    }

    try:
        client = get_adt_client()
        report["connected"] = True
        # Check model existence
        try:
            client.get_model(report["modelId"])  # will raise if not found
            report["modelExists"] = True
        except Exception as me:
            report["error"] = f"Model check failed: {me}"
            return func.HttpResponse(json.dumps(report), mimetype="application/json", status_code=200)

        # Try upsert a temporary twin
        twin_id = "diagnostic_twin_fgcu"
        twin_body = {
            "$metadata": {"$model": report["modelId"]},
            "name": twin_id,
            "latitude": 26.4666,
            "longitude": -81.7726,
            "segmentId": "",               # NEW: keep in sync with model
            "roadName": "",                # NEW
            "currentSpeed": 0.0,
            "freeFlowSpeed": 0.0,
            "jamFactor": 0.0,
            "confidence": 0.0,
            "currentTravelTime": 0,
            "freeFlowTravelTime": 0,
            "delayRatio": 0.0,
            "frc": "",
            "roadClosure": False,
            "coordinatesGeoJson": "",
            "lastUpdatedUtc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            client.upsert_digital_twin(twin_id, twin_body)
            report["upsertTest"] = True
        except Exception as ue:
            report["error"] = f"Upsert test failed: {ue}"
            return func.HttpResponse(json.dumps(report), mimetype="application/json", status_code=200)

        # Try delete
        try:
            client.delete_digital_twin(twin_id)
            report["deleteTest"] = True
        except Exception as de:
            # Not critical; note error
            report["error"] = f"Delete test failed: {de}"
        return func.HttpResponse(json.dumps(report), mimetype="application/json", status_code=200)
    except Exception as e:
        report["error"] = str(e)
        return func.HttpResponse(json.dumps(report), mimetype="application/json", status_code=200)
    

    # -------------------- Export history from blobs (HTTP) -------------------- #
@app.route(
    route="traffic/history/export",
    methods=["GET"],
    auth_level=func.AuthLevel.FUNCTION,
)
def http_export_history(req: func.HttpRequest) -> func.HttpResponse:
    """
    Flatten recent Fort Myers traffic blobs into a CSV (for analytics / ML).
    Usage:
      GET /api/traffic/history/export?blobs=50

    - Reads the last N blobs from the TRAFFIC_CONTAINER/fortmyers folder
    - For each item, outputs one CSV row with key metrics.
    """
    try:
        max_blobs = int(req.params.get("blobs", "50"))
    except ValueError:
        max_blobs = 50

    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        return func.HttpResponse(
            "AzureWebJobsStorage is not configured.", status_code=500
        )

    container_name = _get_env("TRAFFIC_CONTAINER", "traffic")
    prefix = "fortmyers/"

    try:
        service = BlobServiceClient.from_connection_string(conn_str)
        container = service.get_container_client(container_name)
    except Exception as e:
        return func.HttpResponse(
            f"Failed to connect to blob container: {e}", status_code=500
        )

    # List blob names under fortmyers/, newest first
    try:
        names = [b.name for b in container.list_blobs(name_starts_with=prefix)]
        names.sort(reverse=True)
        names = names[:max_blobs]
    except Exception as e:
        return func.HttpResponse(
            f"Failed to list blobs: {e}", status_code=500
        )

    import io
    import csv
    from datetime import datetime as dt

    output = io.StringIO()
    writer = csv.writer(output)

    # CSV header
    writer.writerow(
        [
            "snapshotTimestamp",
            "blobName",
            "twinId",
            "lat",
            "lon",
            "currentSpeed",
            "freeFlowSpeed",
            "jamFactor",
            "currentTravelTime",
            "freeFlowTravelTime",
        ]
    )

    rows = 0

    for name in names:
        try:
            blob_bytes = container.download_blob(name).readall()
            snap = json.loads(blob_bytes)
        except Exception as e:
            logging.warning(f"Failed to read blob {name}: {e}")
            continue

        snap_ts = snap.get("timestamp")
        items = snap.get("items", [])
        for item in items:
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
                twin_id = f"road_{round(lat, 5)}_{round(lon, 5)}"
                flow = (item.get("data") or {}).get("flowSegmentData", {}) or {}
                current_speed = flow.get("currentSpeed")
                free_flow_speed = flow.get("freeFlowSpeed")
                jam_factor = flow.get("jamFactor")
                current_tt = flow.get("currentTravelTime")
                freeflow_tt = flow.get("freeFlowTravelTime")

                writer.writerow(
                    [
                        snap_ts,
                        name,
                        twin_id,
                        lat,
                        lon,
                        current_speed,
                        free_flow_speed,
                        jam_factor,
                        current_tt,
                        freeflow_tt,
                    ]
                )
                rows += 1
            except Exception as e:
                logging.warning(f"Failed to flatten item from {name}: {e}")
                continue

    csv_text = output.getvalue()
    logging.info(f"Exported {rows} rows from {len(names)} blobs")

    headers = {
        "Content-Type": "text/csv",
        "Content-Disposition": 'attachment; filename="traffic_history_fgcu.csv"',
    }
    return func.HttpResponse(csv_text, status_code=200, headers=headers)

def _run_pavement_forecast_internal() -> dict:
    """
    Core logic to:
      - get ADT client
      - load pavement ML model
      - query RoadSegment;2 twins
      - predict future pavement stress index
      - upsert PavementForecast twins

    Returns: {"updated": int, "errors": int}
    """
    # 1) ADT client
    try:
        client = get_adt_client()
    except Exception as e:
        logging.error(f"Pavement forecast: ADT client error: {e}")
        return {"updated": 0, "errors": 1}

    # 2) Load pavement model
    try:
        model = get_pavement_model()
    except Exception as e:
        logging.error(f"Pavement forecast: ML model load error: {e}")
        return {"updated": 0, "errors": 1}

    # 3) Query current RoadSegment;2 twins
    query = "SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')"
    try:
        road_twins = list(client.query_twins(query))
    except Exception as e:
        logging.error(f"Pavement forecast: ADT query error: {e}")
        return {"updated": 0, "errors": 1}

    if not road_twins:
        logging.info("Pavement forecast: no RoadSegment twins found.")
        return {"updated": 0, "errors": 0}

    updated = 0
    errors = 0

    for twin in road_twins:
        try:
            props = twin
            twin_id = props["$dtId"]

            # Features must match training columns:
            delay_ratio = float(props.get("delayRatio", 1.0))
            jam_factor = float(props.get("jamFactor", 0.0))
            current_speed = float(props.get("currentSpeed", 0.0))
            free_flow_speed = float(props.get("freeFlowSpeed", 0.0))
            lat = float(props.get("latitude", 0.0))
            lon = float(props.get("longitude", 0.0))

            now = datetime.now(timezone.utc)
            hour = now.hour
            dow = now.weekday()

            features = np.array(
                [[
                    delay_ratio,
                    jam_factor,
                    current_speed,
                    free_flow_speed,
                    hour,
                    dow,
                    lat,
                    lon,
                ]]
            )

            predicted_stress = float(model.predict(features)[0])

            # Map stress index (≈ 0..1+ ) to a 0–100 "health" score
            # Higher stress -> lower condition score
            condition_score = 100.0 * max(0.0, 1.0 - predicted_stress)
            if condition_score < 0.0:
                condition_score = 0.0
            if condition_score > 100.0:
                condition_score = 100.0

            forecast_id = f"pavementForecast_{twin_id}"
            forecast_body = {
                "$metadata": {"$model": "dtmi:fgcu:traffic:PavementForecast;1"},
                "segmentId": twin_id,
                "generatedAtUtc": now.isoformat(),
                "horizonMinutes": 5,
                "predictedPavementStressIndex": predicted_stress,
                "predictedConditionScore": condition_score,
                "predictionMethod": "rf-regression-pavementStress-next",
                "modelVersion": "local-pavement-rf-v1",
            }

            client.upsert_digital_twin(forecast_id, forecast_body)
            updated += 1

        except Exception as e:
            logging.error(
                f"Pavement forecast failed for twin {twin.get('$dtId', 'unknown')}: {e}"
            )
            errors += 1

    logging.info(f"Pavement forecast complete: updated={updated}, errors={errors}")
    return {"updated": updated, "errors": errors}


def _run_pavement_aggregate_internal(max_blobs: int = 50) -> dict:
    """
    Core logic to aggregate traffic history into PavementSegment twins.
    Returns a summary dict with counts and errors.
    """
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        logging.error("AzureWebJobsStorage is not configured.")
        return {
            "segmentsAggregated": 0,
            "twinsUpdated": 0,
            "errors": 1,
            "blobsUsed": 0,
        }

    container_name = _get_env("TRAFFIC_CONTAINER", "traffic")
    prefix = "fortmyers/"

    # Connect to blob container
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container = blob_service.get_container_client(container_name)
    except Exception as e:
        logging.error(f"Failed to connect to blob container: {e}")
        return {
            "segmentsAggregated": 0,
            "twinsUpdated": 0,
            "errors": 1,
            "blobsUsed": 0,
        }

    # Newest → oldest blobs
    try:
        blob_names = [b.name for b in container.list_blobs(name_starts_with=prefix)]
        blob_names.sort(reverse=True)
        blob_names = blob_names[:max_blobs]
    except Exception as e:
        logging.error(f"Failed to list blobs: {e}")
        return {
            "segmentsAggregated": 0,
            "twinsUpdated": 0,
            "errors": 1,
            "blobsUsed": 0,
        }

    stats = defaultdict(lambda: {"jam": [], "delay": []})
    total_items = 0

    for name in blob_names:
        try:
            blob_bytes = container.download_blob(name).readall()
            snap = json.loads(blob_bytes)
        except Exception as e:
            logging.warning(f"Failed to read blob {name}: {e}")
            continue

        items = snap.get("items", [])
        for item in items:
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
                twin_id = f"road_{round(lat, 5)}_{round(lon, 5)}"

                flow = (item.get("data") or {}).get("flowSegmentData", {}) or {}

                jam = flow.get("jamFactor")
                current_tt = flow.get("currentTravelTime")
                freeflow_tt = flow.get("freeFlowTravelTime")

                # Delay ratio as float
                delay = None
                try:
                    if (
                        current_tt is not None
                        and freeflow_tt is not None
                        and freeflow_tt > 0
                    ):
                        delay = float(current_tt) / float(freeflow_tt)
                except Exception:
                    delay = None

                if jam is not None:
                    stats[twin_id]["jam"].append(float(jam))
                if delay is not None:
                    stats[twin_id]["delay"].append(float(delay))

                total_items += 1
            except Exception as e:
                logging.warning(f"Failed to process item in {name}: {e}")
                continue

    logging.info(
        f"Pavement aggregate raw stats: {len(stats)} segments from {len(blob_names)} blobs, {total_items} items."
    )

    # Connect to ADT
    try:
        client = get_adt_client()
    except Exception as e:
        logging.error(f"ADT client error: {e}")
        return {
            "segmentsAggregated": len(stats),
            "twinsUpdated": 0,
            "errors": 1,
            "blobsUsed": len(blob_names),
        }

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    errors = 0

    for twin_id, agg in stats.items():
        try:
            jams = agg["jam"]
            delays = agg["delay"]

            if not jams and not delays:
                continue

            avg_jam = sum(jams) / len(jams) if jams else 0.0
            peak_jam = max(jams) if jams else 0.0
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            peak_delay = max(delays) if delays else 0.0

            # Simple stress index (you can tweak later)
            stress = float(0.5 * avg_delay + 0.5 * (peak_jam / 10.0))

            # Try to fetch the RoadSegment;2 twin for metadata
            try:
                road_twin = client.get_digital_twin(twin_id)
            except Exception as e:
                logging.warning(f"Could not fetch RoadSegment twin {twin_id}: {e}")
                road_twin = {}

            # Get segmentId, but fall back to twin_id if it's missing or empty
            raw_segment_id = road_twin.get("segmentId")
            if raw_segment_id is not None and str(raw_segment_id).strip() != "":
                segment_id = str(raw_segment_id)
            else:
                segment_id = twin_id  # fallback: use road twin id

            road_name = road_twin.get("roadName", "")
            frc = road_twin.get("frc", "")
            coords = road_twin.get("coordinatesGeoJson", "")

            # Make the Pavement twin id UNIQUE per road segment
            pavement_id = f"pavement_{twin_id}"

            pavement_body = {
                "$metadata": {"$model": "dtmi:fgcu:traffic:PavementSegment;1"},
                "segmentId": segment_id,
                "roadName": road_name,
                "frc": frc,
                "coordinatesGeoJson": coords,
                "avgJamFactor": float(avg_jam),
                "peakHourJamFactor": float(peak_jam),
                "avgDelayRatio": float(avg_delay),
                "peakDelayRatio": float(peak_delay),
                "pavementStressIndex": float(stress),
                "stressIndexMethod": "0.5*avgDelayRatio + 0.5*(peakJamFactor/10.0)",
                "lastAggregatedUtc": now,
            }

            client.upsert_digital_twin(pavement_id, pavement_body)
            updated += 1

        except Exception as e:
            logging.error(
                f"Failed to upsert PavementSegment twin for {twin_id}: {e}"
            )
            errors += 1

    logging.info(
        f"Pavement aggregation complete: segments={len(stats)}, updated={updated}, errors={errors}, blobs={len(blob_names)}"
    )

    return {
        "segmentsAggregated": len(stats),
        "twinsUpdated": updated,
        "errors": errors,
        "blobsUsed": len(blob_names),
    }


def _run_local_forecast_internal() -> dict:
    """
    Core logic to:
      - get ADT client
      - load ML model
      - query RoadSegment twins
      - predict delayRatio_future
      - upsert SegmentForecast twins

    Returns: {"updated": int, "errors": int}
    """
    # 1) Get ADT client
    try:
        client = get_adt_client()
    except Exception as e:
        logging.error(f"ADT client error: {e}")
        return {"updated": 0, "errors": 1}

    # 2) Load ML model from Blob (with cache)
    try:
        model = get_ml_model()
    except Exception as e:
        logging.error(f"ML model load error: {e}")
        return {"updated": 0, "errors": 1}

    # 3) Query current RoadSegment;2 twins
    query = (
        "SELECT * FROM digitaltwins t "
        "WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')"
    )
    try:
        road_twins = list(client.query_twins(query))
    except Exception as e:
        logging.error(f"ADT query error: {e}")
        return {"updated": 0, "errors": 1}

    if not road_twins:
        logging.info("No RoadSegment twins found for forecasting.")
        return {"updated": 0, "errors": 0}

    updated = 0
    errors = 0

    for twin in road_twins:
        try:
            twin_id = twin["$dtId"]

            delay_ratio = float(twin.get("delayRatio", 1.0))
            current_speed = float(twin.get("currentSpeed", 0.0))
            free_flow_speed = float(twin.get("freeFlowSpeed", 0.0))
            lat = float(twin.get("latitude", 0.0))
            lon = float(twin.get("longitude", 0.0))

            now = datetime.now(timezone.utc)
            hour = now.hour
            dow = now.weekday()

            features = np.array(
                [
                    [
                        delay_ratio,
                        current_speed,
                        free_flow_speed,
                        hour,
                        dow,
                        lat,
                        lon,
                    ]
                ]
            )

            predicted = float(model.predict(features)[0])

            forecast_id = f"forecast_{twin_id}"
            forecast_body = {
                "$metadata": {"$model": "dtmi:fgcu:traffic:SegmentForecast;1"},
                "segmentId": twin_id,
                "generatedAtUtc": now.isoformat(),
                "horizonMinutes": 5,
                "predictedJamFactor": 0.0,  # focusing on delayRatio
                "predictedDelayRatio": predicted,
                "predictedStressIndex": 0.0,
                "modelVersion": "local-automl-rf-v1",
            }

            client.upsert_digital_twin(forecast_id, forecast_body)
            updated += 1

        except Exception as e:
            logging.error(
                f"Forecast failed for twin {twin.get('$dtId', 'unknown')}: {e}"
            )
            errors += 1

    logging.info(f"Local ML forecast complete: updated={updated}, errors={errors}")
    return {"updated": updated, "errors": errors}


@app.route(
    route="traffic/forecast/local-ml",
    methods=["POST"],
    auth_level=func.AuthLevel.FUNCTION,
)
def http_apply_local_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manual trigger (HTTP) to run the local ML forecast.
    """
    result = _run_local_forecast_internal()
    return func.HttpResponse(
        json.dumps(result),
        mimetype="application/json",
        status_code=200,
    )


@app.schedule(schedule="0 */5 * * * *", arg_name="timer_forecast")
def scheduled_local_forecast(timer_forecast: func.TimerRequest) -> None:
    """
    Timer-triggered forecast:
    Runs every 5 minutes and updates SegmentForecast twins
    based on the latest RoadSegment;2 data.
    """
    logging.info("Starting scheduled local ML forecast run")
    result = _run_local_forecast_internal()
    logging.info(
        f"Scheduled forecast run finished: updated={result.get('updated')}, "
        f"errors={result.get('errors')}"
    )


@app.route(
    route="pavement/aggregate",
    methods=["GET"],
    auth_level=func.AuthLevel.FUNCTION,
)
def http_pavement_aggregate(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manual / debug trigger for pavement aggregation.
    Example:
      GET /api/pavement/aggregate?blobs=50
    """
    try:
        max_blobs = int(req.params.get("blobs", "50"))
    except Exception:
        max_blobs = 50

    result = _run_pavement_aggregate_internal(max_blobs=max_blobs)
    return func.HttpResponse(
        json.dumps(result),
        mimetype="application/json",
        status_code=200,
    )

@app.schedule(
    schedule="0 */5 * * * *",   # every 5 minutes
    arg_name="timer_pavement",
    run_on_startup=True,        # run once when function host starts
    use_monitor=True,
)
def scheduled_pavement_aggregate(timer_pavement: func.TimerRequest) -> None:
    """
    Timer-triggered PavementSegment aggregation (every 5 minutes).
    Uses the shared _run_pavement_aggregate_internal() helper.
    """
    logging.info(">>> [PAVEMENT TIMER] Starting scheduled pavement aggregation run...")
    try:
        max_blobs = int(os.getenv("PAVEMENT_BLOBS", "50"))
    except Exception:
        max_blobs = 50

    result = _run_pavement_aggregate_internal(max_blobs=max_blobs)
    logging.info(f">>> [PAVEMENT TIMER] Result: {result}")

@app.route(
    route="pavement/forecast/local-ml",
    methods=["POST"],
    auth_level=func.AuthLevel.FUNCTION,
)
def http_pavement_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manual trigger (HTTP) for pavement forecast.
    Call this from Portal / Postman to force a run.
    """
    result = _run_pavement_forecast_internal()
    return func.HttpResponse(
        json.dumps(result),
        mimetype="application/json",
        status_code=200,
    )


@app.schedule(
    schedule="0 */5 * * * *",        # every 5 minutes
    arg_name="timer_pavement_forecast",
    run_on_startup=False,
    use_monitor=True,
)
def scheduled_pavement_forecast(timer_pavement_forecast: func.TimerRequest) -> None:
    """
    Timer-triggered pavement forecast.
    Runs every 15 minutes and updates PavementForecast twins
    based on the latest RoadSegment;2 data.
    """
    logging.info(">>> [PAVEMENT FORECAST TIMER] Starting scheduled pavement forecast...")
    result = _run_pavement_forecast_internal()
    logging.info(
        f">>> [PAVEMENT FORECAST TIMER] Result: updated={result.get('updated')}, "
        f"errors={result.get('errors')}"
    )

# -------------------- Traffic data API (Blob-backed) -------------------- #
@app.route(route="traffic/latest", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_traffic_latest(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container = _get_env("TRAFFIC_CONTAINER", "traffic")
        prefix = req.params.get("prefix", "fortmyers/")
        names = _list_json_blobs(container, prefix)
        if not names:
            return func.HttpResponse("No data", status_code=404)
        names.sort()
        latest = names[-1]
        payload = _read_blob_json(container, latest)
        summary = _summarize_payload(payload)
        summary["blobPath"] = latest
        return func.HttpResponse(json.dumps(summary), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"traffic/latest failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)


@app.route(route="traffic/history", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_traffic_history(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container = _get_env("TRAFFIC_CONTAINER", "traffic")
        prefix = req.params.get("prefix", "fortmyers/")
        limit = int(req.params.get("limit", "10"))
        names = _list_json_blobs(container, prefix)
        if not names:
            return func.HttpResponse("No data", status_code=404)
        names.sort()
        names = names[-limit:]
        records = []
        for n in names:
            try:
                payload = _read_blob_json(container, n)
                records.append({"blobPath": n, "summary": _summarize_payload(payload)})
            except Exception as re:
                logging.warning(f"Failed reading blob {n}: {re}")
        return func.HttpResponse(json.dumps({"records": records}), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"traffic/history failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)


@app.route(route="traffic/forecast", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_traffic_forecast(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container = _get_env("TRAFFIC_CONTAINER", "traffic")
        prefix = req.params.get("prefix", "fortmyers/")
        window = int(req.params.get("window", "5"))
        limit = int(req.params.get("limit", str(window)))
        names = _list_json_blobs(container, prefix)
        if not names:
            return func.HttpResponse("No data", status_code=404)
        names.sort()
        names = names[-limit:]
        jam_avgs = []
        delay_avgs = []
        for n in names:
            try:
                payload = _read_blob_json(container, n)
                s = _summarize_payload(payload)
                if isinstance(s.get("jamFactorAvg"), (int, float)):
                    jam_avgs.append(float(s["jamFactorAvg"]))
                if isinstance(s.get("delayRatioAvg"), (int, float)):
                    delay_avgs.append(float(s["delayRatioAvg"]))
            except Exception as re:
                logging.warning(f"Failed reading blob {n}: {re}")
        def _ma(vals: List[float], w: int) -> float:
            if not vals:
                return None
            w = max(1, min(w, len(vals)))
            return sum(vals[-w:]) / w
        forecast = {
            "jamFactorForecast": _ma(jam_avgs, window),
            "delayRatioForecast": _ma(delay_avgs, window),
            "window": window,
            "samples": len(jam_avgs),
        }
        return func.HttpResponse(json.dumps(forecast), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"traffic/forecast failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)


@app.route(route="traffic/summary", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_traffic_summary(req: func.HttpRequest) -> func.HttpResponse:
    return api_traffic_latest(req)


# -------------------- ADT data API (ADT-backed) -------------------- #
@app.route(route="traffic/adt/latest", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_adt_latest(req: func.HttpRequest) -> func.HttpResponse:
    try:
        client = get_adt_client()
        query = "SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')"
        twins = list(client.query_twins(query))
        # summarize
        count = len(twins)
        jams = []
        delays = []
        for t in twins:
            jf = t.get("jamFactor")
            dr = t.get("delayRatio")
            if isinstance(jf, (int, float)):
                jams.append(float(jf))
            if isinstance(dr, (int, float)):
                delays.append(float(dr))
        summary = {
            "modelId": "dtmi:fgcu:traffic:RoadSegment;2",
            "twinCount": count,
            "jamFactorAvg": (sum(jams) / len(jams)) if jams else None,
            "delayRatioAvg": (sum(delays) / len(delays)) if delays else None,
        }
        return func.HttpResponse(json.dumps(summary), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"adt/latest failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)


@app.route(route="traffic/adt/history", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_adt_history(req: func.HttpRequest) -> func.HttpResponse:
    try:
        client = get_adt_client()
        min_jf = req.params.get("minJamFactor")
        min_dr = req.params.get("minDelayRatio")
        query = "SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')"
        twins = list(client.query_twins(query))
        items = []
        for t in twins:
            props = t
            jf = props.get("jamFactor")
            dr = props.get("delayRatio")
            include = True
            if min_jf is not None:
                try:
                    include = include and (isinstance(jf, (int, float)) and jf >= float(min_jf))
                except Exception:
                    pass
            if min_dr is not None:
                try:
                    include = include and (isinstance(dr, (int, float)) and dr >= float(min_dr))
                except Exception:
                    pass
            if include:
                items.append({"twinId": t.get("$dtId"), "properties": props})
        return func.HttpResponse(json.dumps({"count": len(items), "items": items}), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"adt/history failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)


@app.route(route="traffic/adt/prediction", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_adt_prediction(req: func.HttpRequest) -> func.HttpResponse:
    try:
        client = get_adt_client()
        query = "SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')"
        twins = list(client.query_twins(query))
        pjf = []
        pdr = []
        jams = []
        delays = []
        for t in twins:
            props = t
            if isinstance(props.get("predictedJamFactor"), (int, float)):
                pjf.append(float(props["predictedJamFactor"]))
            if isinstance(props.get("predictedDelayRatio"), (int, float)):
                pdr.append(float(props["predictedDelayRatio"]))
            if isinstance(props.get("jamFactor"), (int, float)):
                jams.append(float(props["jamFactor"]))
            if isinstance(props.get("delayRatio"), (int, float)):
                delays.append(float(props["delayRatio"]))
        def mean(vals: List[float]):
            return (sum(vals) / len(vals)) if vals else None
        resp = {
            "modelId": "dtmi:fgcu:traffic:RoadSegment;2",
            "fromADT": {
                "predictedJamFactorAvg": mean(pjf),
                "predictedDelayRatioAvg": mean(pdr),
                "sources": len(pjf) + len(pdr),
            },
            "fallback": {
                "jamFactorAvg": mean(jams),
                "delayRatioAvg": mean(delays),
            },
        }
        return func.HttpResponse(json.dumps(resp), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"adt/prediction failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)
