import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Tuple

import requests
import azure.functions as func
import math


app = func.FunctionApp()


def _get_env(name: str, default: str = "") -> str:
    """Get an app setting with a simple default."""
    v = os.getenv(name)
    return v if v else default


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
            results.append(
                {
                    "lat": snap_lat,
                    "lon": snap_lon,
                    "original": [la, lo],
                    "url": url,
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
    }

    outblob.set(json.dumps(payload))
    logging.info(
        f"Wrote blob with {len(results)} unique snapped items (failures={failures})"
    )


# Optional HTTP trigger to check configuration / liveness
@app.route(route="traffic/fortmyers", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def http_collect(req: func.HttpRequest) -> func.HttpResponse:
    key_present = bool(_get_env("AZURE_MAPS_SUBSCRIPTION_KEY"))
    return func.HttpResponse(
        json.dumps(
            {
                "status": "ready",
                "mapsKeyConfigured": key_present,
                "trafficTimerCron": "0 */5 * * * *",
            }
        ),
        mimetype="application/json",
        status_code=200,
    )

 