import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Tuple

import requests
import azure.functions as func


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

    # Generate base sample points near FGCU
    base_points = generate_fgcu_points()
    logging.info(f"Generated {len(base_points)} FGCU anchor-offset points")

    results = []
    failures = 0
    seen_snapped = set()  # avoid duplicate segments for the same snapped coordinate

    for (la, lo) in base_points:
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

 