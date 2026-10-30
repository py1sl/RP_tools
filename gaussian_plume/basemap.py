"""Optional georeferenced basemaps and geographic plume plotting.

The plume calculations remain in local metres. Map coordinates use a local
tangent-plane approximation around the release location, suitable for
visualisation over relatively small areas.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Protocol, Sequence
from urllib.request import Request, urlopen

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
WEB_MERCATOR_MAX_LAT = 85.05112878
GeoBounds = tuple[float, float, float, float]


@dataclass
class BasemapImage:
    """A north-up raster with geographic bounds in longitude/latitude."""

    pixels: np.ndarray
    bounds: GeoBounds
    attribution: str
    source: str
    imagery_date: str | None = None

    def __post_init__(self) -> None:
        self.pixels = np.asarray(self.pixels)
        validate_bounds(self.bounds)
        if self.pixels.ndim not in (2, 3) or not all(self.pixels.shape[:2]):
            raise ValueError("pixels must be a non-empty 2D or 3D image array.")
        if not self.attribution or not self.source:
            raise ValueError("Basemap attribution and source must be provided.")


class BasemapProvider(Protocol):
    """Interface implemented by basemap image providers."""

    cache_id: str

    def fetch_image(
        self, bounds: GeoBounds, width: int, height: int
    ) -> BasemapImage:
        """Fetch a north-up image covering bounds at the requested size."""


def validate_bounds(bounds: Sequence[float]) -> GeoBounds:
    """Validate and normalize geographic bounds (west, south, east, north)."""
    if len(bounds) != 4:
        raise ValueError("bounds must be (west, south, east, north).")
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("bounds must contain only finite coordinates.")
    if not (-180 <= west < east <= 180):
        raise ValueError("bounds longitudes must satisfy -180 <= west < east <= 180.")
    if not (-90 <= south < north <= 90):
        raise ValueError("bounds latitudes must satisfy -90 <= south < north <= 90.")
    return west, south, east, north


class XYZTileProvider:
    """Retrieve standard XYZ raster tiles from a provider URL template.

    The URL template must contain ``{z}``, ``{x}``, and ``{y}`` placeholders.
    Providers may require API keys, attribution, or restrict caching; callers
    must configure and use the service in accordance with its terms.
    """

    def __init__(
        self,
        url_template: str,
        attribution: str,
        source: str = "XYZ tiles",
        user_agent: str = "RP_tools/0.1 (basemap visualization)",
        timeout: float = 20.0,
    ) -> None:
        if not all(marker in url_template for marker in ("{z}", "{x}", "{y}")):
            raise ValueError("url_template must include {z}, {x}, and {y}.")
        if not attribution.strip() or not source.strip():
            raise ValueError("attribution and source must not be empty.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self.url_template = url_template
        self.attribution = attribution
        self.source = source
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache_id = hashlib.sha256(url_template.encode()).hexdigest()

    def fetch_image(
        self, bounds: GeoBounds, width: int, height: int
    ) -> BasemapImage:
        from matplotlib.image import imread

        west, south, east, north = validate_bounds(bounds)
        _validate_size(width, height)
        if south < -WEB_MERCATOR_MAX_LAT or north > WEB_MERCATOR_MAX_LAT:
            raise ValueError("XYZ Web Mercator tiles support latitudes up to ±85.0511°.")

        x0, y0 = _mercator_pixel(west, north, zoom=0)
        x1, y1 = _mercator_pixel(east, south, zoom=0)
        zoom_zero_m_per_pixel = 2 * math.pi * EARTH_RADIUS_M / 256
        target_m_per_pixel = max(
            (x1 - x0) * zoom_zero_m_per_pixel / width,
            (y1 - y0) * zoom_zero_m_per_pixel / height,
        )
        zoom = min(22, max(0, math.ceil(math.log2(
            zoom_zero_m_per_pixel / target_m_per_pixel
        ))))
        px0, py0 = _mercator_pixel(west, north, zoom)
        px1, py1 = _mercator_pixel(east, south, zoom)
        tile_count = 1 << zoom
        tile_x0 = max(0, int(px0 // 256))
        tile_y0 = max(0, int(py0 // 256))
        tile_x1 = min(tile_count - 1, int(math.ceil(px1 / 256) - 1))
        tile_y1 = min(tile_count - 1, int(math.ceil(py1 / 256) - 1))

        first_tile = self._fetch_tile(zoom, tile_x0, tile_y0, imread)
        tile_height, tile_width = first_tile.shape[:2]
        if tile_height != 256 or tile_width != 256:
            raise ValueError("XYZ provider returned a tile that is not 256×256 pixels.")
        channels = first_tile.shape[2:] if first_tile.ndim == 3 else ()
        mosaic = np.empty(
            ((tile_y1 - tile_y0 + 1) * 256, (tile_x1 - tile_x0 + 1) * 256) + channels,
            dtype=first_tile.dtype,
        )
        for tile_y in range(tile_y0, tile_y1 + 1):
            for tile_x in range(tile_x0, tile_x1 + 1):
                tile = first_tile if (tile_x, tile_y) == (tile_x0, tile_y0) else (
                    self._fetch_tile(zoom, tile_x, tile_y, imread)
                )
                if tile.shape != first_tile.shape:
                    raise ValueError("XYZ provider returned tiles with inconsistent dimensions.")
                row = (tile_y - tile_y0) * 256
                col = (tile_x - tile_x0) * 256
                mosaic[row:row + 256, col:col + 256] = tile

        crop_x0 = max(0, int(math.floor(px0)) - tile_x0 * 256)
        crop_y0 = max(0, int(math.floor(py0)) - tile_y0 * 256)
        crop_x1 = min(mosaic.shape[1], int(math.ceil(px1)) - tile_x0 * 256)
        crop_y1 = min(mosaic.shape[0], int(math.ceil(py1)) - tile_y0 * 256)
        crop = mosaic[crop_y0:crop_y1, crop_x0:crop_x1]
        pixels = _resize_nearest(crop, width, height)
        return BasemapImage(
            pixels=pixels,
            bounds=(west, south, east, north),
            attribution=self.attribution,
            source=self.source,
        )

    def _fetch_tile(self, zoom: int, x: int, y: int, imread) -> np.ndarray:
        url = self.url_template.format(z=zoom, x=x, y=y)
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            return np.asarray(imread(io.BytesIO(response.read())))


def _mercator_pixel(longitude: float, latitude: float, zoom: int) -> tuple[float, float]:
    size = 256 * (1 << zoom)
    x = (longitude + 180.0) / 360.0 * size
    lat_rad = math.radians(latitude)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * size
    return x, y


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    rows = np.minimum((np.arange(height) * image.shape[0] / height).astype(int),
                      image.shape[0] - 1)
    cols = np.minimum((np.arange(width) * image.shape[1] / width).astype(int),
                      image.shape[1] - 1)
    return image[rows[:, None], cols[None, :]]


def _validate_size(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")


class CachedBasemap:
    """Cache-first wrapper around a basemap provider.

    Matching cached images must cover the requested bounds and have at least
    the requested pixel density. The cache records provider, bounds, size,
    attribution, and imagery date. Use ``offline=True`` to prohibit downloads.
    """

    def __init__(
        self,
        provider: BasemapProvider,
        cache_dir: str | Path,
        offline: bool = False,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.offline = offline

    def get_image(
        self, bounds: GeoBounds, width: int, height: int
    ) -> BasemapImage:
        requested = validate_bounds(bounds)
        _validate_size(width, height)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for metadata_path in self.cache_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                cached_bounds = validate_bounds(metadata["bounds"])
                if (metadata["provider"] != self.provider.cache_id
                        or not _covers(cached_bounds, requested)
                        or metadata["width"] / (cached_bounds[2] - cached_bounds[0])
                        < width / (requested[2] - requested[0])
                        or metadata["height"] / (cached_bounds[3] - cached_bounds[1])
                        < height / (requested[3] - requested[1])):
                    continue
                pixels = np.load(metadata_path.with_suffix(".npz"), allow_pickle=False)["pixels"]
                return BasemapImage(
                    pixels, cached_bounds, metadata["attribution"],
                    metadata["source"], metadata.get("imagery_date"),
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

        if self.offline:
            raise FileNotFoundError("No matching basemap is cached and offline mode is enabled.")
        image = self.provider.fetch_image(requested, width, height)
        if not _covers(image.bounds, requested):
            raise ValueError("Basemap provider returned an image that does not cover the request.")
        if image.pixels.shape[1] < width or image.pixels.shape[0] < height:
            raise ValueError("Basemap provider returned an image below the requested resolution.")
        token = hashlib.sha256(json.dumps(
            [self.provider.cache_id, image.bounds, image.pixels.shape[1],
             image.pixels.shape[0], image.imagery_date], sort_keys=True,
        ).encode()).hexdigest()
        data_path = self.cache_dir / f"{token}.npz"
        metadata_path = self.cache_dir / f"{token}.json"
        np.savez_compressed(data_path, pixels=image.pixels)
        metadata_path.write_text(json.dumps({
            "provider": self.provider.cache_id,
            "bounds": image.bounds,
            "width": image.pixels.shape[1],
            "height": image.pixels.shape[0],
            "source": image.source,
            "attribution": image.attribution,
            "imagery_date": image.imagery_date,
        }), encoding="utf-8")
        return image


def _covers(outer: GeoBounds, inner: GeoBounds) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def plot_plume_on_map(
    plume,
    source_longitude: float,
    source_latitude: float,
    downwind_extent_m: float,
    crosswind_extent_m: float,
    wind_direction_deg: float,
    basemap: BasemapImage | CachedBasemap,
    *,
    resolution_m: float = 100.0,
    nuclide: str | None = None,
    ax=None,
    log_scale: bool = True,
    **kwargs,
):
    """Plot ground-level concentration over a georeferenced map image.

    ``wind_direction_deg`` is the direction the wind blows *toward*, in
    degrees clockwise from north (0° north, 90° east). Local distances are
    converted to longitude/latitude with a spherical tangent-plane
    approximation; use this for visualisation, not precision geodesy.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = (source_longitude, source_latitude, downwind_extent_m,
              crosswind_extent_m, wind_direction_deg, resolution_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinates, extents, direction, and resolution must be finite.")
    if not (-180 <= source_longitude <= 180 and -90 < source_latitude < 90):
        raise ValueError("source coordinates must be valid longitude and latitude.")
    if downwind_extent_m <= 0 or crosswind_extent_m <= 0 or resolution_m <= 0:
        raise ValueError("extents and resolution must be positive.")
    if abs(source_latitude) >= WEB_MERCATOR_MAX_LAT:
        raise ValueError("source latitude is outside the supported Web Mercator range.")

    x_edges = np.arange(0.0, downwind_extent_m, resolution_m)
    y_edges = np.arange(-crosswind_extent_m, crosswind_extent_m, resolution_m)
    x_edges = np.append(x_edges, downwind_extent_m)
    y_edges = np.append(y_edges, crosswind_extent_m)
    if len(x_edges) < 2:
        x_edges = np.array([0.0, downwind_extent_m])
    if len(y_edges) < 2:
        y_edges = np.array([-crosswind_extent_m, crosswind_extent_m])

    bearing = math.radians(wind_direction_deg % 360.0)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges, indexing="ij")
    east_m = x_mesh * math.sin(bearing) + y_mesh * math.cos(bearing)
    north_m = x_mesh * math.cos(bearing) - y_mesh * math.sin(bearing)
    latitude = source_latitude + np.degrees(north_m / EARTH_RADIUS_M)
    longitude = source_longitude + np.degrees(
        east_m / (EARTH_RADIUS_M * math.cos(math.radians(source_latitude)))
    )
    if np.any(np.abs(latitude) > WEB_MERCATOR_MAX_LAT) or np.any(np.abs(longitude) > 180):
        raise ValueError("plume extent crosses the supported Web Mercator/world bounds.")
    bounds = (
        float(longitude.min()), float(latitude.min()),
        float(longitude.max()), float(latitude.max()),
    )
    image = basemap.get_image(
        bounds, max(256, len(x_edges) * 2), max(256, len(y_edges) * 2)
    ) if isinstance(basemap, CachedBasemap) else basemap
    if not _covers(image.bounds, bounds):
        raise ValueError("Basemap image does not cover the plume extent.")

    if ax is None:
        _, ax = plt.subplots()
    west, south, east, north = image.bounds
    ax.imshow(image.pixels, extent=(west, east, south, north), origin="upper")

    nuclide = plume._resolve_nuclide(nuclide)
    concentrations = plume.concentration_on_grid(x_edges, y_edges)[nuclide]
    plot_kwargs = dict(kwargs)
    if log_scale:
        concentrations = np.where(concentrations > 0, concentrations, np.nan)
        if "norm" not in plot_kwargs:
            finite = concentrations[np.isfinite(concentrations)]
            if finite.size:
                plot_kwargs["norm"] = LogNorm(vmin=finite.min(), vmax=finite.max())
    mesh = ax.pcolormesh(
        longitude.T, latitude.T, concentrations.T,
        shading="flat", **plot_kwargs,
    )
    plt.colorbar(mesh, ax=ax, label=f"{nuclide} concentration (Bq/m³)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"{nuclide} ground-level concentration — wind toward "
        f"{wind_direction_deg % 360:g}°"
    )
    ax.set_aspect(1 / math.cos(math.radians(source_latitude)))
    ax.text(
        0.01, 0.01, f"{image.source} — {image.attribution}",
        transform=ax.transAxes, fontsize="x-small", color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
    )
    return ax
