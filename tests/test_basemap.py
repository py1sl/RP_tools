"""Tests for optional geographic basemap plotting and caching."""

import io

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import pytest

import gaussian_plume.basemap as basemap_module
from gaussian_plume.basemap import (
    BasemapImage,
    CachedBasemap,
    plot_plume_on_map,
    validate_bounds,
)
from gaussian_plume.plume import GaussianPlume

matplotlib.use("Agg")


class FakeProvider:
    cache_id = "fake-style-v1"

    def __init__(self):
        self.calls = []

    def fetch_image(self, bounds, width, height):
        self.calls.append((bounds, width, height))
        return BasemapImage(
            pixels=np.zeros((height, width, 3), dtype=np.uint8),
            bounds=bounds,
            attribution="Test attribution",
            source="Test basemap",
            imagery_date="2025-01",
        )


def test_validate_bounds_rejects_invalid_order():
    with pytest.raises(ValueError, match="longitudes"):
        validate_bounds((10, -1, 0, 1))


def test_cache_fetches_once_then_serves_matching_image(tmp_path):
    provider = FakeProvider()
    cache = CachedBasemap(provider, tmp_path)
    bounds = (-1.0, 50.0, 0.0, 51.0)

    first = cache.get_image(bounds, 32, 24)
    cached = CachedBasemap(provider, tmp_path, offline=True).get_image(bounds, 32, 24)

    assert len(provider.calls) == 1
    assert np.array_equal(first.pixels, cached.pixels)
    assert cached.imagery_date == "2025-01"
    assert list(tmp_path.glob("*.json"))


def test_offline_cache_miss_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline mode"):
        CachedBasemap(FakeProvider(), tmp_path, offline=True).get_image(
            (-1, 50, 0, 51), 32, 24
        )


def test_cache_does_not_reuse_image_below_requested_resolution(tmp_path):
    provider = FakeProvider()
    cache = CachedBasemap(provider, tmp_path)
    bounds = (-1.0, 50.0, 0.0, 51.0)
    cache.get_image(bounds, 32, 24)

    with pytest.raises(FileNotFoundError):
        CachedBasemap(provider, tmp_path, offline=True).get_image(bounds, 64, 48)


def test_xyz_provider_fetches_tiles_and_returns_georeferenced_image(monkeypatch):
    png = io.BytesIO()
    imsave(png, np.full((256, 256, 3), 0.5))
    requested_urls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return png.getvalue()

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr(basemap_module, "urlopen", fake_urlopen)
    provider = basemap_module.XYZTileProvider(
        "https://tiles.example/{z}/{x}/{y}.png", "Provider attribution"
    )
    bounds = (-0.01, -0.01, 0.01, 0.01)
    image = provider.fetch_image(bounds, 64, 48)

    assert image.pixels.shape[:2] == (48, 64)
    assert image.pixels.shape[2] in (3, 4)
    assert image.bounds == bounds
    assert requested_urls
    assert all(url.startswith("https://tiles.example/") for url in requested_urls)


def test_plot_orients_plume_toward_wind_direction():
    plume = GaussianPlume(
        release={"Cs137": 1.0},
        wind_speed=2.0,
        stability_category="D",
        release_height=0.0,
    )
    image = BasemapImage(
        pixels=np.zeros((300, 300, 3), dtype=np.uint8),
        bounds=(-0.02, -0.02, 0.02, 0.02),
        attribution="Test",
        source="Test",
    )
    ax = plot_plume_on_map(
        plume, 0.0, 0.0, 1000, 200, 90, image, resolution_m=100
    )

    coordinates = ax.collections[0].get_coordinates()
    assert coordinates[0, -1, 0] > coordinates[0, 0, 0]
    assert "toward 90°" in ax.get_title()
    plt.close(ax.figure)


def test_plot_rejects_basemap_that_does_not_cover_plume():
    plume = GaussianPlume(
        release={"Cs137": 1.0},
        wind_speed=2.0,
        stability_category="D",
        release_height=0.0,
    )
    image = BasemapImage(
        pixels=np.zeros((10, 10, 3), dtype=np.uint8),
        bounds=(-0.001, -0.001, 0.001, 0.001),
        attribution="Test",
        source="Test",
    )

    with pytest.raises(ValueError, match="does not cover"):
        plot_plume_on_map(plume, 0.0, 0.0, 1000, 200, 0, image)
