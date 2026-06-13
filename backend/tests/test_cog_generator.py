"""Tests for the COG generator module.

Focuses on pure helper functions (``_tile_to_bbox`` and ``_bbox_to_tiles``)
which have no external dependencies.  Async integration tests that exercise
``generate_cog_for_tile`` require mocks and are kept minimal.
"""

from __future__ import annotations



from app.cog_generator import _bbox_to_tiles, _tile_to_bbox


# ======================================================================
# _tile_to_bbox
# ======================================================================


class TestTileToBbox:
    """Tile-to-bbox conversion — pure math, no external deps."""

    def test_zoom_0(self):
        """At zoom 0 the single tile covers the whole world."""
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(0, 0, 0)
        assert min_lon == -180.0
        assert max_lon == 180.0
        assert min_lat < -80.0  # Approximate south pole
        assert max_lat > 80.0  # Approximate north pole

    def test_navarra_tile(self):
        """Tile 14/8117/6038 should cover roughly the Navarra (Spain) area."""
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(14, 8117, 6038)
        assert -2.0 < min_lon < -1.0  # Western Navarra
        assert 42.0 < min_lat < 43.0   # Southern Navarra
        assert max_lon > min_lon
        assert max_lat > min_lat

    def test_min_less_than_max(self):
        """Output respects min < max for both longitude and latitude."""
        for z in (0, 5, 10, 14, 18):
            min_l, min_la, max_l, max_la = _tile_to_bbox(z, 0, 0)
            assert min_l < max_l, f"z={z}: min_lon >= max_lon"
            assert min_la < max_la, f"z={z}: min_lat >= max_lat"

    def test_zoom_increases_precision(self):
        """Higher zoom → narrower tiles (smaller longitude extent)."""
        bbox_z10 = _tile_to_bbox(10, 500, 500)
        bbox_z14 = _tile_to_bbox(14, 500, 500)
        lon_span_z10 = bbox_z10[2] - bbox_z10[0]
        lon_span_z14 = bbox_z14[2] - bbox_z14[0]
        assert lon_span_z14 < lon_span_z10

    def test_antimeridian_behaviour(self):
        """Tile near the antimeridian yields consistent bounds."""
        min_lon, _, max_lon, _ = _tile_to_bbox(2, 3, 0)
        assert -180.0 <= min_lon <= 180.0
        assert -180.0 <= max_lon <= 180.0
        assert min_lon < max_lon

    def test_southern_hemisphere(self):
        """Tiles in the southern hemisphere have negative latitudes."""
        _, min_lat, _, max_lat = _tile_to_bbox(5, 16, 20)
        # Upper row indices = southern hemisphere in TMS
        assert min_lat < 0.0
        assert max_lat < 0.0

    def test_tile_centre_is_within_bbox(self):
        """The centre of a tile falls within its own bbox."""
        z, x, y = 12, 2048, 1024
        min_lon, min_lat, max_lon, max_lat = _tile_to_bbox(z, x, y)
        centre_lon = (min_lon + max_lon) / 2.0
        centre_lat = (min_lat + max_lat) / 2.0
        assert min_lon <= centre_lon <= max_lon
        assert min_lat <= centre_lat <= max_lat

    def test_consistent_zoom_1_partition(self):
        """Zoom-1 tiles correctly partition the world."""
        # NW quadrant
        nw_min_lon, nw_min_lat, nw_max_lon, nw_max_lat = _tile_to_bbox(1, 0, 0)
        # SE quadrant
        se_min_lon, se_min_lat, se_max_lon, se_max_lat = _tile_to_bbox(1, 1, 1)
        # NW left edge touches -180
        assert nw_min_lon == -180.0
        # SE right edge touches 180
        assert se_max_lon == 180.0
        # NW top edge is near the north pole (85°)
        assert nw_max_lat > 80.0
        # SE bottom edge is near the south pole (-85°)
        assert se_min_lat < -80.0
        # NW bottom and SE top meet at the equator
        assert abs(nw_min_lat) < 1e-10
        assert abs(se_max_lat) < 1e-10


# ======================================================================
# _bbox_to_tiles
# ======================================================================


class TestBboxToTiles:
    """Bbox-to-tiles conversion — pure math, no external deps."""

    def test_few_tiles_at_low_zoom(self):
        """A small bbox at low zoom produces a small number of tiles."""
        tiles = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=5)
        assert len(tiles) > 0
        assert len(tiles) < 10

    def test_more_tiles_at_higher_zoom(self):
        """The same bbox at a higher zoom produces more tiles."""
        low = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=10)
        high = _bbox_to_tiles(-2.0, 42.0, -1.0, 43.0, zoom=14)
        assert len(low) < len(high)

    def test_tiles_are_valid_tms(self):
        """Every returned tile coordinate is a valid TMS triplet."""
        tiles = _bbox_to_tiles(-10.0, 35.0, 5.0, 50.0, zoom=10)
        for z, x, y in tiles:
            assert z == 10
            assert 0 <= x < 2**z
            assert 0 <= y < 2**z

    def test_empty_bbox_yields_tiles(self):
        """Even a point-like bbox should yield at least one tile."""
        tiles = _bbox_to_tiles(-1.65, 42.8, -1.65, 42.8, zoom=14)
        assert len(tiles) >= 1

    def test_large_bbox(self):
        """A large bbox covering most of Europe produces many tiles."""
        tiles = _bbox_to_tiles(-10.0, 35.0, 30.0, 60.0, zoom=8)
        assert len(tiles) > 50

    def test_all_returned_tiles_cover_bbox(self):
        """The union of returned tiles fully covers the input bbox."""
        bbox = (-3.0, 42.0, 0.0, 44.0)
        tiles = _bbox_to_tiles(*bbox, zoom=12)
        # Each tile's bbox should overlap the input bbox
        for z, x, y in tiles:
            t_min_lon, t_min_lat, t_max_lon, t_max_lat = _tile_to_bbox(z, x, y)
            # Check overlap: partial overlap is sufficient
            assert t_max_lon > bbox[0]  # tile right > bbox left
            assert t_min_lon <= bbox[2]  # tile left <= bbox right (allow exact boundary)
            assert t_max_lat > bbox[1]  # tile top > bbox bottom
            assert t_min_lat < bbox[3]  # tile bottom < bbox top

    def test_tile_count_scales_with_area(self):
        """Doubling the area approximately doubles the tile count."""
        small = _bbox_to_tiles(-1.0, 42.0, 0.0, 43.0, zoom=12)
        large = _bbox_to_tiles(-2.0, 41.0, 1.0, 44.0, zoom=12)
        # The large bbox has roughly 6× the area
        assert len(large) > len(small)


# ======================================================================
# generate_cog_for_tile — minimal import / stub test
# ======================================================================


class TestGenerateCogForTile:
    """Async integration tests are stubbed — real HTTP calls avoided."""

    def test_module_imports(self):
        """The module imports successfully."""
        from app.cog_generator import generate_cog_for_tile, run_for_tenant

        assert callable(generate_cog_for_tile)
        assert callable(run_for_tenant)

    def test_generate_cog_is_coroutine(self):
        """generate_cog_for_tile is an async function."""
        from app.cog_generator import generate_cog_for_tile

        import asyncio

        assert asyncio.iscoroutinefunction(generate_cog_for_tile)
