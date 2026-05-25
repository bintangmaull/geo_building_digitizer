"""
road_processor.py
Pipeline digitasi Jalan & Infrastruktur berbasis Road Fingerprint + Mahalanobis Distance.

Pipeline (Fase 5–11):
  Fase 5  — Scoring         : Mahalanobis distance → probability map per piksel
  Fase 6  — Masking         : Threshold + exclude vegetasi
  Fase 7  — Morphology      : Close gap, hapus blob kecil & terlalu bulat
  Fase 8  — Skeletonization : Distance transform + medial axis → centerline
  Fase 9  — Vectorize       : Skeleton → graph → prune → LineString halus
  Fase 10 — Adaptive Buffer : Buffer adaptif lebar per segmen → Polygon
  Fase 11 — Post-process    : Simplify, area filter

Output: 2 SHP
  - *_jalan_polygon.shp   : Polygon lebar jalan (dari adaptive buffer)
  - *_jalan_centerline.shp: LineString garis tengah jalan (dari centerline)

CATATAN: File ini berdiri sendiri dan TIDAK mengimpor dari postprocess.py
         (kode bangunan tidak disentuh).
"""

import os
import gc
import json
import threading
import warnings
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Konstanta internal (tidak diekspos ke UI)
# ──────────────────────────────────────────────────────────────────────────────
_MAHAL_THRESHOLD     = 3.0   # sigma — piksel dengan dist > threshold dianggap bukan jalan
_MIN_BLOB_AREA_PX    = 200   # pixel² — blob terlalu kecil dihapus
_MAX_COMPACTNESS     = 0.65  # Polsby-Popper — blob terlalu bulat dihapus (bukan jalan)
_MORPH_CLOSE_RADIUS  = 5     # pixel — radius morphological closing untuk tutup gap
_MIN_BRANCH_LEN_PX   = 20    # pixel — dead-end pendek di-prune
_SMOOTH_WINDOW       = 5     # node — window spline smoothing centerline


class RoadProcessor:
    """
    Processor mandiri untuk digitasi Jalan & Infrastruktur.
    Menggunakan Road Fingerprint (road_fingerprint.json) sebagai acuan
    karakteristik visual jalan.

    Cara pemakaian:
        processor = RoadProcessor(fingerprint_path="road_fingerprint.json", ...)
        polygon_gdf, centerline_gdf = processor.process_raster(target_path, ...)
    """

    OBJECT_CLASS  = "Jalan"
    OBJECT_COLOR  = "#FF6B35"

    def __init__(
        self,
        fingerprint_path: str,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Args:
            fingerprint_path: Path ke road_fingerprint.json hasil Fase 1–4
        """
        self.fingerprint_path = fingerprint_path
        self.log_callback     = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event    = threading.Event()
        self._fingerprint     = None   # Loaded on demand

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[Jalan] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Load Fingerprint
    # ──────────────────────────────────────────────────────────────────────────

    def load_fingerprint(self):
        """Muat road_fingerprint.json ke memori."""
        if not os.path.isfile(self.fingerprint_path):
            raise FileNotFoundError(
                f"Road Fingerprint tidak ditemukan: {self.fingerprint_path}\n"
                f"Silakan buat fingerprint terlebih dahulu menggunakan 'Buat Fingerprint'."
            )
        with open(self.fingerprint_path, "r") as f:
            self._fingerprint = json.load(f)
        self._log(f"Fingerprint dimuat: {Path(self.fingerprint_path).name}")
        self._log(f"  Sumber referensi: {self._fingerprint.get('source_raster', '?')}")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5: SCORING (Mahalanobis Distance → Probability Map)
    # ══════════════════════════════════════════════════════════════════════════

    def _score_tile(self, tile_img_norm: np.ndarray) -> np.ndarray:
        """
        Hitung road probability map [0–1] untuk satu tile.

        Args:
            tile_img_norm: (H, W, C) citra tile setelah normalisasi percentile stretch

        Returns:
            prob_map: (H, W) float32, nilai 1.0 = sangat mirip profil jalan
        """
        from core.objects.road_fingerprint import RoadFingerprintBuilder

        fp  = self._fingerprint["fingerprint"]
        loc = np.array(fp["mahalanobis"]["location"], dtype=np.float64)
        VI  = np.array(fp["mahalanobis"]["precision_matrix"], dtype=np.float64)

        # Ekstrak fitur per piksel
        features, _ = RoadFingerprintBuilder.extract_features(tile_img_norm)
        H, W, F = features.shape
        flat = features.reshape(-1, F).astype(np.float64)

        # Mahalanobis distance — per piksel (batch)
        diff = flat - loc[np.newaxis, :]           # (N, F)
        dist = np.sqrt(np.maximum(
            np.einsum("ni,ij,nj->n", diff, VI, diff), 0
        ))                                          # (N,) — Mahalanobis distance

        # Konversi ke probability [0–1] via exp(-0.5 * d²)
        # Makin kecil distance → makin besar probability
        prob = np.exp(-0.5 * (dist / _MAHAL_THRESHOLD) ** 2)
        return prob.reshape(H, W).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE A & C: ROAD SPACE & MASKING
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_road_space(
        self,
        building_mask: np.ndarray,
        pixel_size_m: float,
        dilate_m: float = 6.0,
    ) -> np.ndarray:
        """
        [FASE A] Dilate polygon bangunan → gap antar bangunan = kandidat area jalan.
        """
        import cv2
        if building_mask is None:
            return None
            
        dilate_px = max(1, int(dilate_m / pixel_size_m))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        
        building_uint8 = building_mask.astype(np.uint8)
        dilated = cv2.dilate(building_uint8, kernel).astype(bool)
        
        # Gap = area yang diliputi dilasi tapi BUKAN bangunan asli
        road_space = dilated & ~building_mask.astype(bool)
        return road_space

    def _apply_mask(
        self,
        prob_map: np.ndarray,
        img_norm: np.ndarray,
        threshold: float = 0.45,
        building_mask: Optional[np.ndarray] = None,
        road_space: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Buat binary road mask dari probability map.
        [FIX 1] Subtract building mask lebih awal (jika ada).
        [FASE C] Gabungkan probabilitas jalan dengan road_space, dikurangi vegetasi.
        """
        if building_mask is not None:
            prob_map[building_mask == 1] = 0.0

        # Threshold probability
        prob_binary = (prob_map >= threshold)

        # Exclude vegetasi (ExG tinggi)
        R = img_norm[:, :, 0].astype(float)
        G = img_norm[:, :, 1].astype(float)
        B = img_norm[:, :, 2].astype(float) if img_norm.shape[2] >= 3 else np.zeros_like(R)
        exg = 2.0 * G - R - B
        veg_mask = (exg > 15)

        # [FASE C] Combine
        # Jika pixel memiliki skor Mahalanobis tinggi ATAU berada di gap antar bangunan
        if road_space is not None:
            binary = (prob_binary | road_space) & ~veg_mask
        else:
            binary = prob_binary & ~veg_mask

        return binary.astype(np.uint8) * 255

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 7: MORPHOLOGY
    # ══════════════════════════════════════════════════════════════════════════

    def _morphology(
        self,
        binary_mask: np.ndarray,
        close_radius: int = _MORPH_CLOSE_RADIUS,
        min_area_px: int = _MIN_BLOB_AREA_PX,
        min_elongation: float = 2.5,
        max_solidity: float = 0.85,
    ) -> np.ndarray:
        """
        Morphological cleaning:
          1. Morphological closing (tutup gap kecil antar piksel jalan)
          2. Hapus blob terlalu kecil (noise)
          [FIX 2] 3. Hapus blob dengan Solidity > max_solidity (kemungkinan bangunan)
          [FIX 2] 4. Hapus blob dengan Elongation < min_elongation (kemungkinan bangunan)
        """
        import cv2
        import math

        # 1. Morphological Closing
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 2, 3, 4. Filter per blob
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        cleaned = np.zeros_like(closed)
        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area_px:
                continue  # Terlalu kecil → hapus

            # Ambil kontur untuk menghitung fitur bentuk
            blob_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
                
            cnt = contours[0]
            
            # [FIX 2] Solidity = area / convex_hull_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity > max_solidity:
                    continue  # Terlalu solid/kotak → hapus
                    
            # [FIX 2] Elongation = major_axis / minor_axis
            if len(cnt) >= 5:  # fitEllipse butuh minimal 5 titik
                (x, y), (minor, major), angle = cv2.fitEllipse(cnt)
                if minor > 0:
                    elongation = major / minor
                    if elongation < min_elongation:
                        continue  # Terlalu membulat/persegi → hapus

            cleaned[labels == label] = 255

        return cleaned

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 8: SKELETONIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _skeletonize(
        self,
        binary_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hasilkan:
          - skeleton    : (H, W) bool — centerline 1-pixel width
          - dist_transform: (H, W) float32 — lebar lokal / 2 per piksel

        Returns:
            (skeleton, dist_transform)
        """
        try:
            from skimage.morphology import medial_axis
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise ImportError(
                "scikit-image & scipy diperlukan. "
                "Install: pip install scikit-image scipy"
            )

        mask_bool = binary_mask > 0

        # Distance transform — nilai = jarak tiap piksel ke tepi mask
        # Artinya: lebar jalan lokal ≈ 2 × dist_transform pada centerline
        dist_transform = distance_transform_edt(mask_bool).astype(np.float32)

        # Medial axis — thinning ke centerline 1-pixel
        skeleton, _ = medial_axis(mask_bool, return_distance=False), None
        # Alternatif: pakai morphological skeleton jika medial_axis lambat
        # skeleton = morphology.skeletonize(mask_bool)

        return skeleton.astype(np.uint8), dist_transform

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 9: VECTORIZE CENTERLINE
    # ══════════════════════════════════════════════════════════════════════════

    def _skeleton_to_linestrings(
        self,
        skeleton: np.ndarray,
        dist_transform: np.ndarray,
        transform,
        min_branch_len: int = _MIN_BRANCH_LEN_PX,
        smooth_window: int = _SMOOTH_WINDOW,
        pixel_size_m: float = 0.1,
    ) -> list:
        """
        Konversi skeleton pixel → list LineString (koordinat geo).

        Steps:
          1. Skeleton pixel → graph (8-connectivity)
          2. Prune dead-end pendek (< min_branch_len)
          [FIX 3] 3. Gap Filling (menyambung dead-end yang terputus < 20m)
          4. Traversal graph → segmen koordinat pixel
          5. Pixel → koordinat geo via rasterio transform
          6. Smooth LineString dengan moving average
        """
        import networkx as nx
        import math
        from itertools import combinations
        from shapely.geometry import LineString

        H, W = skeleton.shape
        skel_pts = np.argwhere(skeleton > 0)  # (row, col)
        if len(skel_pts) == 0:
            return [], []

        # Build graph
        G = nx.Graph()
        skel_set = set(map(tuple, skel_pts))

        for (r, c) in skel_pts:
            G.add_node((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in skel_set:
                        dist = 1.414 if (dr != 0 and dc != 0) else 1.0
                        G.add_edge((r, c), nb, weight=dist)

        # Prune dead-ends pendek
        changed = True
        while changed:
            changed = False
            leaves = [n for n in G.nodes() if G.degree(n) == 1]
            for leaf in leaves:
                # Telusuri cabang dari leaf sampai non-leaf atau panjang >= min_branch_len
                path = [leaf]
                cur = leaf
                prev = None
                while True:
                    nbs = [n for n in G.neighbors(cur) if n != prev]
                    if len(nbs) != 1:
                        break  # Bukan dead-end lagi
                    prev = cur
                    cur  = nbs[0]
                    path.append(cur)
                    if len(path) >= min_branch_len:
                        break
                if len(path) < min_branch_len and G.degree(path[-1]) != 1:
                    # Hapus semua node di cabang kecuali junction
                    for node in path[:-1]:
                        G.remove_node(node)
                    changed = True

        if G.number_of_nodes() == 0:
            return [], []

        # [FIX 3] Connectivity-Aware Gap Filling
        # Sambungkan dead-end berdekatan (jarak < 20 meter & saling berhadapan)
        gap_fill_radius_m = 20.0
        gap_fill_radius_px = gap_fill_radius_m / pixel_size_m if pixel_size_m > 0 else 50
        leaves = [n for n in G.nodes() if G.degree(n) == 1]
        
        def get_leaf_vector(leaf):
            nbs = list(G.neighbors(leaf))
            if not nbs: return (0, 0)
            nb = nbs[0]
            return (leaf[0] - nb[0], leaf[1] - nb[1])
            
        for u, v in combinations(leaves, 2):
            if G.has_edge(u, v): continue
            
            dist = math.hypot(u[0] - v[0], u[1] - v[1])
            if dist > gap_fill_radius_px:
                continue
                
            # Hanya sambungkan jika mereka tidak terhubung
            if nx.has_path(G, u, v):
                if nx.shortest_path_length(G, u, v, weight=None) < dist * 2:
                    continue
                    
            vec_u = get_leaf_vector(u)
            vec_v = get_leaf_vector(v)
            mag_u = math.hypot(vec_u[0], vec_u[1]) + 1e-6
            mag_v = math.hypot(vec_v[0], vec_v[1]) + 1e-6
            nu = (vec_u[0]/mag_u, vec_u[1]/mag_u)
            nv = (vec_v[0]/mag_v, vec_v[1]/mag_v)
            
            vec_uv = (v[0] - u[0], v[1] - u[1])
            mag_uv = math.hypot(vec_uv[0], vec_uv[1]) + 1e-6
            n_uv = (vec_uv[0]/mag_uv, vec_uv[1]/mag_uv)
            
            dot_u = nu[0]*n_uv[0] + nu[1]*n_uv[1]
            dot_v = nv[0]*(-n_uv[0]) + nv[1]*(-n_uv[1])
            
            if dot_u > 0.5 and dot_v > 0.5:
                G.add_edge(u, v, weight=dist)

        # Traversal edge-paths dari graph
        def _pixel_to_geo(r, c):
            """Pixel (row, col) → (lon, lat) atau (x, y)."""
            x, y = transform * (c + 0.5, r + 0.5)
            return (x, y)

        lines  = []
        widths = []
        visited_edges = set()

        # Mulai traversal dari node-node junction atau endpoint
        for start_node in list(G.nodes()):
            for end_node in list(G.neighbors(start_node)):
                edge_key = tuple(sorted([start_node, end_node]))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                # Kumpulkan koordinat geo
                coords = [_pixel_to_geo(*start_node), _pixel_to_geo(*end_node)]

                # Lebar: rata-rata dist_transform di node-node ini
                r1, c1 = start_node
                r2, c2 = end_node
                d1 = dist_transform[r1, c1] if 0 <= r1 < H and 0 <= c1 < W else 1.0
                d2 = dist_transform[r2, c2] if 0 <= r2 < H and 0 <= c2 < W else 1.0
                avg_width_px = (d1 + d2) / 2.0
                widths.append(avg_width_px)

                if len(coords) >= 2:
                    lines.append(LineString(coords))

        # Gabungkan segmen pendek yang satu-arah menjadi LineString lebih panjang
        lines, widths = self._merge_line_segments(G, transform, dist_transform, H, W)

        return lines, widths

    def _merge_line_segments(
        self,
        G,
        transform,
        dist_transform: np.ndarray,
        H: int,
        W: int,
    ) -> Tuple[list, list]:
        """
        Traversal DFS dari semua ujung/junction, gabungkan node menjadi
        LineString panjang (bukan edge-per-edge pendek).
        """
        from shapely.geometry import LineString

        def _pixel_to_geo(r, c):
            x, y = transform * (c + 0.5, r + 0.5)
            return (x, y)

        def _get_width(r, c):
            if 0 <= r < H and 0 <= c < W:
                return float(dist_transform[r, c])
            return 1.0

        lines  = []
        widths = []
        visited_nodes = set()

        # Start dari junction (degree != 2) atau endpoint (degree == 1)
        start_nodes = [n for n in G.nodes() if G.degree(n) != 2]
        if not start_nodes:
            start_nodes = list(G.nodes())[:1]

        for start in start_nodes:
            for nb in G.neighbors(start):
                if (start, nb) in visited_nodes or (nb, start) in visited_nodes:
                    continue

                # Traversal chain
                path  = [start, nb]
                widths_path = [_get_width(*start), _get_width(*nb)]
                visited_nodes.add((start, nb))

                cur  = nb
                prev = start
                while G.degree(cur) == 2:
                    nbs = [n for n in G.neighbors(cur) if n != prev]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    edge_key = (min(cur, nxt), max(cur, nxt))
                    if edge_key in visited_nodes:
                        break
                    visited_nodes.add(edge_key)
                    path.append(nxt)
                    widths_path.append(_get_width(*nxt))
                    prev = cur
                    cur  = nxt

                if len(path) < 2:
                    continue

                # Smooth koordinat dengan moving average sederhana
                coords = [_pixel_to_geo(r, c) for r, c in path]
                coords = self._smooth_coords(coords, window=_SMOOTH_WINDOW)

                if len(coords) >= 2:
                    lines.append(LineString(coords))
                    widths.append(float(np.mean(widths_path)))

        return lines, widths

    @staticmethod
    def _smooth_coords(coords: list, window: int = 5) -> list:
        """Moving average smoothing pada list koordinat (x, y)."""
        if len(coords) <= window:
            return coords
        coords_arr = np.array(coords, dtype=float)
        half = window // 2
        smoothed = []
        for i in range(len(coords_arr)):
            i0 = max(0, i - half)
            i1 = min(len(coords_arr), i + half + 1)
            smoothed.append(coords_arr[i0:i1].mean(axis=0))
        return [tuple(p) for p in smoothed]

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 10: ADAPTIVE BUFFER
    # ══════════════════════════════════════════════════════════════════════════

    def _adaptive_buffer(
        self,
        lines: list,
        widths_px: list,
        pixel_size_m: float,
        min_width_m: float = 3.0,
    ) -> "geopandas.GeoDataFrame":
        """
        Buffer adaptif per segmen LineString berdasarkan lebar dari distance transform.

        Args:
            lines       : list of shapely LineString (geo-coordinates)
            widths_px   : list of float — lebar rata-rata tiap segmen dalam pixel
            pixel_size_m: ukuran pixel dalam meter
            min_width_m : lebar minimum jalan (meter)

        Returns:
            GeoDataFrame dengan kolom geometry (Polygon)
        """
        import geopandas as gpd
        from shapely.ops import unary_union

        polygons = []
        for line, w_px in zip(lines, widths_px):
            # Lebar jalan = 2 × radius distance transform (karena dist = jarak ke tepi)
            width_m = max(w_px * 2.0 * pixel_size_m, min_width_m)
            buf = line.buffer(
                width_m / 2.0,
                cap_style=2,   # flat cap
                join_style=1,  # round join
            )
            if buf and not buf.is_empty:
                polygons.append(buf)

        if not polygons:
            return gpd.GeoDataFrame(geometry=[], crs=None)

        # Union semua buffer → MultiPolygon bersih
        from shapely.geometry import MultiPolygon, Polygon
        merged = unary_union(polygons)
        if isinstance(merged, Polygon):
            geom_list = [merged]
        elif isinstance(merged, MultiPolygon):
            geom_list = list(merged.geoms)
        elif hasattr(merged, "geoms"):
            geom_list = [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]
        else:
            geom_list = [merged]

        return gpd.GeoDataFrame(geometry=geom_list, crs=None)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 11: POST-PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def _postprocess_polygons(
        self,
        gdf_polygon: "geopandas.GeoDataFrame",
        min_area_m2: float = 50.0,
        simplify_tol_m: float = 0.5,
    ) -> "geopandas.GeoDataFrame":
        """
        Post-process polygon jalan:
          - Simplify geometri
          - Filter area minimum
        """
        if len(gdf_polygon) == 0:
            return gdf_polygon

        import geopandas as gpd

        # Simplify
        gdf_polygon["geometry"] = gdf_polygon.geometry.simplify(
            simplify_tol_m, preserve_topology=True
        )

        # Area filter (pakai CRS metric jika tersedia)
        if gdf_polygon.crs and gdf_polygon.crs.is_geographic:
            utm_crs  = gdf_polygon.estimate_utm_crs()
            gdf_m    = gdf_polygon.to_crs(utm_crs)
            areas_m2 = gdf_m.geometry.area
        else:
            areas_m2 = gdf_polygon.geometry.area

        mask   = areas_m2 >= min_area_m2
        result = gdf_polygon[mask].copy().reset_index(drop=True)
        result["class"] = self.OBJECT_CLASS
        return result

    def _postprocess_lines(
        self,
        gdf_lines: "geopandas.GeoDataFrame",
        min_length_m: float = 5.0,
        simplify_tol_m: float = 0.5,
    ) -> "geopandas.GeoDataFrame":
        """
        Post-process centerline jalan:
          - Simplify
          - Filter panjang minimum
        """
        if len(gdf_lines) == 0:
            return gdf_lines

        gdf_lines["geometry"] = gdf_lines.geometry.simplify(
            simplify_tol_m, preserve_topology=True
        )

        if gdf_lines.crs and gdf_lines.crs.is_geographic:
            utm_crs = gdf_lines.estimate_utm_crs()
            gdf_m   = gdf_lines.to_crs(utm_crs)
            lengths = gdf_m.geometry.length
        else:
            lengths = gdf_lines.geometry.length

        mask   = lengths >= min_length_m
        result = gdf_lines[mask].copy().reset_index(drop=True)
        result["class"] = self.OBJECT_CLASS
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE UTAMA: PROCESS RASTER
    # ══════════════════════════════════════════════════════════════════════════

    def process_raster(
        self,
        raster_path: str,
        output_dir: str,
        tile_size: int = 1024,
        overlap: int = 128,
        min_area_m2: float = 50.0,
        prob_threshold: float = 0.45,
        building_gdf: Optional["geopandas.GeoDataFrame"] = None,
    ) -> Tuple["geopandas.GeoDataFrame", "geopandas.GeoDataFrame"]:
        """
        Proses citra target secara tile-by-tile dan hasilkan polygon + centerline.

        Args:
            raster_path  : Path citra target (ECW / GeoTIFF)
            output_dir   : Folder output untuk SHP
            tile_size    : Ukuran tile dalam pixel
            overlap      : Overlap antar tile dalam pixel
            min_area_m2  : Luas minimum polygon jalan (m²)
            prob_threshold: Threshold probability map [0–1]

        Returns:
            (gdf_polygon, gdf_centerline)
            — masing-masing GeoDataFrame siap disimpan ke SHP
        """
        import rasterio
        import geopandas as gpd
        from shapely.geometry import LineString

        from core.objects.road_fingerprint import RoadFingerprintBuilder
        from core.tiling import tiles_generator

        if self._fingerprint is None:
            self.load_fingerprint()

        norm_params = self._fingerprint.get("normalization", {})
        if not norm_params:
            raise ValueError("Fingerprint tidak memiliki parameter normalisasi. "
                             "Pastikan fingerprint dibuat dengan versi terbaru.")

        self._log(f"Pipeline jalan dimulai: {Path(raster_path).name}")
        self._progress(5, "Memulai deteksi jalan ...")

        os.makedirs(output_dir, exist_ok=True)

        # Kumpulkan semua LineString dan Polygon dari tiap tile
        all_lines   = []
        all_polys   = []
        tile_crs    = None

        project_root   = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        tiles_temp_dir = os.path.join(project_root, "temp", "road_tiles")
        os.makedirs(tiles_temp_dir, exist_ok=True)

        # Hitung ukuran pixel dari raster
        with rasterio.open(raster_path) as src:
            transform_full = src.transform
            pixel_size_m   = abs(transform_full.a)
            tile_crs       = src.crs
            if tile_crs and tile_crs.is_geographic:
                # Konversi degrees → meters (approx 1° ≈ 111,320 m)
                pixel_size_m = abs(transform_full.a) * 111_320

        tile_count = 0
        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=False,
        ):
            if self.is_cancelled():
                self._log("Dibatalkan.")
                break

            pct = 10 + int(80 * idx / max(total, 1))
            self._progress(pct, f"Tile {idx+1}/{total}: {Path(tile_path).name}")
            self._log(f"[{idx+1}/{total}] Memproses tile: {Path(tile_path).stem}")

            try:
                with rasterio.open(tile_path) as src:
                    n_bands    = min(src.count, 3)
                    img_raw    = src.read(list(range(1, n_bands + 1)))
                    img_raw    = np.moveaxis(img_raw, 0, -1).astype(np.float32)
                    tile_transform = src.transform
                    tile_crs_local = src.crs

                if img_raw.shape[2] < 3:
                    self._log(f"  ⚠ Tile hanya {img_raw.shape[2]} band, dilewati.")
                    continue

                # Pastikan norm_params punya key yang sesuai jumlah band
                n_bands = img_raw.shape[2]
                tile_norm = {
                    f"band_{c}": norm_params.get(f"band_{c}", {"p_low": 0, "p_high": 255})
                    for c in range(n_bands)
                }

                # Fase 5: Normalisasi + Scoring
                img_norm = RoadFingerprintBuilder.normalize_image(img_raw, tile_norm)
                prob_map = self._score_tile(img_norm)

                # [FIX 1] Rasterize building_gdf for this tile
                building_mask_arr = None
                road_space = None
                if building_gdf is not None and len(building_gdf) > 0:
                    from rasterio.features import geometry_mask
                    shapes = [geom for geom in building_gdf.geometry if geom is not None]
                    try:
                        building_mask_arr = geometry_mask(
                            shapes,
                            transform=tile_transform,
                            invert=True,  # True inside buildings
                            out_shape=(img_raw.shape[0], img_raw.shape[1])
                        )
                        # [FASE A] Road Space Estimation
                        road_space = self._estimate_road_space(
                            building_mask_arr, pixel_size_m, dilate_m=6.0
                        )
                    except Exception as e:
                        self._log(f"  ⚠ Gagal memproses building mask: {e}")

                # [FASE C] Masking & Combine
                binary_mask = self._apply_mask(
                    prob_map, img_norm, threshold=prob_threshold, 
                    building_mask=building_mask_arr, road_space=road_space
                )

                road_px_count = (binary_mask > 0).sum()
                self._log(f"  Prob map → {road_px_count:,} piksel kandidat jalan")

                if road_px_count < _MIN_BLOB_AREA_PX:
                    self._log("  → Tidak ada jalan ditemukan di tile ini, dilewati.")
                    continue

                # Fase 7: Morphology
                cleaned_mask = self._morphology(binary_mask)
                if (cleaned_mask > 0).sum() < _MIN_BLOB_AREA_PX:
                    self._log("  → Setelah morphology, tidak ada piksel tersisa.")
                    continue

                # Fase 8: Skeletonization
                skeleton, dist_tf = self._skeletonize(cleaned_mask)

                if skeleton.sum() == 0:
                    self._log("  → Skeleton kosong, dilewati.")
                    continue

                # Fase 9: Vectorize
                lines, widths = self._skeleton_to_linestrings(
                    skeleton, dist_tf, tile_transform, pixel_size_m=pixel_size_m
                )
                self._log(f"  → {len(lines)} segmen centerline")

                if not lines:
                    continue

                # Fase 10: Adaptive Buffer
                gdf_buf = self._adaptive_buffer(lines, widths, pixel_size_m)
                if gdf_buf.crs is None and tile_crs_local:
                    gdf_buf = gdf_buf.set_crs(tile_crs_local)

                all_polys.extend(gdf_buf.geometry.tolist())
                all_lines.extend(lines)
                tile_count += 1

            except Exception as e:
                self._log(f"  ⚠ Error pada tile {Path(tile_path).stem}: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                # Bersihkan tile temp
                try:
                    if os.path.exists(tile_path):
                        os.remove(tile_path)
                except Exception:
                    pass

        self._log(f"Selesai: {tile_count} tile berhasil diproses.")

        if not all_polys and not all_lines:
            self._log("⚠ Tidak ada jalan terdeteksi di seluruh citra.")
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs=tile_crs)
            return empty_gdf, empty_gdf

        # ── Fase 11: Post-processing ──────────────────────────────────────────
        self._progress(92, "Post-processing polygon & centerline jalan ...")

        # Gabung semua polygon dari semua tile → union → split
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon, Polygon

        if all_polys:
            merged_poly = unary_union(all_polys)
            if isinstance(merged_poly, Polygon):
                poly_list = [merged_poly]
            elif isinstance(merged_poly, MultiPolygon):
                poly_list = list(merged_poly.geoms)
            else:
                poly_list = [g for g in merged_poly.geoms
                             if isinstance(g, (Polygon, MultiPolygon))]
            gdf_polygon = gpd.GeoDataFrame(geometry=poly_list, crs=tile_crs)
        else:
            gdf_polygon = gpd.GeoDataFrame(geometry=[], crs=tile_crs)

        if all_lines:
            gdf_centerline = gpd.GeoDataFrame(geometry=all_lines, crs=tile_crs)
        else:
            gdf_centerline = gpd.GeoDataFrame(geometry=[], crs=tile_crs)

        gdf_polygon    = self._postprocess_polygons(gdf_polygon, min_area_m2)
        gdf_centerline = self._postprocess_lines(gdf_centerline)

        self._log(
            f"=== SELESAI: {len(gdf_polygon)} polygon jalan, "
            f"{len(gdf_centerline)} segmen centerline ==="
        )
        self._progress(100, f"Jalan selesai: {len(gdf_polygon)} polygon")
        gc.collect()

        return gdf_polygon, gdf_centerline
