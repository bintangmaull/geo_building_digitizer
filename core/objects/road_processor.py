"""
road_processor.py
Pipeline digitasi Jalan & Infrastruktur berbasis Road Fingerprint + Mahalanobis Distance.

Pipeline (Fase 5–11):
  Fase 5  — Scoring         : Mahalanobis distance → probability map per piksel
  Fase 6  — Masking         : Threshold + exclude vegetasi + exclude bangunan
  Fase 7  — Morphology      : Close gap, hapus blob kecil, compactness & elongation filter
  Fase 7b — Proximity Filter: Hanya blob dekat bangunan yang dipertahankan
  Fase 8  — Skeletonization : Distance transform + medial axis → centerline
  Fase 9  — Vectorize       : Skeleton → graph → prune → LineString halus
  Fase 10 — Adaptive Buffer : Buffer adaptif lebar per segmen → Polygon
  Fase 11 — Post-process    : Simplify, area filter

Output: 2 SHP
  - *_jalan_polygon.shp   : Polygon lebar jalan (dari adaptive buffer)
  - *_jalan_centerline.shp: LineString garis tengah jalan (dari centerline)
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
# Konstanta internal
# ──────────────────────────────────────────────────────────────────────────────
_MAHAL_THRESHOLD     = 3.0   # sigma
_MIN_BLOB_AREA_PX    = 200   # pixel²
_MAX_COMPACTNESS     = 0.50  # Dikembalikan ke 0.50 agar segmen jalan pendek tidak terhapus
_MORPH_CLOSE_RADIUS  = 5     # pixel
_MIN_BRANCH_LEN_PX   = 20    # pixel
_SMOOTH_WINDOW       = 5     # node
_MAX_DIST_TO_BUILDING_M = 30.0  # [FIX 2] blob lebih jauh dari ini dihapus


class RoadProcessor:
    """
    Processor mandiri untuk digitasi Jalan & Infrastruktur.
    """

    OBJECT_CLASS = "Jalan"
    OBJECT_COLOR = "#FF6B35"

    def __init__(
        self,
        fingerprint_path: str,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.fingerprint_path  = fingerprint_path
        self.log_callback      = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event     = threading.Event()
        self._fingerprint      = None

    def cancel(self):       self._cancel_event.set()
    def is_cancelled(self): return self._cancel_event.is_set()
    def reset_cancel(self): self._cancel_event.clear()

    def _log(self, msg: str):           self.log_callback(f"[Jalan] {msg}")
    def _progress(self, pct, msg):      self.progress_callback(pct, msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Load Fingerprint
    # ──────────────────────────────────────────────────────────────────────────

    def load_fingerprint(self):
        if not os.path.isfile(self.fingerprint_path):
            raise FileNotFoundError(
                f"File tidak ditemukan: {self.fingerprint_path}\n"
                "Silakan buat fingerprint (.json) atau sediakan model YOLO (.pt)."
            )
            
        if self.fingerprint_path.endswith('.pt'):
            self.mode = 'yolo'
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self.fingerprint_path)
                self._log(f"Model UAV-YOLOv12 dimuat: {Path(self.fingerprint_path).name}")
            except Exception as e:
                self._log(f"Gagal memuat YOLO: {e}")
                raise
        else:
            self.mode = 'mahalanobis'
            with open(self.fingerprint_path) as f:
                self._fingerprint = json.load(f)
            self._log(f"Fingerprint dimuat: {Path(self.fingerprint_path).name}")
            self._log(f"  Sumber referensi: {self._fingerprint.get('source_raster', '?')}")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5: SCORING
    # ══════════════════════════════════════════════════════════════════════════

    def _score_tile(self, tile_img_norm: np.ndarray, img_raw: np.ndarray = None) -> np.ndarray:
        """Mahalanobis distance atau YOLO-seg → probability map [0–1]."""
        if getattr(self, 'mode', 'mahalanobis') == 'yolo' and img_raw is not None:
            import cv2
            # Gunakan model UAV-YOLOv12 untuk memprediksi masker jalan
            H, W = img_raw.shape[:2]
            # Konversi float32 RGB (dari rasterio) menjadi uint8 BGR (standar OpenCV/YOLO)
            img_bgr = cv2.cvtColor(np.clip(img_raw, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            # Kembalikan conf ke 0.25 agar tidak mendeteksi noise sembarangan
            results = self._yolo.predict(img_bgr, verbose=False, conf=0.25)
            prob_map = np.zeros((H, W), dtype=np.float32)
            
            if len(results) > 0 and results[0].masks is not None:
                import cv2
                masks = results[0].masks.data.cpu().numpy()  # [N, H_out, W_out]
                
                # YOLOv8/11/12 kadang meresize output mask, pastikan ukurannya sama
                if masks.shape[1:] != (H, W):
                    masks_resized = []
                    for m in masks:
                        masks_resized.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR))
                    masks = np.array(masks_resized)
                
                # Gabungkan semua masker objek jalan di tile ini
                if len(masks) > 0:
                    prob_map = np.max(masks, axis=0)
            return prob_map
            
        # ── Jalur Klasik: Mahalanobis Distance ──
        from core.objects.road_fingerprint import RoadFingerprintBuilder

        fp  = self._fingerprint["fingerprint"]
        loc = np.array(fp["mahalanobis"]["location"],        dtype=np.float64)
        VI  = np.array(fp["mahalanobis"]["precision_matrix"], dtype=np.float64)

        features, _ = RoadFingerprintBuilder.extract_features(tile_img_norm)
        H, W, F = features.shape
        flat = features.reshape(-1, F).astype(np.float64)

        diff = flat - loc[np.newaxis, :]
        dist = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", diff, VI, diff), 0))
        prob = np.exp(-0.5 * (dist / _MAHAL_THRESHOLD) ** 2)
        return prob.reshape(H, W).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 6: MASKING
    # [FIX 1] road_space TIDAK di-OR ke prob_binary
    #         → hanya dipakai sebagai context di proximity filter
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_mask(
        self,
        prob_map: np.ndarray,
        img_norm: np.ndarray,
        threshold: float = 0.45,
        building_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Binary road mask dari probability map saja.
        - Exclude bangunan sebelum threshold  [FIX 1]
        - Exclude vegetasi (ExG > 15)
        - TIDAK menggunakan road_space sebagai OR — mencegah blob raksasa
        """
        # [FIX 1] Blacklist area bangunan sebelum apapun
        if building_mask is not None:
            prob_map = prob_map.copy()
            prob_map[building_mask.astype(bool)] = 0.0

        prob_binary = prob_map >= threshold

        R = img_norm[:, :, 0].astype(float)
        G = img_norm[:, :, 1].astype(float)
        B = img_norm[:, :, 2].astype(float) if img_norm.shape[2] >= 3 else np.zeros_like(R)
        exg = 2.0 * G - R - B
        veg_mask = exg > 15.0

        if getattr(self, 'mode', 'mahalanobis') == 'yolo':
            # YOLO mendeteksi jalan di bawah pohon (jika dilatih). Jangan hapus secara paksa.
            # Kita gunakan veg_mask nanti sebagai syarat "Smart Bridge".
            binary = prob_binary
        else:
            binary = prob_binary & ~veg_mask
            
        return binary.astype(np.uint8) * 255

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 7: MORPHOLOGY
    # [FIX 4] Compactness threshold diperketat: 0.5 → 0.25
    # ══════════════════════════════════════════════════════════════════════════

    def _morphology(
        self,
        binary_mask: np.ndarray,
        close_radius: int  = _MORPH_CLOSE_RADIUS,
        min_area_px: int   = _MIN_BLOB_AREA_PX,
        min_elongation: float = 2.5,
        max_solidity: float   = 0.85,
        veg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        1. Morphological closing
        2. Hapus blob terlalu kecil
        3. [FIX 4] Compactness filter lebih ketat (< 0.25)
        4. Elongation & solidity filter
        """
        import cv2, math

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # [SMART BRIDGE] Hanya pertahankan jembatan jika area tersebut adalah pohon
        if veg_mask is not None:
            added = (closed > 0) & (binary_mask == 0)
            n_labels, labels = cv2.connectedComponents(added.astype(np.uint8))
            valid_closed = binary_mask.copy()
            for lbl in range(1, n_labels):
                bridge_mask = (labels == lbl)
                bridge_area = bridge_mask.sum()
                if bridge_area == 0: continue
                # Jika lebih dari 15% area jembatan adalah pohon, anggap valid
                overlap = (bridge_mask & veg_mask).sum() / bridge_area
                if overlap > 0.15:
                    valid_closed[bridge_mask] = 255
            closed = valid_closed

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        cleaned = np.zeros_like(closed)

        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area_px:
                continue

            blob_mask = (labels == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # [FIX 4] Compactness diperketat
            compactness = (4 * math.pi * area) / (perimeter ** 2)
            if compactness > _MAX_COMPACTNESS:   # 0.25 — jalan sangat memanjang
                continue

            # Solidity
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity  = area / hull_area if hull_area > 0 else 1.0

            # Elongation
            elongation = 1.0
            if len(cnt) >= 5:
                (_, _), (minor, major), _ = cv2.fitEllipse(cnt)
                if minor > 0:
                    elongation = major / minor

            # Lolos jika memanjang ATAU berliku/bercabang
            if elongation < min_elongation and solidity >= 0.75:
                continue

            cleaned[labels == lbl] = 255

        return cleaned

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 7b: PROXIMITY FILTER (NEW)
    # [FIX 2] Hapus blob yang terlalu jauh dari bangunan
    # ══════════════════════════════════════════════════════════════════════════

    def _filter_by_proximity_to_buildings(
        self,
        cleaned_mask: np.ndarray,
        building_mask: Optional[np.ndarray],
        pixel_size_m: float,
        max_dist_m: float = _MAX_DIST_TO_BUILDING_M,
    ) -> np.ndarray:
        """
        [FIX 2] Hanya pertahankan blob road yang dalam jarak max_dist_m dari bangunan.

        Jalan desa selalu berada di dekat bangunan — blob yang jauh di tengah
        sawah/kebun pasti false positive.

        Jika building_mask tidak tersedia, kembalikan mask tidak diubah.
        """
        import cv2
        from scipy.ndimage import distance_transform_edt

        if building_mask is None or not building_mask.any():
            self._log("  ⚠ Building mask kosong — proximity filter dilewati")
            return cleaned_mask

        # Jarak tiap piksel ke bangunan terdekat (dalam meter)
        dist_to_bldg_px = distance_transform_edt(~building_mask.astype(bool))
        dist_to_bldg_m  = dist_to_bldg_px * pixel_size_m

        # Mask piksel yang cukup dekat bangunan
        near_building = dist_to_bldg_m <= max_dist_m

        # Per blob: pertahankan jika ADA pikselnya yang dekat bangunan
        # (bukan semua piksel harus dekat — ini untuk jalan panjang)
        n_labels, labels_map = cv2.connectedComponents(
            (cleaned_mask > 0).astype(np.uint8), connectivity=8
        )
        result = np.zeros_like(cleaned_mask)

        kept, dropped = 0, 0
        for lbl in range(1, n_labels):
            blob_px   = labels_map == lbl
            # Blob dianggap "dekat bangunan" jika minimal 15% pikselnya dalam jarak max_dist_m
            overlap_ratio = (blob_px & near_building).sum() / (blob_px.sum() + 1e-6)
            if overlap_ratio >= 0.15:
                result[blob_px] = 255
                kept += 1
            else:
                dropped += 1

        self._log(f"  Proximity filter: {kept} blob dipertahankan, {dropped} dihapus")
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 8: SKELETONIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _skeletonize(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from skimage.morphology import medial_axis
        from scipy.ndimage import distance_transform_edt

        mask_bool    = binary_mask > 0
        dist_transform = distance_transform_edt(mask_bool).astype(np.float32)
        skeleton, _  = medial_axis(mask_bool, return_distance=False), None
        return skeleton.astype(np.uint8), dist_transform

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 9: VECTORIZE CENTERLINE
    # [FIX 3] Gap filling threshold diperketat: 0.5 → 0.70
    # ══════════════════════════════════════════════════════════════════════════

    def _skeleton_to_linestrings(
        self,
        skeleton: np.ndarray,
        dist_transform: np.ndarray,
        transform,
        min_branch_len: int = _MIN_BRANCH_LEN_PX,
        smooth_window: int  = _SMOOTH_WINDOW,
        pixel_size_m: float = 0.1,
    ) -> Tuple[list, list]:
        import networkx as nx
        import math
        from itertools import combinations
        from shapely.geometry import LineString

        H, W     = skeleton.shape
        skel_pts = np.argwhere(skeleton > 0)
        if len(skel_pts) == 0:
            return [], []

        # Build graph
        G       = nx.Graph()
        skel_set = set(map(tuple, skel_pts))
        for (r, c) in skel_pts:
            G.add_node((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in skel_set:
                        G.add_edge((r, c), nb, weight=1.414 if dr and dc else 1.0)

        # Prune dead-ends pendek
        changed = True
        while changed:
            changed = False
            for leaf in [n for n in G.nodes() if G.degree(n) == 1]:
                path = [leaf]
                cur, prev = leaf, None
                while True:
                    nbs = [n for n in G.neighbors(cur) if n != prev]
                    if len(nbs) != 1:
                        break
                    prev, cur = cur, nbs[0]
                    path.append(cur)
                    if len(path) >= min_branch_len:
                        break
                if len(path) < min_branch_len and G.degree(path[-1]) != 1:
                    for node in path[:-1]:
                        G.remove_node(node)
                    changed = True

        if G.number_of_nodes() == 0:
            return [], []

        # [FIX 3] Gap filling — threshold diperketat 0.5 → 0.70
        # Juga tambah syarat: jarak gap maksimal 50% dari radius
        gap_fill_radius_px = 20.0 / pixel_size_m if pixel_size_m > 0 else 50
        leaves = [n for n in G.nodes() if G.degree(n) == 1]

        def _leaf_vector(leaf):
            nbs = list(G.neighbors(leaf))
            if not nbs:
                return (0, 0)
            nb = nbs[0]
            return (leaf[0] - nb[0], leaf[1] - nb[1])

        for u, v in combinations(leaves, 2):
            if G.has_edge(u, v):
                continue
            dist_uv = math.hypot(u[0] - v[0], u[1] - v[1])
            # [FIX 3a] Jarak maksimal dikembalikan penuh agar gap jauh bisa tersambung
            if dist_uv > gap_fill_radius_px:
                continue
            if nx.has_path(G, u, v):
                if nx.shortest_path_length(G, u, v, weight=None) < dist_uv * 2:
                    continue

            vec_u = _leaf_vector(u)
            vec_v = _leaf_vector(v)
            mag_u = math.hypot(*vec_u) + 1e-6
            mag_v = math.hypot(*vec_v) + 1e-6
            nu    = (vec_u[0] / mag_u, vec_u[1] / mag_u)
            nv    = (vec_v[0] / mag_v, vec_v[1] / mag_v)

            vec_uv = (v[0] - u[0], v[1] - u[1])
            mag_uv = math.hypot(*vec_uv) + 1e-6
            n_uv   = (vec_uv[0] / mag_uv, vec_uv[1] / mag_uv)

            dot_u = nu[0] * n_uv[0] + nu[1] * n_uv[1]
            dot_v = nv[0] * (-n_uv[0]) + nv[1] * (-n_uv[1])

            # [FIX 3b] Threshold dilonggarkan kembali ke 0.50 agar jalan berliku bisa nyambung
            if dot_u > 0.50 and dot_v > 0.50:
                G.add_edge(u, v, weight=dist_uv)

        return self._merge_line_segments(G, transform, dist_transform, H, W, pixel_size_m)

    def _merge_line_segments(
        self,
        G,
        transform,
        dist_transform: np.ndarray,
        H: int,
        W: int,
        pixel_size_m: float = 0.1,
    ) -> Tuple[list, list]:
        from shapely.geometry import LineString

        def _pixel_to_geo(r, c):
            x, y = transform * (c + 0.5, r + 0.5)
            return (x, y)

        def _get_width(r, c):
            if 0 <= r < H and 0 <= c < W:
                return float(dist_transform[r, c])
            return 1.0

        lines, widths = [], []
        visited_nodes = set()

        start_nodes = [n for n in G.nodes() if G.degree(n) != 2] or list(G.nodes())[:1]

        for start in start_nodes:
            for nb in G.neighbors(start):
                if (start, nb) in visited_nodes or (nb, start) in visited_nodes:
                    continue

                path         = [start, nb]
                widths_path  = [_get_width(*start), _get_width(*nb)]
                visited_nodes.add((start, nb))

                cur, prev = nb, start
                while G.degree(cur) == 2:
                    nbs = [n for n in G.neighbors(cur) if n != prev]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    ek  = (min(cur, nxt), max(cur, nxt))
                    if ek in visited_nodes:
                        break
                    visited_nodes.add(ek)
                    path.append(nxt)
                    widths_path.append(_get_width(*nxt))
                    prev, cur = cur, nxt

                if len(path) < 2:
                    continue

                # Prune cabang buntu pendek < 15m
                path_length_m = len(path) * pixel_size_m
                is_dead_end   = G.degree(path[0]) == 1 or G.degree(path[-1]) == 1
                if is_dead_end and path_length_m < 15.0:
                    continue

                coords = [_pixel_to_geo(r, c) for r, c in path]
                coords = self._smooth_coords(coords, window=_SMOOTH_WINDOW)

                if len(coords) >= 2:
                    lines.append(LineString(coords))
                    widths.append(float(np.mean(widths_path)))

        return lines, widths

    @staticmethod
    def _smooth_coords(coords: list, window: int = 5) -> list:
        if len(coords) <= window:
            return coords
        arr     = np.array(coords, dtype=float)
        half    = window // 2
        smoothed = []
        for i in range(len(arr)):
            i0 = max(0, i - half)
            i1 = min(len(arr), i + half + 1)
            smoothed.append(arr[i0:i1].mean(axis=0))
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
        max_width_m: float = 5.0,
    ) -> "geopandas.GeoDataFrame":
        import geopandas as gpd
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon, Polygon

        polygons = []
        for line, w_px in zip(lines, widths_px):
            # [AESTHETIK MAPFLOW 1] Simplify centerline sebelum di-buffer agar lurus dan smooth, tidak bergerigi
            smooth_line = line.simplify(2.0, preserve_topology=True)
            
            # [AESTHETIK MAPFLOW 2] Gunakan lebar konstan untuk semua jalan (misal 4.0 meter).
            # Mapflow selalu menampilkan jalan dengan lebar yang seragam, tidak menggelembung/mengecil
            width_m = 4.0
            
            buf     = smooth_line.buffer(width_m / 2.0, cap_style=2, join_style=1)
            if buf and not buf.is_empty:
                polygons.append(buf)

        if not polygons:
            return gpd.GeoDataFrame(geometry=[], crs=None)

        merged = unary_union(polygons)
        if isinstance(merged, Polygon):
            geom_list = [merged]
        elif isinstance(merged, MultiPolygon):
            geom_list = list(merged.geoms)
        else:
            geom_list = [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]

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
        if len(gdf_polygon) == 0:
            return gdf_polygon
        import geopandas as gpd

        gdf_polygon["geometry"] = gdf_polygon.geometry.simplify(
            simplify_tol_m, preserve_topology=True
        )
        if gdf_polygon.crs and gdf_polygon.crs.is_geographic:
            utm_crs = gdf_polygon.estimate_utm_crs()
            areas_m2 = gdf_polygon.to_crs(utm_crs).geometry.area
        else:
            utm_crs = gdf_polygon.crs
            areas_m2 = gdf_polygon.geometry.area

        result = gdf_polygon[areas_m2 >= min_area_m2].copy().reset_index(drop=True)
        
        # [FIX] Sambungkan poligon yang terputus (Geometric Closing) khusus untuk YOLO
        if getattr(self, 'mode', 'mahalanobis') == 'yolo' and len(result) > 0:
            try:
                from shapely.geometry import MultiPolygon, Polygon
                from shapely.ops import unary_union
                
                # Transform ke UTM untuk satuan meter
                is_geo = result.crs and result.crs.is_geographic
                if is_geo:
                    result = result.to_crs(utm_crs)
                
                # Jembatani celah hingga 20 meter (Buffer 10m keluar, lalu 10m ke dalam)
                closed_geom = result.geometry.buffer(10.0).buffer(-10.0)
                merged_geom = unary_union(closed_geom)
                
                # Pisahkan kembali jika menjadi MultiPolygon
                if isinstance(merged_geom, MultiPolygon):
                    poly_list = list(merged_geom.geoms)
                elif isinstance(merged_geom, Polygon):
                    poly_list = [merged_geom]
                else:
                    poly_list = [g for g in merged_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                    
                result = gpd.GeoDataFrame(geometry=poly_list, crs=utm_crs)
                
                # Kembalikan ke CRS awal
                if is_geo:
                    result = result.to_crs(gdf_polygon.crs)
            except Exception as e:
                self._log(f"  ⚠ Geometric closing gagal: {e}")

        result["class"] = self.OBJECT_CLASS
        return result

    def _postprocess_lines(
        self,
        gdf_lines: "geopandas.GeoDataFrame",
        min_length_m: float = 5.0,
        simplify_tol_m: float = 0.5,
    ) -> "geopandas.GeoDataFrame":
        if len(gdf_lines) == 0:
            return gdf_lines

        gdf_lines["geometry"] = gdf_lines.geometry.simplify(
            simplify_tol_m, preserve_topology=True
        )
        if gdf_lines.crs and gdf_lines.crs.is_geographic:
            lengths = gdf_lines.to_crs(gdf_lines.estimate_utm_crs()).geometry.length
        else:
            lengths = gdf_lines.geometry.length

        result = gdf_lines[lengths >= min_length_m].copy().reset_index(drop=True)

        # [FIX] Sambungkan garis centerline yang terputus (Snapping) khusus untuk YOLO
        if getattr(self, 'mode', 'mahalanobis') == 'yolo' and len(result) > 1:
            try:
                from shapely.geometry import LineString, MultiLineString
                from shapely.ops import linemerge
                import itertools, math
                
                is_geo = result.crs and result.crs.is_geographic
                utm_crs = result.estimate_utm_crs() if is_geo else result.crs
                result_utm = result.to_crs(utm_crs) if is_geo else result
                
                lines = []
                for geom in result_utm.geometry:
                    if isinstance(geom, MultiLineString):
                        lines.extend(list(geom.geoms))
                    elif isinstance(geom, LineString):
                        lines.append(geom)
                        
                new_lines = []
                endpoints = []
                for i, line in enumerate(lines):
                    if len(line.coords) >= 2:
                        endpoints.append({'id': i, 'pt': line.coords[0], 'type': 'start'})
                        endpoints.append({'id': i, 'pt': line.coords[-1], 'type': 'end'})
                
                def dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                
                connections = []
                for e1, e2 in itertools.combinations(endpoints, 2):
                    if e1['id'] == e2['id']: continue
                    d = dist(e1['pt'], e2['pt'])
                    if d <= 25.0:  # Radius sambungan 25 meter
                        connections.append((d, e1, e2))
                
                connections.sort(key=lambda x: x[0])
                connected = set()
                
                for d, e1, e2 in connections:
                    k1 = (e1['id'], e1['type'])
                    k2 = (e2['id'], e2['type'])
                    if k1 not in connected and k2 not in connected:
                        connected.add(k1)
                        connected.add(k2)
                        new_lines.append(LineString([e1['pt'], e2['pt']]))
                
                if new_lines:
                    merged = linemerge(lines + new_lines)
                    if isinstance(merged, MultiLineString):
                        final_lines = list(merged.geoms)
                    elif isinstance(merged, LineString):
                        final_lines = [merged]
                    else:
                        final_lines = [g for g in merged.geoms if isinstance(g, (LineString, MultiLineString))]
                        
                    result = gpd.GeoDataFrame(geometry=final_lines, crs=utm_crs)
                    if is_geo:
                        result = result.to_crs(gdf_lines.crs)
            except Exception as e:
                self._log(f"  ⚠ Centerline snapping gagal: {e}")

        result["class"] = self.OBJECT_CLASS
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE UTAMA
    # ══════════════════════════════════════════════════════════════════════════

    def process_raster(
        self,
        raster_path: str,
        output_dir: str,
        tile_size: int = 1024,
        overlap: int   = 128,
        min_area_m2: float    = 50.0,
        prob_threshold: float = 0.45,
        building_gdf: Optional["geopandas.GeoDataFrame"] = None,
    ) -> Tuple["geopandas.GeoDataFrame", "geopandas.GeoDataFrame"]:
        import rasterio
        import geopandas as gpd
        from shapely.geometry import LineString
        from core.objects.road_fingerprint import RoadFingerprintBuilder
        from core.tiling import tiles_generator

        if getattr(self, 'mode', 'mahalanobis') == 'mahalanobis':
            if self._fingerprint is None:
                self.load_fingerprint()
            norm_params = self._fingerprint.get("normalization", {})
            if not norm_params:
                raise ValueError("Fingerprint tidak memiliki parameter normalisasi.")
        else:
            if not hasattr(self, '_yolo') or self._yolo is None:
                self.load_fingerprint()
            norm_params = {}

        self._log(f"Pipeline jalan dimulai: {Path(raster_path).name}")
        self._progress(5, "Memulai deteksi jalan ...")
        os.makedirs(output_dir, exist_ok=True)

        all_lines, all_polys = [], []
        tile_crs = None

        project_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tiles_temp_dir = os.path.join(project_root, "temp", "road_tiles")
        os.makedirs(tiles_temp_dir, exist_ok=True)

        with rasterio.open(raster_path) as src:
            transform_full = src.transform
            pixel_size_m   = abs(transform_full.a)
            tile_crs       = src.crs
            if tile_crs and tile_crs.is_geographic:
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
                    n_bands        = min(src.count, 3)
                    img_raw        = src.read(list(range(1, n_bands + 1)))
                    img_raw        = np.moveaxis(img_raw, 0, -1).astype(np.float32)
                    tile_transform = src.transform
                    tile_crs_local = src.crs

                if img_raw.shape[2] < 3:
                    self._log(f"  ⚠ Hanya {img_raw.shape[2]} band, dilewati.")
                    continue

                if getattr(self, 'mode', 'mahalanobis') == 'mahalanobis':
                    tile_norm = {
                        f"band_{c}": norm_params.get(f"band_{c}", {"p_low": 0, "p_high": 255})
                        for c in range(img_raw.shape[2])
                    }
                    img_norm = RoadFingerprintBuilder.normalize_image(img_raw, tile_norm)
                else:
                    img_norm = img_raw  # YOLO tidak butuh normalisasi khusus
                    
                # ── Fase 5: Scoring (Mahalanobis atau YOLO) ──────────────────
                prob_map = self._score_tile(img_norm, img_raw=img_raw)

                # ── Rasterize building_gdf untuk tile ini ─────────────────────
                building_mask_arr = None
                if building_gdf is not None and len(building_gdf) > 0:
                    from rasterio.features import geometry_mask
                    shapes = [g for g in building_gdf.geometry if g is not None]
                    try:
                        building_mask_arr = geometry_mask(
                            shapes,
                            transform=tile_transform,
                            invert=True,
                            out_shape=(img_raw.shape[0], img_raw.shape[1]),
                        )
                    except Exception as e:
                        self._log(f"  ⚠ Building mask gagal: {e}")

                # ── Fase 6: Masking [FIX 1] ───────────────────────────────────
                binary_mask = self._apply_mask(
                    prob_map, img_norm,
                    threshold=prob_threshold,
                    building_mask=building_mask_arr,
                )

                road_px = (binary_mask > 0).sum()
                self._log(f"  Prob map → {road_px:,} piksel kandidat jalan")
                if road_px < _MIN_BLOB_AREA_PX:
                    self._log("  → Tidak ada jalan, dilewati.")
                    continue

                # ── Fase 7: Morphology [FIX 4] ────────────────────────────────
                # Deteksi pohon khusus untuk tile ini (sebagai syarat Smart Bridge)
                R_tile = img_norm[:, :, 0].astype(float)
                G_tile = img_norm[:, :, 1].astype(float)
                B_tile = img_norm[:, :, 2].astype(float) if img_norm.shape[2] >= 3 else np.zeros_like(R_tile)
                tile_veg_mask = (2.0 * G_tile - R_tile - B_tile) > 15.0

                # YOLO memprediksi per-segmen kotak, kadang ada celah kecil antar segmen.
                # Kita perbesar radius closing untuk menyambung celah tersebut.
                # Radius 80 piksel akan menyambung celah hingga ~160 piksel (~20 meter di dunia nyata)
                is_yolo = getattr(self, 'mode', 'mahalanobis') == 'yolo'
                close_rad = 80 if is_yolo else _MORPH_CLOSE_RADIUS
                
                # Gunakan Smart Bridge (lewat veg_mask) hanya untuk YOLO agar aman
                pass_veg = tile_veg_mask if is_yolo else None
                cleaned_mask = self._morphology(binary_mask, close_radius=close_rad, veg_mask=pass_veg)

                # ── Fase 7b: Proximity Filter [FIX 2] ────────────────────────
                cleaned_mask = self._filter_by_proximity_to_buildings(
                    cleaned_mask,
                    building_mask=building_mask_arr,
                    pixel_size_m=pixel_size_m,
                    max_dist_m=_MAX_DIST_TO_BUILDING_M,
                )

                if (cleaned_mask > 0).sum() < _MIN_BLOB_AREA_PX:
                    self._log("  → Setelah filter, tidak ada piksel tersisa.")
                    continue

                # ── Fase 8: Skeletonization ────────────────────────────────────
                skeleton, dist_tf = self._skeletonize(cleaned_mask)
                if skeleton.sum() == 0:
                    self._log("  → Skeleton kosong, dilewati.")
                    continue

                # ── Fase 9: Vectorize [FIX 3] ────────────────────────────────
                lines, widths = self._skeleton_to_linestrings(
                    skeleton, dist_tf, tile_transform, pixel_size_m=pixel_size_m
                )
                self._log(f"  → {len(lines)} segmen centerline")
                if not lines:
                    continue

                # ── Fase 10: Adaptive Buffer ──────────────────────────────────
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
                try:
                    if os.path.exists(tile_path):
                        os.remove(tile_path)
                except Exception:
                    pass

        self._log(f"Selesai: {tile_count} tile diproses.")

        if not all_polys and not all_lines:
            self._log("⚠ Tidak ada jalan terdeteksi.")
            empty = gpd.GeoDataFrame(geometry=[], crs=tile_crs)
            return empty, empty

        # ── Fase 11: Post-processing ──────────────────────────────────────────
        self._progress(92, "Post-processing polygon & centerline ...")

        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon, Polygon

        if all_polys:
            merged    = unary_union(all_polys)
            poly_list = (
                [merged] if isinstance(merged, Polygon)
                else list(merged.geoms) if isinstance(merged, MultiPolygon)
                else [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]
            )
            gdf_polygon = gpd.GeoDataFrame(geometry=poly_list, crs=tile_crs)
        else:
            gdf_polygon = gpd.GeoDataFrame(geometry=[], crs=tile_crs)

        gdf_centerline = (
            gpd.GeoDataFrame(geometry=all_lines, crs=tile_crs)
            if all_lines
            else gpd.GeoDataFrame(geometry=[], crs=tile_crs)
        )

        gdf_polygon    = self._postprocess_polygons(gdf_polygon, min_area_m2)
        gdf_centerline = self._postprocess_lines(gdf_centerline)

        self._log(
            f"=== SELESAI: {len(gdf_polygon)} polygon jalan, "
            f"{len(gdf_centerline)} segmen centerline ==="
        )
        self._progress(100, f"Jalan selesai: {len(gdf_polygon)} polygon")
        gc.collect()
        return gdf_polygon, gdf_centerline
