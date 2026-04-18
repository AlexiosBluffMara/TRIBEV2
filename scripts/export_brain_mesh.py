"""
Export the fsaverage5 pial surface (left + right hemisphere merged) to:
  webapp/public/brain.glb        — GLTF Binary (Three.js GLTFLoader)
  webapp/public/networks.bin     — Int16Array, one Yeo-7 index per vertex

Usage:
    python scripts/export_brain_mesh.py [--no-networks]

Requires: nibabel, nilearn, numpy (already in requirements.txt)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fio
import numpy as np

try:
    from nilearn import datasets as _nl_datasets
except ImportError:
    _nl_datasets = None

ROOT   = Path(__file__).parent.parent
OUT_GL  = ROOT / 'webapp' / 'public' / 'brain.glb'
OUT_NET = ROOT / 'webapp' / 'public' / 'networks.bin'

# ── Yeo-7 network name → integer index ───────────────────────────────────────
YEO7_MAP = {
    'Vis':         0,
    'SomMot':      1,
    'DorsAttn':    2,
    'SalVentAttn': 3,
    'Limbic':      4,
    'Cont':        5,
    'Default':     6,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_surface(path: str):
    """
    Read a brain surface file — supports both:
      - GIFTI (.gii, .surf.gii) — two darrays: coords + faces
      - FreeSurfer binary (.pial, .white, .inflated)
    Returns (coords float32, faces uint32).
    """
    p = str(path)
    if '.gii' in p.lower():
        img = nib.load(p)
        coords = img.darrays[0].data.astype(np.float32)
        faces  = img.darrays[1].data.astype(np.uint32)
        return coords, faces
    else:
        # FreeSurfer binary
        coords, faces = fio.read_geometry(p)
        return coords.astype(np.float32), faces.astype(np.uint32)


def load_fsaverage5_pial():
    """Return (lh_coords, lh_faces, rh_coords, rh_faces) as numpy arrays."""
    if _nl_datasets is None:
        raise RuntimeError('nilearn not installed')
    print('[mesh] Fetching fsaverage5 from nilearn cache…')
    fs5 = _nl_datasets.fetch_surf_fsaverage('fsaverage5')
    print(f'[mesh] pial_left  = {fs5["pial_left"]}')
    print(f'[mesh] pial_right = {fs5["pial_right"]}')
    lh_c, lh_f = _read_surface(fs5['pial_left'])
    rh_c, rh_f = _read_surface(fs5['pial_right'])
    return (lh_c, lh_f, rh_c, rh_f)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (n_verts × 3)."""
    normals = np.zeros_like(verts, dtype=np.float64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)            # face normals (un-normalised)
    for i in range(3):
        np.add.at(normals, faces[:, i], cross)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (normals / norms).astype(np.float32)


def build_glb(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> bytes:
    """
    Pack a GLB 2.0 file with POSITION + NORMAL attributes and uint32 indices.

    Buffer layout (one contiguous buffer):
      [POSITION: n_verts*3*4][NORMAL: n_verts*3*4][INDICES: n_faces*3*4]
    """
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    pos_bytes  = verts.tobytes()
    nor_bytes  = normals.tobytes()
    idx_bytes  = faces.astype(np.uint32).tobytes()

    # Pad each view to 4-byte boundary
    def pad4(b):
        r = len(b) % 4
        return b + b'\x00' * (4 - r) if r else b

    pos_bytes = pad4(pos_bytes)
    nor_bytes = pad4(nor_bytes)
    idx_bytes = pad4(idx_bytes)

    buf_bytes = pos_bytes + nor_bytes + idx_bytes
    buf_len   = len(buf_bytes)

    pos_off = 0
    nor_off = len(pos_bytes)
    idx_off = nor_off + len(nor_bytes)

    # Compute bounding box for POSITION accessor
    mn = verts.min(axis=0).tolist()
    mx = verts.max(axis=0).tolist()

    gltf_json = {
        'asset': {'version': '2.0', 'generator': 'JemmaBrain export_brain_mesh.py'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes':  [{'mesh': 0, 'name': 'fsaverage5_pial'}],
        'meshes': [{
            'name': 'brain',
            'primitives': [{
                'attributes': {'POSITION': 0, 'NORMAL': 1},
                'indices':    2,
                'mode':       4,   # TRIANGLES
            }],
        }],
        'accessors': [
            {   # 0 — POSITION
                'bufferView':    0,
                'componentType': 5126,   # FLOAT
                'count':         n_verts,
                'type':          'VEC3',
                'min':           mn,
                'max':           mx,
            },
            {   # 1 — NORMAL
                'bufferView':    1,
                'componentType': 5126,
                'count':         n_verts,
                'type':          'VEC3',
            },
            {   # 2 — INDICES
                'bufferView':    2,
                'componentType': 5125,   # UNSIGNED_INT
                'count':         n_faces * 3,
                'type':          'SCALAR',
            },
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': pos_off, 'byteLength': len(pos_bytes), 'target': 34962},  # ARRAY_BUFFER
            {'buffer': 0, 'byteOffset': nor_off, 'byteLength': len(nor_bytes), 'target': 34962},
            {'buffer': 0, 'byteOffset': idx_off, 'byteLength': len(idx_bytes), 'target': 34963},  # ELEMENT_ARRAY_BUFFER
        ],
        'buffers': [{'byteLength': buf_len}],
    }

    json_str   = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    # Pad JSON chunk to 4-byte boundary with spaces
    r = len(json_bytes) % 4
    if r:
        json_bytes += b' ' * (4 - r)

    # GLB header + chunks
    MAGIC   = 0x46546C67   # "glTF"
    VERSION = 2
    JSON_CHUNK = 0x4E4F534A  # "JSON"
    BIN_CHUNK  = 0x004E4942  # "BIN\0"

    json_chunk  = struct.pack('<II', len(json_bytes), JSON_CHUNK) + json_bytes
    bin_chunk   = struct.pack('<II', buf_len, BIN_CHUNK)  + buf_bytes
    total_len   = 12 + len(json_chunk) + len(bin_chunk)
    header      = struct.pack('<III', MAGIC, VERSION, total_len)

    return header + json_chunk + bin_chunk


# ── Yeo-7 network labels ──────────────────────────────────────────────────────

def fetch_yeo7_labels(fs5_info: dict, n_lh: int, n_rh: int) -> np.ndarray:
    """
    Try to load Yeo-7 vertex labels from the fsaverage5 dataset.
    Returns Int16Array of length n_lh + n_rh, values 0-6 or -1 = unknown.
    """
    labels = np.full(n_lh + n_rh, -1, dtype=np.int16)

    # Strategy 1: look for .annot files next to the pial files
    for hemi, offset, n in [('lh', 0, n_lh), ('rh', n_lh, n_rh)]:
        pial_path = Path(fs5_info.get(f'pial_{hemi[:-1]}left' if hemi == 'lh' else 'pial_right', ''))
        surf_dir  = pial_path.parent if pial_path.exists() else None

        candidates = []
        if surf_dir:
            candidates = list(surf_dir.glob(f'{hemi}.Yeo2011_7Networks*.annot'))
            if not candidates:
                candidates = list(surf_dir.parent.glob(f'**/{hemi}.Yeo2011_7Networks*.annot'))

        if candidates:
            annot_path = candidates[0]
            print(f'[networks] reading Yeo-7 annot: {annot_path}')
            try:
                vtx_labels, ctab, names = fio.read_annot(str(annot_path))
                # Map FreeSurfer Yeo-7 label indices to our 0-6 scheme
                names_decoded = [n.decode() if isinstance(n, bytes) else n for n in names]
                for fi, name in enumerate(names_decoded):
                    net_key = next((k for k in YEO7_MAP if k in name), None)
                    if net_key:
                        labels[offset: offset + n][vtx_labels == fi] = YEO7_MAP[net_key]
                return labels
            except Exception as e:
                print(f'[networks] annot read failed: {e}')

    # Strategy 2: try nilearn's built-in Schaefer -> Yeo-7 mapping
    try:
        print('[networks] attempting Schaefer-100 -> Yeo-7 vertex mapping...')
        from nilearn import datasets as _d, surface as _s

        fs5 = _nl_datasets.fetch_surf_fsaverage('fsaverage5')

        for hemi, offset, n, h_key in [('left', 0, n_lh, 'left'), ('right', n_lh, n_rh, 'right')]:
            # Load Destrieux parcellation as a proxy (148 labels)
            # Map each label to closest Yeo network heuristically (not perfect)
            destrieux = _nl_datasets.fetch_atlas_surf_destrieux()
            parcel    = np.array(destrieux[f'map_{hemi}'], dtype=np.int16)

            # Destrieux parcels roughly aligned:
            # parcels 1-10: visual, 11-20: somatomotor, etc.
            # Use a rough heuristic: % position in parcel range → network
            max_p = parcel.max() or 1
            for vi in range(n):
                p = parcel[vi] if vi < len(parcel) else 0
                if p <= 0:
                    labels[offset + vi] = -1
                    continue
                frac = p / max_p
                if   frac < 0.14: labels[offset + vi] = 0  # Vis
                elif frac < 0.28: labels[offset + vi] = 1  # SomMot
                elif frac < 0.42: labels[offset + vi] = 2  # DorsAttn
                elif frac < 0.56: labels[offset + vi] = 3  # SalVentAttn
                elif frac < 0.70: labels[offset + vi] = 4  # Limbic
                elif frac < 0.84: labels[offset + vi] = 5  # Cont
                else:             labels[offset + vi] = 6  # Default

        print('[networks] Destrieux → Yeo-7 heuristic applied')
        return labels

    except Exception as e:
        print(f'[networks] fallback failed: {e}')

    # Strategy 3: uniform unknown (−1)
    print('[networks] WARNING: all vertices assigned to Unknown network')
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Export fsaverage5 pial → GLB + Yeo-7 labels')
    ap.add_argument('--no-networks', action='store_true', help='Skip network label export')
    ap.add_argument('--force',       action='store_true', help='Re-export even if files exist')
    args = ap.parse_args()

    OUT_GL.parent.mkdir(parents=True, exist_ok=True)

    if OUT_GL.exists() and not args.force:
        print(f'[mesh] {OUT_GL} already exists — pass --force to regenerate')
    else:
        # ── Load pial surfaces ────────────────────────────────────────────────
        lh_c, lh_f, rh_c, rh_f = load_fsaverage5_pial()
        n_lh = len(lh_c)
        n_rh = len(rh_c)
        print(f'[mesh] LH: {n_lh} verts, {len(lh_f)} faces')
        print(f'[mesh] RH: {n_rh} verts, {len(rh_f)} faces')

        # ── Merge hemispheres ─────────────────────────────────────────────────
        # Offset RH slightly on X axis so hemispheres don't overlap
        rh_c_shifted = rh_c.copy()
        # Centre LH and RH individually, then place side-by-side
        lh_c -= lh_c.mean(axis=0)
        rh_c_shifted -= rh_c_shifted.mean(axis=0)
        lh_max_x = lh_c[:, 0].max()
        rh_min_x = rh_c_shifted[:, 0].min()
        gap       = 4.0  # mm gap between hemispheres
        rh_c_shifted[:, 0] += (lh_max_x - rh_min_x + gap)

        verts = np.vstack([lh_c, rh_c_shifted])
        faces = np.vstack([lh_f, rh_f + n_lh])

        print(f'[mesh] Combined: {len(verts)} verts, {len(faces)} faces')

        # ── Compute normals ───────────────────────────────────────────────────
        print('[mesh] Computing vertex normals…')
        normals = compute_vertex_normals(verts, faces)

        # ── Build GLB ─────────────────────────────────────────────────────────
        print('[mesh] Building GLB…')
        glb = build_glb(verts, normals, faces)
        OUT_GL.write_bytes(glb)
        print(f'[mesh] OK {OUT_GL}  ({len(glb) / 1e6:.2f} MB)')

    if not args.no_networks:
        if OUT_NET.exists() and not args.force:
            print(f'[networks] {OUT_NET} already exists — pass --force to regenerate')
        else:
            lh_c, lh_f, rh_c, rh_f = load_fsaverage5_pial()
            fs5_info = _nl_datasets.fetch_surf_fsaverage('fsaverage5') if _nl_datasets else {}
            labels = fetch_yeo7_labels(fs5_info, len(lh_c), len(rh_c))
            OUT_NET.write_bytes(labels.tobytes())
            unique, cnts = np.unique(labels, return_counts=True)
            print(f'[networks] OK {OUT_NET}  ({len(labels)} vertices)')
            for u, c in zip(unique, cnts):
                name = {v: k for k, v in YEO7_MAP.items()}.get(u, 'Unknown')
                print(f'           {name:14s}: {c:5d} vertices')

    print('[export] Done.')


if __name__ == '__main__':
    main()
