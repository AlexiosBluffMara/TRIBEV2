"""Render TRIBE v2 predictions as a slide-ready cortex PNG and ROI stream MP4."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .pipeline import InferenceResult


def render_peak_cortex(result: InferenceResult, out_path: Path | None = None) -> Path:
    """Static inflated cortex (LH+RH) at peak predicted activity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    fsavg5 = datasets.fetch_surf_fsaverage("fsaverage5")
    frame = result.preds[result.peak_t]
    lh, rh = frame[:10242], frame[10242:]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plotting.plot_surf_stat_map(
        fsavg5.infl_left, lh, hemi="left", view="lateral",
        bg_map=fsavg5.sulc_left, cmap="cold_hot", axes=ax1,
        title=f"LH @ t={result.peak_t/2:.1f}s", colorbar=False,
    )
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plotting.plot_surf_stat_map(
        fsavg5.infl_right, rh, hemi="right", view="lateral",
        bg_map=fsavg5.sulc_right, cmap="cold_hot", axes=ax2,
        title=f"RH @ t={result.peak_t/2:.1f}s", colorbar=True,
    )
    fig.suptitle(
        f"TRIBE v2 — predicted BOLD @ {result.peak_t/2:.1f}s (peak activity)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = out_path or config.OUT_DIR / "brain_peak.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_roi_stream(result: InferenceResult, out_path: Path | None = None) -> Path:
    """Streaming panel: top-12 ROI bar chart + running mean line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    df = result.roi_df[result.top_rois]
    T = df.shape[0]

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[1, 0])

    bars = ax_bar.barh(range(len(result.top_rois)), np.zeros(len(result.top_rois)))
    ax_bar.set_yticks(range(len(result.top_rois)))
    ax_bar.set_yticklabels([r[:40] for r in result.top_rois], fontsize=8)
    ax_bar.set_xlim(float(df.min().min()) * 1.1, float(df.max().max()) * 1.1)
    ax_bar.invert_yaxis()
    ax_bar.set_title("Top-12 Schaefer-400 ROI activations (live)")

    (line,) = ax_ts.plot([], [], lw=1.2)
    ax_ts.set_xlim(0, T / 2)
    ax_ts.set_ylim(df.mean(axis=1).min() * 1.1, df.mean(axis=1).max() * 1.1)
    ax_ts.set_xlabel("time (s)")
    ax_ts.set_ylabel("mean top-12 ROI z")

    def update(t: int):
        vals = df.iloc[t].values
        for b, v in zip(bars, vals):
            b.set_width(float(v))
        line.set_data(np.arange(t + 1) / 2, df.iloc[: t + 1].mean(axis=1).values)
        return [*bars, line]

    anim = FuncAnimation(fig, update, frames=T, interval=500, blit=False)
    out_path = out_path or config.OUT_DIR / "tribev2_stream.mp4"
    try:
        anim.save(str(out_path), writer=FFMpegWriter(fps=2, bitrate=2400))
    except Exception as e:
        print(f"[visualize] FFmpeg unavailable ({e}); falling back to GIF.")
        out_path = out_path.with_suffix(".gif")
        anim.save(str(out_path), writer="pillow", fps=2)
    plt.close(fig)
    return out_path
