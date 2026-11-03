"""Gráficos do sanity_check: comparação por participante, EAR no tempo e distribuição do MAR."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detection import parameters as P


def style_axis(ax, title: str, ylabel: str = "") -> None:
    """Estilo recessivo: sem moldura, grade só horizontal e discreta."""
    ax.set_title(title, loc="left", fontsize=P.TITLE_FONT_SIZE, color=P.INK_COLOR)
    ax.set_ylabel(ylabel, fontsize=P.LABEL_FONT_SIZE, color=P.INK_COLOR)
    ax.tick_params(labelsize=P.LABEL_FONT_SIZE, colors=P.INK_COLOR, length=0)
    ax.grid(axis="y", color=P.GRID_COLOR, linewidth=P.GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(P.GRID_COLOR)


def plot_comparison(table: pd.DataFrame, out_dir: Path) -> None:
    """Barras agrupadas (rótulos 0, 5, 10) por participante, uma janela por métrica."""
    metrics = [("mean_ear", "EAR médio", ""), ("perclos_pct", "PERCLOS", "% dos frames"),
               ("blinks_per_min", "Piscadas por minuto", "piscadas/min"),
               ("yawns", "Bocejos", "contagem (>=1 s)"), ("mean_pitch", "Pitch médio", "graus"),
               ("pct_jump_any", f"Instabilidade (salto > {P.JUMP_DEG:.0f}° pitch ou yaw)",
                "% dos pares de frames")]
    participants = sorted(table["participant"].unique())
    fig, axes = plt.subplots(2, 3, figsize=P.COMPARISON_FIGSIZE, constrained_layout=True)
    for ax, (column, title, ylabel) in zip(axes.ravel(), metrics):
        for k, label in enumerate(P.LABEL_NAMES):
            values = [table[(table.participant == p) & (table.label == label)][column].mean()
                      for p in participants]
            ax.bar(np.arange(len(participants)) + (k - 1) * P.BAR_WIDTH, values, P.BAR_WIDTH * P.BAR_FILL,
                   color=P.LABEL_COLORS[label], label=P.LABEL_NAMES[label])
        ax.set_xticks(range(len(participants)), [f"P{p}" for p in participants])
        style_axis(ax, title, ylabel)
        if column != "mean_pitch":        # contagens e porcentagens nunca são negativas
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=P.LABEL_FONT_SIZE, frameon=False, loc="lower left")
    fig.suptitle("Alerta vs sonolento por participante (limiares relativos ao próprio vídeo)",
                 x=0.01, ha="left", fontsize=P.SUPTITLE_FONT_SIZE, color=P.INK_COLOR)
    fig.savefig(out_dir / "comparacao_por_participante.png", dpi=P.COMPARISON_DPI)
    plt.close(fig)


def plot_ear_timeseries(data: dict, out_dir: Path) -> None:
    """EAR médio ao longo do tempo, um painel por participante, uma linha por rótulo."""
    participants = sorted({key[0] for key in data})
    fig, axes = plt.subplots(len(participants), 1, figsize=(P.TIMESERIES_WIDTH, P.TIMESERIES_ROW_HEIGHT * len(participants)),
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, participant in zip(axes.ravel(), participants):
        for label in P.LABEL_NAMES:
            if (participant, label) not in data:
                continue
            frame = data[(participant, label)]
            smoothed = frame["ear_mean"].rolling(P.TIMESERIES_SMOOTH_FRAMES, center=True, min_periods=1).median()
            ax.plot(frame["timestamp_s"] / P.SECONDS_PER_MINUTE, smoothed, lw=P.TIMESERIES_LINE_WIDTH,
                    color=P.LABEL_COLORS[label], label=P.LABEL_NAMES[label])
        style_axis(ax, f"Participante {participant}: EAR médio ao longo do tempo "
                       f"(mediana móvel de {P.TIMESERIES_SMOOTH_FRAMES} frames)", "EAR")
        ax.set_xlabel("minutos", fontsize=P.LABEL_FONT_SIZE, color=P.INK_COLOR)
    axes[0, 0].legend(fontsize=P.LABEL_FONT_SIZE, frameon=False, ncol=3, loc="lower right")
    fig.savefig(out_dir / "ear_ao_longo_do_tempo.png", dpi=P.TIMESERIES_DPI)
    plt.close(fig)


def plot_mar_distribution(table: pd.DataFrame, out_dir: Path) -> None:
    """Percentis do MAR por vídeo, com o limiar de bocejo relativo de cada um."""
    table = table.sort_values(["participant", "label"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(P.MAR_PLOT_WIDTH, P.MAR_PLOT_ROW_HEIGHT * len(table) + P.MAR_PLOT_PADDING),
                           constrained_layout=True)
    for y, row in table.iterrows():
        color = P.LABEL_COLORS[row.label]
        ax.hlines(y, row.mar_p5, row.mar_p95, color=color, lw=P.MAR_PLOT_THIN_LINE)
        ax.hlines(y, row.mar_p25, row.mar_p75, color=color, lw=P.MAR_PLOT_THICK_LINE)
        ax.plot(row.mar_p50, y, "o", color="white", ms=P.MAR_PLOT_MEDIAN_SIZE, mec=color)
        ax.plot(row.mar_yawn_threshold, y, "|", color=P.YAWN_MARK_COLOR, ms=P.MAR_PLOT_THRESHOLD_SIZE)
    ax.set_yticks(range(len(table)), [f"P{row.participant} / {row.label}" for _, row in table.iterrows()])
    ax.invert_yaxis()
    ax.axvline(P.MAR_FIXED_REFERENCE, color=P.MAR_FIXED_REFERENCE_COLOR, lw=P.GRID_LINE_WIDTH, ls=":")
    ax.text(P.MAR_FIXED_REFERENCE + 0.005, 0.98, "0,50 (limiar fixo atual)", fontsize=P.NOTE_FONT_SIZE,
            color=P.MAR_FIXED_LABEL_COLOR, transform=ax.get_xaxis_transform(), va="top")
    style_axis(ax, "MAR por vídeo: p5–p95 (linha fina), p25–p75 (grossa), mediana (ponto), "
                   "limiar de bocejo relativo (traço vermelho)")
    ax.grid(axis="x", color=P.GRID_COLOR)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAR", fontsize=P.LABEL_FONT_SIZE, color=P.INK_COLOR)
    fig.savefig(out_dir / "mar_distribuicao.png", dpi=P.MAR_PLOT_DPI)
    plt.close(fig)
