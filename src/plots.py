"""Funciones de visualizacion para precio + indicadores tecnicos."""
import matplotlib.pyplot as plt


def plot_overlays(df, cols, title="Precio con overlays", figsize=(13, 5), ax=None):
    """Dibuja Close y una lista de columnas que viven en la misma escala de precio
    (medias moviles, bandas de Bollinger, etc.) sobre el mismo eje."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.1)
    for col in cols:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.0, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    if created:
        fig.tight_layout()
        return fig
    return ax


def plot_rsi(df, window_label="rsi_14", figsize=(13, 3), ax=None):
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[window_label], label=window_label, color="darkorange")
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_title("RSI")
    ax.set_ylabel("RSI")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    if created:
        fig.tight_layout()
        return fig
    return ax


def plot_macd(df, figsize=(13, 3), ax=None):
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df["macd"], label="MACD", color="blue", linewidth=1.0)
    ax.plot(df.index, df["macd_signal"], label="Signal", color="red", linewidth=1.0)
    ax.bar(df.index, df["macd_hist"], label="Hist", color="gray", alpha=0.4, width=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("MACD")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    if created:
        fig.tight_layout()
        return fig
    return ax


def plot_full_panel(df, overlay_cols=("sma_10", "sma_100"), figsize=(13, 11)):
    """Panel completo: precio+overlays, RSI y MACD apilados compartiendo el eje X."""
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3, 1.3]},
    )
    plot_overlays(df, overlay_cols, title="Precio + overlays", ax=axes[0])
    plot_rsi(df, ax=axes[1])
    plot_macd(df, ax=axes[2])
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    return fig
