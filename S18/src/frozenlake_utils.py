"""Funciones auxiliares para la sesión S17 (MDP, Bellman y DP) con FrozenLake.

Aquí solo hay código "de apoyo": construir P y R desde gymnasium y graficar.
Los algoritmos (evaluación, value iteration, policy iteration) se programan
en el notebook.
"""
import numpy as np
import matplotlib.pyplot as plt

ACCIONES = ["←", "↓", "→", "↑"]          # 0=Left, 1=Down, 2=Right, 3=Up
FLECHAS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def construir_modelo(env):
    """Extrae del entorno el modelo del MDP.

    Devuelve P con forma (nS, nA, nS) y R con forma (nS, nA):
        P[s, a, s'] = P(s' | s, a)
        R[s, a]     = sum_{s'} P(s'|s,a) * r(s,a,s')
    """
    nS, nA = env.observation_space.n, env.action_space.n
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for prob, s2, r, _ in env.unwrapped.P[s][a]:
                P[s, a, s2] += prob
                R[s, a] += prob * r
    return P, R


def estados_terminales(env):
    """Índices de hoyos y meta (los que se quedan en sí mismos con prob. 1)."""
    desc = env.unwrapped.desc.astype(str).flatten()
    return [i for i, c in enumerate(desc) if c in "HG"]


def _celdas(env):
    return env.unwrapped.desc.astype(str)


def graficar_V(V, env, titulo="V(s)", ax=None, fmt=".3f"):
    """Mapa de calor 4x4 del valor de cada celda."""
    desc = _celdas(env)
    n = desc.shape[0]
    ax = ax or plt.subplots(figsize=(4.2, 4.2))[1]
    M = np.round(np.array(V, dtype=float), 10).reshape(n, n) + 0.0   # evita "-0.000"
    terminal = np.isin(desc, ["H", "G"])
    ax.imshow(np.where(terminal, np.nan, M), cmap="YlGn", vmin=0)   # terminales en blanco
    for i in range(n):
        for j in range(n):
            c = desc[i, j]
            if c in "HG":
                ax.text(j, i, c, ha="center", va="center", fontsize=16,
                        color="red" if c == "H" else "darkgreen", weight="bold")
            else:
                ax.text(j, i, f"{M[i, j] + 0.0:{fmt}}", ha="center", va="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo)
    return ax


def graficar_Q(Q, env, titulo="Q(s,a)", ax=None, vmax=None):
    """Cada celda se divide en 4 triángulos (una por acción) con su valor Q."""
    desc = _celdas(env)
    n = desc.shape[0]
    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    terminal = np.isin(desc.flatten(), ["H", "G"])
    if vmax is None:                                   # escala solo con celdas no terminales
        vmax = max(np.max(np.array(Q)[~terminal]), 1e-9)
    cmap = plt.cm.YlGn
    # centros relativos de los textos para: ←, ↓, →, ↑
    off = {0: (-0.30, 0), 1: (0, 0.30), 2: (0.30, 0), 3: (0, -0.30)}
    for s in range(n * n):
        i, j = divmod(s, n)
        c = desc[i, j]
        cx, cy = j, i
        esquinas = [(cx - .5, cy - .5), (cx + .5, cy - .5),
                    (cx + .5, cy + .5), (cx - .5, cy + .5)]
        tri = {0: [esquinas[0], esquinas[3], (cx, cy)],
               1: [esquinas[3], esquinas[2], (cx, cy)],
               2: [esquinas[2], esquinas[1], (cx, cy)],
               3: [esquinas[1], esquinas[0], (cx, cy)]}
        if c in "HG":
            ax.add_patch(plt.Rectangle((cx - .5, cy - .5), 1, 1,
                                       color="#f4b6b6" if c == "H" else "#b6e8b6"))
            ax.text(cx, cy, c, ha="center", va="center", fontsize=16, weight="bold")
            continue
        for a in range(4):
            ax.add_patch(plt.Polygon(tri[a], facecolor=cmap(Q[s, a] / vmax),
                                     edgecolor="gray", lw=0.5))
            dx, dy = off[a]
            ax.text(cx + dx, cy + dy, f"{Q[s, a]:.2f}", ha="center",
                    va="center", fontsize=7)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo)
    return ax


def graficar_politica(politica, env, titulo="Política", ax=None):
    """Flechas de la acción elegida en cada celda. politica: array (nS,)."""
    desc = _celdas(env)
    n = desc.shape[0]
    ax = ax or plt.subplots(figsize=(4.2, 4.2))[1]
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    for s in range(n * n):
        i, j = divmod(s, n)
        c = desc[i, j]
        color = {"H": "#f4b6b6", "G": "#b6e8b6", "S": "#cfe3ff"}.get(c, "white")
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=color, edgecolor="gray"))
        txt = c if c in "HG" else FLECHAS[int(politica[s])]
        ax.text(j, i, txt, ha="center", va="center", fontsize=20)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo)
    return ax


def simular_politica(env, politica, n_episodios=2000, semilla=0):
    """Fracción de episodios que terminan en la meta siguiendo `politica` (nS,)."""
    exitos = 0
    obs, _ = env.reset(seed=semilla)
    for _ in range(n_episodios):
        obs, _ = env.reset()
        terminado = truncado = False
        while not (terminado or truncado):
            obs, r, terminado, truncado, _ = env.step(int(politica[obs]))
        exitos += r
    return exitos / n_episodios
