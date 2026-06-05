#!/usr/bin/env python3
"""
Busqueda heuristica bidireccional: A* Bidireccional y MM (Meet in the Middle).
"""

import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# Grafo de ejemplo: aristas (u, v, peso) y posiciones (x, y) para la heuristica
EDGES = [
    ("A", "B", 4), ("A", "C", 2),
    ("B", "C", 5), ("B", "D", 10),
    ("C", "E", 3),
    ("D", "F", 11), ("D", "G", 4),
    ("E", "D", 4), ("E", "H", 7),
    ("F", "Z", 3),
    ("G", "Z", 6),
    ("H", "G", 2), ("H", "Z", 12),
]

POSITIONS = {
    "A": (0, 2), "B": (1, 3), "C": (1, 1),
    "D": (3, 3), "E": (2, 1), "F": (5, 3),
    "G": (4, 2), "H": (3, 0), "Z": (6, 2),
}


def build_adj(edges):
    """Construye lista de adyacencia para grafo no dirigido."""
    adj = {}
    for u, v, w in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def heuristic(node, goal, pos):
    """Distancia euclidea entre dos nodos."""
    x1, y1 = pos[node]
    x2, y2 = pos[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path(parent_fwd, parent_bwd, meeting):
    """Une el camino hacia adelante y hacia atras desde el nodo de encuentro."""
    path = []
    node = meeting
    while node is not None:
        path.append(node)
        node = parent_fwd.get(node)
    path.reverse()

    node = parent_bwd.get(meeting)
    while node is not None:
        path.append(node)
        node = parent_bwd.get(node)

    return path


def astar_bidireccional(adj, pos, start, goal):
    """
    A* bidireccional: expande desde start y goal alternativamente.
    Devuelve (camino, coste, cerrados_fwd, cerrados_bwd).
    """
    # Cola: (f, g, nodo)
    open_fwd = [(heuristic(start, goal, pos), 0, start)]
    open_bwd = [(heuristic(goal, start, pos), 0, goal)]

    g_fwd = {start: 0}
    g_bwd = {goal: 0}

    parent_fwd = {start: None}
    parent_bwd = {goal: None}

    closed_fwd = set()
    closed_bwd = set()

    best = math.inf
    meeting = None

    while open_fwd or open_bwd:

        # Expandimos el frente hacia adelante
        if open_fwd:
            f, g, node = heapq.heappop(open_fwd)
            if node not in closed_fwd:
                closed_fwd.add(node)
                for neighbor, w in adj.get(node, []):
                    new_g = g_fwd[node] + w
                    if neighbor not in g_fwd or new_g < g_fwd[neighbor]:
                        g_fwd[neighbor] = new_g
                        parent_fwd[neighbor] = node
                        f_new = new_g + heuristic(neighbor, goal, pos)
                        heapq.heappush(open_fwd, (f_new, new_g, neighbor))
                    # Comprobamos si el vecino ya fue alcanzado por el otro frente
                    if neighbor in g_bwd:
                        candidate = g_fwd[node] + w + g_bwd[neighbor]
                        if candidate < best:
                            best = candidate
                            meeting = neighbor

        # Expandimos el frente hacia atras
        if open_bwd:
            f, g, node = heapq.heappop(open_bwd)
            if node not in closed_bwd:
                closed_bwd.add(node)
                for neighbor, w in adj.get(node, []):
                    new_g = g_bwd[node] + w
                    if neighbor not in g_bwd or new_g < g_bwd[neighbor]:
                        g_bwd[neighbor] = new_g
                        parent_bwd[neighbor] = node
                        f_new = new_g + heuristic(neighbor, start, pos)
                        heapq.heappush(open_bwd, (f_new, new_g, neighbor))
                    if neighbor in g_fwd:
                        candidate = g_bwd[node] + w + g_fwd[neighbor]
                        if candidate < best:
                            best = candidate
                            meeting = neighbor

        # Paramos cuando el minimo f de ambas colas supera la mejor solucion encontrada
        min_fwd = open_fwd[0][0] if open_fwd else math.inf
        min_bwd = open_bwd[0][0] if open_bwd else math.inf
        if min_fwd + min_bwd >= best:
            break

    if meeting is None:
        return None, math.inf, closed_fwd, closed_bwd

    path = reconstruct_path(parent_fwd, parent_bwd, meeting)
    return path, best, closed_fwd, closed_bwd


def pr(g, h):
    """Funcion de prioridad del algoritmo MM: max(g+h, 2g)."""
    return max(g + h, 2 * g)


def mm(adj, pos, start, goal):
    """
    Algoritmo MM (Meet in the Middle): garantiza optimalidad con menos expansiones que A*.
    Devuelve (camino, coste, cerrados_fwd, cerrados_bwd).
    """
    open_fwd = [(pr(0, heuristic(start, goal, pos)), 0, start)]
    open_bwd = [(pr(0, heuristic(goal, start, pos)), 0, goal)]

    g_fwd = {start: 0}
    g_bwd = {goal: 0}

    parent_fwd = {start: None}
    parent_bwd = {goal: None}

    closed_fwd = set()
    closed_bwd = set()

    U = math.inf
    meeting = None

    while open_fwd and open_bwd:

        # Condicion de parada: la mejor solucion es menor o igual que el minimo de prioridad
        C = min(open_fwd[0][0], open_bwd[0][0])
        if U <= C:
            break

        # Expandimos el frente con menor prioridad
        if open_fwd[0][0] <= open_bwd[0][0]:
            _, g, node = heapq.heappop(open_fwd)
            if node not in closed_fwd:
                closed_fwd.add(node)
                for neighbor, w in adj.get(node, []):
                    new_g = g_fwd[node] + w
                    if neighbor not in g_fwd or new_g < g_fwd[neighbor]:
                        g_fwd[neighbor] = new_g
                        parent_fwd[neighbor] = node
                        h = heuristic(neighbor, goal, pos)
                        heapq.heappush(open_fwd, (pr(new_g, h), new_g, neighbor))
                    if neighbor in g_bwd:
                        candidate = g_fwd[node] + w + g_bwd[neighbor]
                        if candidate < U:
                            U = candidate
                            meeting = neighbor
        else:
            _, g, node = heapq.heappop(open_bwd)
            if node not in closed_bwd:
                closed_bwd.add(node)
                for neighbor, w in adj.get(node, []):
                    new_g = g_bwd[node] + w
                    if neighbor not in g_bwd or new_g < g_bwd[neighbor]:
                        g_bwd[neighbor] = new_g
                        parent_bwd[neighbor] = node
                        h = heuristic(neighbor, start, pos)
                        heapq.heappush(open_bwd, (pr(new_g, h), new_g, neighbor))
                    if neighbor in g_fwd:
                        candidate = g_bwd[node] + w + g_fwd[neighbor]
                        if candidate < U:
                            U = candidate
                            meeting = neighbor

    if meeting is None:
        return None, math.inf, closed_fwd, closed_bwd

    path = reconstruct_path(parent_fwd, parent_bwd, meeting)
    return path, U, closed_fwd, closed_bwd


def draw_result(G, pos, path, explored_fwd, explored_bwd, start, goal, title):
    """Dibuja el grafo coloreando los nodos segun su estado en la busqueda."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = []
    for n in G.nodes():
        if n == start:
            colors.append("#2ecc71")
        elif n == goal:
            colors.append("#e74c3c")
        elif path and n in path:
            colors.append("#f39c12")
        elif n in explored_fwd and n in explored_bwd:
            colors.append("#9b59b6")
        elif n in explored_fwd:
            colors.append("#3498db")
        elif n in explored_bwd:
            colors.append("#e67e22")
        else:
            colors.append("#bdc3c7")

    path_edges = list(zip(path, path[1:])) if path else []
    other_edges = [e for e in G.edges()
                   if e not in path_edges and (e[1], e[0]) not in path_edges]

    nx.draw_networkx_edges(G, pos, edgelist=other_edges, ax=ax, edge_color="#95a5a6", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax, edge_color="#e74c3c", width=3.5)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "weight"), ax=ax, font_size=9)

    legend = [
        mpatches.Patch(color="#2ecc71", label="Inicio"),
        mpatches.Patch(color="#e74c3c", label="Destino"),
        mpatches.Patch(color="#f39c12", label="Camino optimo"),
        mpatches.Patch(color="#3498db", label="Explorado (adelante)"),
        mpatches.Patch(color="#e67e22", label="Explorado (atras)"),
        mpatches.Patch(color="#9b59b6", label="Explorado (ambos)"),
        mpatches.Patch(color="#bdc3c7", label="No explorado"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    START = "A"
    GOAL = "Z"

    adj = build_adj(EDGES)

    print(f"Busqueda bidireccional: {START} -> {GOAL}\n")

    path_astar, cost_astar, closed_fwd_a, closed_bwd_a = astar_bidireccional(adj, POSITIONS, START, GOAL)
    print("[A* Bidireccional]")
    print(f"  Camino : {' -> '.join(path_astar) if path_astar else 'No encontrado'}")
    print(f"  Coste  : {cost_astar}")
    print(f"  Nodos explorados (fwd): {len(closed_fwd_a)}, (bwd): {len(closed_bwd_a)}")

    path_mm, cost_mm, closed_fwd_m, closed_bwd_m = mm(adj, POSITIONS, START, GOAL)
    print("\n[MM - Meet in the Middle]")
    print(f"  Camino : {' -> '.join(path_mm) if path_mm else 'No encontrado'}")
    print(f"  Coste  : {cost_mm}")
    print(f"  Nodos explorados (fwd): {len(closed_fwd_m)}, (bwd): {len(closed_bwd_m)}")

    G = nx.Graph()
    for u, v, w in EDGES:
        G.add_edge(u, v, weight=w)

    draw_result(G, POSITIONS, path_astar, closed_fwd_a, closed_bwd_a, START, GOAL,
                f"A* Bidireccional  |  {' -> '.join(path_astar or [])}  |  Coste: {cost_astar}")
    draw_result(G, POSITIONS, path_mm, closed_fwd_m, closed_bwd_m, START, GOAL,
                f"MM (Meet in the Middle)  |  {' -> '.join(path_mm or [])}  |  Coste: {cost_mm}")

    plt.show()


if __name__ == "__main__":
    main()
