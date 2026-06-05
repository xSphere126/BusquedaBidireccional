# Busqueda Bidireccional

Implementacion de dos algoritmos de busqueda heuristica bidireccional en Python sobre un grafo ponderado no dirigido.

## Algoritmos

- **A* Bidireccional** — expande desde el nodo inicio y destino a la vez, parando cuando los dos frentes se encuentran
- **MM (Meet in the Middle)** — variante con una funcion de prioridad distinta que garantiza optimalidad con menos expansiones

## Requisitos

```
pip install matplotlib networkx
```

## Uso

```
python busqueda_bidireccional.py
```

Al ejecutarlo muestra por consola el camino optimo y los nodos explorados por cada frente, y abre dos graficas con la visualizacion de cada algoritmo.

Para cambiar el grafo o el par inicio/destino, edita las variables `EDGES`, `POSITIONS` y `START`/`GOAL` al principio del fichero.
