from graphviz import Digraph

def plot_cascade_model():
    # Crear un grafo dirigido
    dot = Digraph(comment='Modelo en Cascada de 3 Niveles')

    # Definir nodos
    dot.node('A', 'Nivel 1: Modelo de Clasificación')
    dot.node('B', 'Nivel 2: Modelo de Clasificación')
    dot.node('C', 'Nivel 3: Modelo de Clasificación')

    # Conectar los nodos (cascada)
    dot.edge('A', 'B', label='Salida -> Entrada')
    dot.edge('B', 'C', label='Salida -> Entrada')

    # Renderizar y mostrar el diagrama
    dot.render('modelo_en_cascada', format='png', view=True)

# Llamar a la función para crear la gráfica
plot_cascade_model()
