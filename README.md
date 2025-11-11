# Gestión de Energía en una Microrred con Generación Solar y Batería

Este proyecto implementa un sistema de gestión energética para una microrred compuesta por:
- Generación fotovoltaica (PV)
- Batería de almacenamiento
- Conexión bidireccional a la red eléctrica

El objetivo es **minimizar el costo total de operación** tomando decisiones óptimas de:
1. Cuánta energía comprar o vender a la red
2. Cuánta energía almacenar o extraer de la batería
3. Cómo garantizar operación segura evitando carga/descarga simultánea

El modelo se formula como un **Problema de Programación Lineal Entera Mixta (MILP)** y se resuelve utilizando `PuLP` en Python.

---

##  Estructura del repositorio
├── data/ # Datos de generación y demanda (o script para generarlos)
├── src/ # Funciones auxiliares (si aplica)
├── optproject.ipynb # Implementación completa del modelo y simulaciones
├── README.md # Este documento
└── results/ # Gráficas o resultados generados (opcional)

## Requerimientos

- Python 3.9 o superior
- PuLP
- NumPy
- Matplotlib
- Pandas (si aplica)

Instalación rápida:

```bash
!pip install pulp numpy matplotlib pandas
