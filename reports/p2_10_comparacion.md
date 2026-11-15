# Comparación: PPO y controlador basado en reglas

Episodios evaluados: 10

| Episodio | PPO | Controlador basado en reglas |
|---:|---:|---:|
| 1 | 481.0 | 55.0 |
| 2 | 264.0 | 48.0 |
| 3 | 251.0 | 58.0 |
| 4 | 424.0 | 47.0 |
| 5 | 219.0 | 51.0 |
| 6 | 233.0 | 39.0 |
| 7 | 259.0 | 56.0 |
| 8 | 477.0 | 34.0 |
| 9 | 500.0 | 46.0 |
| 10 | 463.0 | 46.0 |

## Resumen

| Método | Mínima | Promedio | Máxima |
|---|---:|---:|---:|
| PPO | 219.0 | 357.10 | 500.0 |
| Reglas | 34.0 | 48.00 | 58.0 |

## Ventajas y desventajas

- **PPO:** aprende una política a partir de la experiencia y usa la observación completa; requiere entrenamiento previo.
- **Controlador por reglas:** es sencillo y no necesita entrenamiento; esta regla considera solo el ángulo del péndulo.

## Conclusión

PPO superó al controlador basado en reglas por 309.10 puntos de recompensa promedio.
