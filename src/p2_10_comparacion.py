from pathlib import Path
from statistics import mean
import csv

import gymnasium as gym
from stable_baselines3 import PPO


NUM_EPISODIOS = 10


def evaluar_ppo(entorno, modelo, semilla):
    observacion, _ = entorno.reset(seed=semilla)
    terminado = False
    truncado = False
    recompensa_total = 0

    while not (terminado or truncado):
        accion, _ = modelo.predict(observacion, deterministic=True)
        observacion, recompensa, terminado, truncado, _ = entorno.step(
            int(accion)
        )
        recompensa_total += recompensa

    return recompensa_total


def evaluar_controlador_reglas(entorno, semilla):
    observacion, _ = entorno.reset(seed=semilla)
    terminado = False
    truncado = False
    recompensa_total = 0

    while not (terminado or truncado):
        angulo = observacion[2]
        accion = 1 if angulo > 0 else 0

        observacion, recompensa, terminado, truncado, _ = entorno.step(
            accion
        )
        recompensa_total += recompensa

    return recompensa_total


def resumir(recompensas):
    return {
        "mínima": min(recompensas),
        "promedio": mean(recompensas),
        "máxima": max(recompensas),
    }


def main():
    raiz = Path(__file__).resolve().parent.parent
    ruta_modelo = raiz / "models" / "ppo_cartpole.zip"
    directorio_reportes = raiz / "reports"

    ruta_csv = directorio_reportes / "p2_10_comparacion.csv"
    ruta_reporte = directorio_reportes / "p2_10_comparacion.md"

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    directorio_reportes.mkdir(parents=True, exist_ok=True)
    modelo = PPO.load(str(ruta_modelo))
    entorno = gym.make("CartPole-v1")

    resultados = []

    try:
        for episodio in range(1, NUM_EPISODIOS + 1):
            semilla = 200 + episodio

            recompensa_ppo = evaluar_ppo(entorno, modelo, semilla)
            recompensa_reglas = evaluar_controlador_reglas(
                entorno, semilla
            )

            resultados.append({
                "episodio": episodio,
                "recompensa_ppo": recompensa_ppo,
                "recompensa_controlador": recompensa_reglas,
            })
    finally:
        entorno.close()

    ppo = [fila["recompensa_ppo"] for fila in resultados]
    reglas = [fila["recompensa_controlador"] for fila in resultados]

    resumen_ppo = resumir(ppo)
    resumen_reglas = resumir(reglas)
    diferencia = resumen_ppo["promedio"] - resumen_reglas["promedio"]

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        campos = [
            "episodio",
            "recompensa_ppo",
            "recompensa_controlador",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)

    if diferencia > 0:
        conclusion = (
            f"PPO superó al controlador basado en reglas por "
            f"{diferencia:.2f} puntos de recompensa promedio."
        )
    elif diferencia < 0:
        conclusion = (
            f"El controlador basado en reglas superó a PPO por "
            f"{abs(diferencia):.2f} puntos de recompensa promedio."
        )
    else:
        conclusion = (
            "Ambos métodos obtuvieron la misma recompensa promedio."
        )

    lineas = [
        "# Comparación: PPO y controlador basado en reglas",
        "",
        f"Episodios evaluados: {NUM_EPISODIOS}",
        "",
        "| Episodio | PPO | Controlador basado en reglas |",
        "|---:|---:|---:|",
    ]

    for fila in resultados:
        lineas.append(
            f"| {fila['episodio']} | {fila['recompensa_ppo']} "
            f"| {fila['recompensa_controlador']} |"
        )

    lineas.extend([
        "",
        "## Resumen",
        "",
        "| Método | Mínima | Promedio | Máxima |",
        "|---|---:|---:|---:|",
        (
            f"| PPO | {resumen_ppo['mínima']} "
            f"| {resumen_ppo['promedio']:.2f} "
            f"| {resumen_ppo['máxima']} |"
        ),
        (
            f"| Reglas | {resumen_reglas['mínima']} "
            f"| {resumen_reglas['promedio']:.2f} "
            f"| {resumen_reglas['máxima']} |"
        ),
        "",
        "## Ventajas y desventajas",
        "",
        "- **PPO:** aprende una política a partir de la experiencia y "
        "usa la observación completa; requiere entrenamiento previo.",
        "- **Controlador por reglas:** es sencillo y no necesita "
        "entrenamiento; esta regla considera solo el ángulo del péndulo.",
        "",
        "## Conclusión",
        "",
        conclusion,
        "",
    ])

    ruta_reporte.write_text("\n".join(lineas), encoding="utf-8")

    print("COMPARACIÓN COMPLETADA")
    print(
        f"PPO — mínima: {resumen_ppo['mínima']}, "
        f"promedio: {resumen_ppo['promedio']:.2f}, "
        f"máxima: {resumen_ppo['máxima']}"
    )
    print(
        f"Reglas — mínima: {resumen_reglas['mínima']}, "
        f"promedio: {resumen_reglas['promedio']:.2f}, "
        f"máxima: {resumen_reglas['máxima']}"
    )
    print(f"\nConclusión: {conclusion}")
    print(f"CSV guardado en: {ruta_csv}")
    print(f"Reporte guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()