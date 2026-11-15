from pathlib import Path
from statistics import mean
import csv

import gymnasium as gym
from stable_baselines3 import PPO


NUM_EPISODIOS = 10


def ejecutar_episodio_ppo(entorno, modelo, semilla):
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


def ejecutar_episodio_aleatorio(entorno, semilla):
    entorno.action_space.seed(semilla + 1000)
    observacion, _ = entorno.reset(seed=semilla)
    terminado = False
    truncado = False
    recompensa_total = 0

    while not (terminado or truncado):
        accion = entorno.action_space.sample()
        observacion, recompensa, terminado, truncado, _ = entorno.step(accion)
        recompensa_total += recompensa

    return recompensa_total


def resumir_recompensas(recompensas):
    return {
        "mínima": min(recompensas),
        "promedio": mean(recompensas),
        "máxima": max(recompensas),
    }


def main():
    raiz_proyecto = Path(__file__).resolve().parent.parent
    ruta_modelo = raiz_proyecto / "models" / "ppo_cartpole.zip"
    directorio_reportes = raiz_proyecto / "reports"
    ruta_csv = directorio_reportes / "p2_09_evaluacion_rl.csv"
    ruta_reporte = directorio_reportes / "p2_09_evaluacion_rl.md"

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    directorio_reportes.mkdir(parents=True, exist_ok=True)
    modelo = PPO.load(str(ruta_modelo))
    entorno = gym.make("CartPole-v1")

    resultados = []

    try:
        for episodio in range(1, NUM_EPISODIOS + 1):
            semilla = 100 + episodio

            recompensa_ppo = ejecutar_episodio_ppo(
                entorno, modelo, semilla
            )
            recompensa_aleatoria = ejecutar_episodio_aleatorio(
                entorno, semilla
            )

            resultados.append({
                "episodio": episodio,
                "recompensa_ppo": recompensa_ppo,
                "recompensa_aleatoria": recompensa_aleatoria,
            })
    finally:
        entorno.close()

    recompensas_ppo = [fila["recompensa_ppo"] for fila in resultados]
    recompensas_aleatorias = [
        fila["recompensa_aleatoria"] for fila in resultados
    ]

    resumen_ppo = resumir_recompensas(recompensas_ppo)
    resumen_aleatorio = resumir_recompensas(recompensas_aleatorias)
    diferencia = resumen_ppo["promedio"] - resumen_aleatorio["promedio"]

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        campos = ["episodio", "recompensa_ppo", "recompensa_aleatoria"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)

    if diferencia > 0:
        conclusion = (
            f"En estos 10 episodios, PPO obtuvo una recompensa promedio "
            f"{diferencia:.2f} puntos mayor que el agente aleatorio."
        )
    elif diferencia < 0:
        conclusion = (
            f"En estos 10 episodios, PPO obtuvo una recompensa promedio "
            f"{abs(diferencia):.2f} puntos menor que el agente aleatorio."
        )
    else:
        conclusion = (
            "En estos 10 episodios, ambos agentes obtuvieron la misma "
            "recompensa promedio."
        )

    lineas = [
        "# Evaluación del agente PPO en CartPole",
        "",
        f"Episodios evaluados: {NUM_EPISODIOS}",
        "",
        "| Episodio | PPO | Agente aleatorio |",
        "|---:|---:|---:|",
    ]

    for fila in resultados:
        lineas.append(
            f"| {fila['episodio']} | {fila['recompensa_ppo']} "
            f"| {fila['recompensa_aleatoria']} |"
        )

    lineas.extend([
        "",
        "## Resumen de recompensas",
        "",
        "| Agente | Mínima | Promedio | Máxima |",
        "|---|---:|---:|---:|",
        (
            f"| PPO | {resumen_ppo['mínima']} "
            f"| {resumen_ppo['promedio']:.2f} "
            f"| {resumen_ppo['máxima']} |"
        ),
        (
            f"| Aleatorio | {resumen_aleatorio['mínima']} "
            f"| {resumen_aleatorio['promedio']:.2f} "
            f"| {resumen_aleatorio['máxima']} |"
        ),
        "",
        "## Conclusión",
        "",
        conclusion,
        "",
    ])

    ruta_reporte.write_text("\n".join(lineas), encoding="utf-8")

    print("EVALUACIÓN COMPLETADA")
    print(f"Episodios: {NUM_EPISODIOS}")
    print(
        f"PPO — mínima: {resumen_ppo['mínima']}, "
        f"promedio: {resumen_ppo['promedio']:.2f}, "
        f"máxima: {resumen_ppo['máxima']}"
    )
    print(
        f"Aleatorio — mínima: {resumen_aleatorio['mínima']}, "
        f"promedio: {resumen_aleatorio['promedio']:.2f}, "
        f"máxima: {resumen_aleatorio['máxima']}"
    )
    print(f"\nConclusión: {conclusion}")
    print(f"CSV guardado en: {ruta_csv}")
    print(f"Reporte guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()
    