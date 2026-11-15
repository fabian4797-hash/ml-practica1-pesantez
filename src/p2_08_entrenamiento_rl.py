from pathlib import Path

import gymnasium as gym
import pandas as pd
import plotly.graph_objects as go
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    """Entrena PPO en CartPole y guarda el modelo y la curva de recompensas."""
    raiz_proyecto = Path(__file__).resolve().parent.parent

    directorio_modelos = raiz_proyecto / "models"
    directorio_registros = raiz_proyecto / "reports" / "rl_monitor"
    directorio_graficos = raiz_proyecto / "reports" / "figures"

    directorio_modelos.mkdir(parents=True, exist_ok=True)
    directorio_registros.mkdir(parents=True, exist_ok=True)
    directorio_graficos.mkdir(parents=True, exist_ok=True)

    ruta_csv = directorio_registros / "monitor.csv"
    ruta_modelo = directorio_modelos / "ppo_cartpole"
    ruta_grafico = directorio_graficos / "ppo_cartpole_recompensas.html"

    entorno = Monitor(
        gym.make("CartPole-v1"),
        filename=str(directorio_registros)
    )

    modelo = PPO(
        policy="MlpPolicy",
        env=entorno,
        verbose=1,
        seed=42
    )

    print("Iniciando entrenamiento de PPO...")
    modelo.learn(total_timesteps=10_000)

    modelo.save(str(ruta_modelo))
    entorno.close()

    datos = pd.read_csv(ruta_csv, comment="#")
    datos["episodio"] = range(1, len(datos) + 1)
    datos["promedio_ultimos_10"] = datos["r"].rolling(
        window=10,
        min_periods=1
    ).mean()

    figura = go.Figure()

    figura.add_trace(go.Scatter(
        x=datos["episodio"],
        y=datos["r"],
        mode="markers",
        name="Recompensa por episodio"
    ))

    figura.add_trace(go.Scatter(
        x=datos["episodio"],
        y=datos["promedio_ultimos_10"],
        mode="lines",
        name="Promedio de los últimos 10 episodios"
    ))

    figura.update_layout(
        title="Entrenamiento PPO en CartPole",
        xaxis_title="Episodio",
        yaxis_title="Recompensa acumulada",
        template="plotly_white"
    )

    figura.write_html(str(ruta_grafico))

    print(f"\nEpisodios completados: {len(datos)}")
    print(f"Recompensa promedio de los últimos 10 episodios: "
          f"{datos['r'].tail(10).mean():.2f}")
    print(f"Modelo guardado en: {ruta_modelo}.zip")
    print(f"Gráfica guardada en: {ruta_grafico}")


if __name__ == "__main__":
    main()
