"""Sesión 7: creación de un entorno RL personalizado."""

import gymnasium as gym
from gymnasium import spaces


class EntornoLinea(gym.Env):
    """Entorno de cinco posiciones con una meta al extremo derecho."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode="ansi", max_pasos=10):
        """Inicializa el entorno y define sus espacios."""
        super().__init__()

        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode debe ser None o 'ansi'.")

        self.render_mode = render_mode
        self.max_pasos = max_pasos
        self.meta = 4

        # Estados posibles: posiciones 0, 1, 2, 3 y 4.
        self.observation_space = spaces.Discrete(5)

        # Acciones: 0 = izquierda, 1 = derecha.
        self.action_space = spaces.Discrete(2)

        self.posicion = 0
        self.pasos = 0

    def reset(self, seed=None, options=None):
        """Reinicia el episodio en la posición inicial."""
        super().reset(seed=seed)

        self.posicion = 0
        self.pasos = 0

        observacion = self.posicion
        informacion = {"posicion": self.posicion}

        return observacion, informacion

    def step(self, accion):
        """Ejecuta una acción y devuelve observación, recompensa y finalización."""
        if not self.action_space.contains(accion):
            raise ValueError(f"Acción inválida: {accion}")

        if accion == 0:
            self.posicion = max(0, self.posicion - 1)
        else:
            self.posicion = min(self.meta, self.posicion + 1)

        self.pasos += 1

        llegado_a_meta = self.posicion == self.meta
        recompensa = 10 if llegado_a_meta else -1

        terminado = llegado_a_meta
        truncado = self.pasos >= self.max_pasos and not terminado

        observacion = self.posicion
        informacion = {
            "posicion": self.posicion,
            "pasos": self.pasos,
        }

        return observacion, recompensa, terminado, truncado, informacion

    def render(self):
        """Representa la línea y la posición actual del agente."""
        celdas = []

        for posicion in range(self.meta + 1):
            if posicion == self.posicion == self.meta:
                celdas.append("A/G")
            elif posicion == self.posicion:
                celdas.append("A")
            elif posicion == self.meta:
                celdas.append("G")
            else:
                celdas.append("·")

        return "[" + " | ".join(celdas) + "]"


def main():
    """Prueba el entorno con cinco episodios de acciones aleatorias."""
    entorno = EntornoLinea(render_mode="ansi")

    for numero_episodio in range(1, 6):
        observacion, informacion = entorno.reset(seed=40 + numero_episodio)
        recompensa_total = 0
        terminado = False
        truncado = False

        print(f"\n--- Episodio {numero_episodio} ---")
        print(entorno.render())

        while not (terminado or truncado):
            accion = entorno.action_space.sample()

            (
                observacion,
                recompensa,
                terminado,
                truncado,
                informacion,
            ) = entorno.step(accion)

            recompensa_total += recompensa

            nombre_accion = "izquierda" if accion == 0 else "derecha"

            print(
                f"Acción: {nombre_accion:10} | "
                f"Estado: {observacion} | "
                f"Recompensa: {recompensa:>3} | "
                f"Terminado: {terminado} | "
                f"Truncado: {truncado}"
            )
            print(entorno.render())

        print(f"Recompensa total: {recompensa_total}")

    entorno.close()


if __name__ == "__main__":
    main()