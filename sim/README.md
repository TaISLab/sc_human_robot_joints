# Simulación numérica rápida (sin ROS, sin robot real)

Cierra el bucle de control compartido con un "usuario virtual" sintético
y un modelo cinemático del FR3, reutilizando **el mismo código** que
correrá en el robot real (`shared_control_core.py`, `performance.py`,
`path_follower.py`, `dh_utils.py`), para poder ajustar parámetros y
detectar problemas antes de la sesión con voluntarios.

## Qué NO sustituye

No sustituye la validación con humanos: el "usuario virtual" es un
controlador de admitancia sintético con ruido, no una persona real. Sirve
para verificar el software y ajustar ganancias/umbrales, no para
respaldar las hipótesis clínicas del paper.

## Uso

```bash
cd shared_control_pkg
python3 sim/closed_loop_sim.py --placement stressed --n_users 20
python3 sim/closed_loop_sim.py --placement nominal --n_users 20
```

Genera un CSV por condición (`sim_<placement>_<controlador>.csv`) con,
por usuario virtual: error de seguimiento medio, margen articular
mínimo, % tiempo con margen bajo, manipulabilidad mínima, % tiempo con
manipulabilidad baja.

## Ajuste de ganancias (`tune_gains.py`) — ya realizado

Con los pesos por defecto (todos =1), el controlador extendido (m=4)
**no** conseguía una manipulabilidad mínima más alta que el baseline
(m=2) en la colocación "exigente": la trayectoria circular obliga a
pasar por la zona de baja manipulabilidad, y con un peso bajo el factor
de manipulabilidad no era lo bastante influyente en la mezcla como para
cambiar el comportamiento resultante — de hecho, en varios casos
generaba picos de velocidad articular mayores (peor) que sin él.

Se hizo un barrido pareado (`sim/tune_gains.py`, mismos usuarios/semillas
para baseline y extendido) sobre `performance_weights.manipulability`,
`C_manipulability` y `dt_lookahead`. Conclusión:

- **Seguridad articular** (`joint_safety`): ya mejoraba el margen
  articular mínimo incluso con la ganancia por defecto, y sigue
  mejorando hasta `weight≈16`, `C_s≈12` (a partir de ahí, rendimientos
  decrecientes).
- **Manipulabilidad**: necesitó ganancias mucho más altas de lo esperado
  — con `weight≈1-8` el efecto era nulo o incluso contraproducente; a
  partir de `weight≈16-24`, `C_m≈16-24`, `dt_lookahead≈0.2 s` el efecto
  se invierte y mejora **simultáneamente** manipulabilidad mínima, %
  tiempo por debajo del umbral, pico de velocidad articular cerca de la
  singularidad, y el error de seguimiento (los cuatro a la vez, sin
  compromiso entre ellos).

Valores ya actualizados en `config/shared_control.yaml`
(`joint_safety: 16`, `manipulability: 24`, `C_joint_safety: 12`,
`C_manipulability: 24`, `dt_lookahead: 0.2`). Verificado también con el
controlador completo (m=4, ambos factores nuevos activos a la vez) sobre
3 usuarios virtuales adicionales: mejora consistente en margen
articular, manipulabilidad mínima y error de seguimiento en los tres.

Esto es un punto de partida razonable para la sesión real, no un
sustituto de reajustar con datos de personas reales — el usuario
sintético es un controlador de admitancia simplificado, no un modelo de
comportamiento humano.

## Módulos de este directorio

- `franka_fk.py` — cinemática del FR3 (DH modificada, valores publicados
  de Panda/FR3) **solo para simulación**; el nodo real usa el jacobiano
  exacto de `franka_ros`, no este módulo.
- `virtual_human_ik.py` — IK numérica del brazo humano de 4 GdL, solo
  para mover al "usuario virtual".
- `closed_loop_sim.py` — el bucle cerrado y el barrido Monte Carlo.
