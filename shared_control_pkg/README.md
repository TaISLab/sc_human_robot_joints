# shared_control_pkg

Módulo que calcula **únicamente** la acción robótica compartida (v_s) a
partir de las fuerzas de interacción (ya convertidas en v_h por vuestro
control de admitancia/impedancia existente) y del estado articular de
ambos brazos (humano, del sistema visuo-táctil; robot, del FR3). No
reimplementa nada de lo que ya tenéis: ni el control de
admitancia/impedancia, ni el pipeline sensorial.

## Contenido

- `dh_utils.py` – Cinemática del brazo humano de 4 GdL (Tabla DH del
  paper visuo-táctil) y su jacobiano numérico, usados solo para
  proyectar un comando cartesiano candidato a velocidades articulares
  humanas (q1̇..q4̇).
- `performance.py` – Los cuatro factores de rendimiento local:
  *smoothness*, *directness* (ya existentes en el framework de control
  compartido reactivo) y los **dos factores nuevos**: seguridad
  articular humana y evitación de singularidades del robot
  (manipulabilidad de Yoshikawa). Aquí está el núcleo de la
  contribución novedosa.
- `path_follower.py` – Seguidor de trayectoria reactivo (esfera
  virtual) idéntico al del framework de control compartido, más una
  clase `CirclePath` lista para la tarea de trazado circular.
- `robot_model.py` – Interfaz mínima para obtener el jacobiano del FR3
  en una configuración articular arbitraria (necesario para predecir
  el efecto de cada comando candidato sobre la manipulabilidad).
  Backend por defecto `franka_state` (usa `O_Jac_EE` de
  `franka_msgs/FrankaState`, sin dependencias extra); backend opcional
  `kdl` (PyKDL + `robot_description`) si necesitáis predicción exacta
  a un horizonte mayor.
- `shared_control_node.py` – Nodo ROS que conecta todo: se suscribe a
  vuestros topics existentes y publica el comando final.
- `config/shared_control.yaml` – Parámetros (pesos, constantes,
  límites articulares, definición de la trayectoria).
- `test/test_performance_offline.py` – Verificación sin ROS ni
  hardware (ya ejecutada: todo pasa).

## Integración: topics que el nodo espera

| Topic | Tipo | Origen |
|---|---|---|
| `/admittance_control/human_velocity` | `geometry_msgs/TwistStamped` | Vuestro control de admitancia (v_h) |
| `/human_arm/joint_state` | `sensor_msgs/JointState` (nombres `q1,q2,q3,q4`, opcional `l1,l2`) | Vuestro sistema visuo-táctil (salida de la IK, Sec. 3.4 del paper 1) |
| `/franka_state_controller/franka_states` | `franka_msgs/FrankaState` | Nativo de `franka_ros` |

Publica en `/shared_control/cartesian_velocity_command`
(`geometry_msgs/TwistStamped`), listo para alimentar el controlador de
velocidad cartesiana del FR3.

**Si vuestros nombres de topic/mensaje difieren**, solo hay que tocar
las tres suscripciones en `shared_control_node.py` (líneas del
`__init__` / callbacks); el resto del código no depende de ellos.

## Límites articulares por defecto

En `performance.py::DEFAULT_JOINT_LIMITS` hay valores orientativos de
goniometría (ROM) para q1..q4. **Recomendado ajustarlos** a los rangos
reales de vuestro protocolo/población antes de los experimentos
(parametrizables también vía `config/shared_control.yaml` si preferís
pasarlos como parámetro ROS en vez de constante).

## Cómo verificar antes del robot real

```bash
cd shared_control_pkg
python3 test/test_performance_offline.py
```

## Limitación conocida (documentada también en el paper)

Con la tabla D-H de 4 GdL del paper visuo-táctil, la columna del
jacobiano correspondiente a q4 (flexo-extensión de codo) tiene norma
casi nula en algunas posturas, porque esa articulación traslada la
muñeca a lo largo de su propio eje de rotación. Por eso el factor de
seguridad articular combina un término **estático** (proximidad al
límite, siempre activo) con uno **dinámico** (tasa de cierre proyectada
vía jacobiano); así q4 no queda "ciego" a la penalización aunque el
acoplamiento cartesiano instantáneo sea débil.
