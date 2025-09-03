

# uv run main la camara de aca 
# uv run main  --dataset test2 corre el video del pasto (hablar con luca manana)


# Visión general rápida
**Qué ves en pantalla:** el visor 3D (OpenGL + pygame) dibuja:
- una **franja de cámaras** (frustums) que representan las poses estimadas (roja la actual, verdes las pasadas)
- una **polilínea** azul con la trayectoria (traducciones acumuladas)
- además, hay un visor 2D donde se pintan los **matches de features** entre cuadros consecutivos.

**Flujo de datos de un frame:**
1) `main.py` carga config y dataset → crea `Camera` y un `DataLoader` (imágenes o video).
2) En el bucle: redimensiona el frame y llama a `vo.process_frame(img, i)`.
3) `VisualOdometry` detecta y describe features (ORB), matchea contra el frame previo, estima **R** y **t** (matriz esencial) y **acumula la pose**.
4) `Display2D` pinta matches en el frame; `Display3D` recibe `poses` y `translations` y actualiza el mapa 3D.

---

# `camera.py`
**Propósito:** encapsular parámetros intrínsecos de cámara y convertir puntos entre coordenadas de pixel y **coordenadas normalizadas** de cámara.

- `Camera.__init__(width, height, fx, fy, cx, cy)`
  - Guarda dims y parámetros intrínsecos.
  - Construye la matriz **K** = \[ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ] y su inversa `Kinv`.
  - `pp = (cx, cy)` es el **punto principal**.

- `__str__`: imprime `K` conveniente para debug.

- `normalize_pts(self, pts)`: **No implementado**. El comentario dice *“3D world → 2D image”*, pero no se usa.

- `denormalize_pts(self, pts)`
  - **OJO con el nombre/comentario:** el comentario dice *“2D image → 3D world”*, pero **lo que hace realmente** es: toma puntos **en pixel** `(u, v)`, les añade 1 para homogéneas y aplica `Kinv` → devuelve **coordenadas normalizadas** `(x', y')` en el plano z=1 de la cámara.
  - Se usa antes de estimar la **matriz esencial** con focal=1 y pp=(0,0).

> 💡 Mejora sugerida: renombrar a `pixels_to_normalized()` y corregir comentarios.

---

# `dataset.py`
**Propósito:** lectura de datos (imágenes o video) como iteradores Python.

- `DataLoader(path)`
  - Clase base: valida que `path` exista. Define interfaz con `__iter__` y `__next__` abstractos.

- `ImageLoader(path)`
  - Lista los archivos **.png** (ordenados) dentro de `path` y los mete en un `deque`.
  - **Asume** que hay imágenes y toma la **última** para obtener `height`, `width`.
  - `__next__`: hace `cv2.imread()` del siguiente PNG; levanta `StopIteration` cuando se acabó.
  - **Detalle:** sólo acepta `.png`. Si tu dataset es `.jpg`, no lo va a leer.

- `VideoLoader(path)`
  - Usa `cv2.VideoCapture` para leer MP4.
  - Expone `width` y `height` del video.
  - `__next__`: lee un frame; si no hay más, `StopIteration`. Convierte BGR→RGB.

> 💡 Mejora sugerida: permitir extensiones múltiples (`.jpg`, `.jpeg`) y ordenar por timestamp si hace falta.

---

# `display2d.py`
**Propósito:** mostrar el frame actual con matches dibujados.

- Inicializa `pygame`, crea `screen` y una `surface` del mismo tamaño.
- `paint(img)`: procesa eventos para que la ventana no se cuelgue, copia el array a la surface y hace `flip()`.

> Nota: espera **arrays RGB** con shape `(H, W, 3)`.

---

# `display3d.py`
**Propósito:** visor 3D en **otro proceso** (multiprocessing) que renderiza frustums y trayectoria.

- `VisualOdometryState`: contenedor ligero con `poses` (N×4×4) y `translations` (N×3).
- `Display.__init__()`
  - Crea una `Queue()` y un `Process` (`viewer_thread`) que corre el loop de render.
  - Esto evita que el render 3D **bloquee** el hilo principal.

- `viewer_thread(q)`
  - Configura ventana (1024×550), inicia loop de eventos y refresco a 60 FPS.
  - `viewer_refresh(q)`: consume el último estado de la cola, actualiza cámara 3D y dibuja.

- `viewer_init(w, h)`
  - Setea OpenGL: depth test, proyección `gluPerspective(45°, aspect, 0.1, 5000)`.
  - Define una cámara virtual (`gluLookAt`) con posición inicial (`camera_*`) y target `(0,0,0)`.

- `draw_camera_frustum(pose, scale)`
  - Dibuja un frustum de cámara aplicando la **pose 4×4** (usa `glMultMatrixf(pose.T.flatten())`).

- `draw_line_strip(points)`
  - Dibuja una **línea** conectando la secuencia de `translations`.

- `update(vo)`
  - El hilo principal empaqueta `vo.poses` y `vo.translations` en un `VisualOdometryState` y lo mete en la cola.

> 💡 Tips: no hay controles de usuario (rotar/orbitar). La cámara "persigue" la última pose moviendo `camera_x/y/z` en función de `current_pose[:3,3]`.

---

# `features.py`
**Propósito:** detección, descripción y **matching** de puntos de interés.

- `FeatureManager.__init__()` crea un extractor ORB.
- `detect(frame)`
  - Usa **Shi–Tomasi** (`goodFeaturesToTrack`) para detectar hasta 2000 esquinas.
  - Las convierte a `cv2.KeyPoint` (size=5) para que ORB pueda describirlas.
  - **Recrea** el extractor ORB cada vez (no hace falta, pero no duele).

- `compute(frame, kps)`
  - Llama `ORB.compute` para obtener descriptores binarios; guarda `self.kps`, `self.des`.

- `detect_and_compute(frame)` = detect + compute.

- `get_matches(cur_kps, ref_kps, cur_des, ref_des)`
  - `BFMatcher(NORM_HAMMING).knnMatch(ref_des, cur_des, k=2)` y **ratio test de Lowe** (0.75).
  - `assert len(good) > 8` para garantizar mínimo de pares para pose (si falla, se cae).
  - Devuelve `ref_pts` y `cur_pts` (arrays Nx2) **en pixeles** de los keypoints matcheados.

> 💡 Mejora sugerida: manejar el caso de pocos matches sin `assert` (p. ej. saltar frame o relajar umbral).

---

# `main.py`
**Propósito:** orquestación.

- `parse_cfg('config.yaml', dataset)` lee YAML y devuelve el bloque de ese dataset (`fx, fy, cx, cy, w, h, path`).
- Crea `Camera(w, h, fx, fy, cx, cy)`.
- Según `path` sea carpeta o archivo: `ImageLoader` o `VideoLoader`.
- `_main(camera, loader)`
  - Inicializa `Display2D` (w×h) y `Display3D`.
  - Crea `VisualOdometry(camera)`.
  - Prepara un lienzo 800×800 para el mapa 2D (`mapd`).
  - Loop sobre frames: `vo.process_frame(img, i)`.
  - Dibuja la imagen con matches (`display2D.paint(vo.draw_img)`).
  - A partir del frame 2, proyecta `vo.translations[-1]` a coordenadas del mapa 2D con `fit_point` y va pintando círculos + texto (x,y).
  - Llama `display3D.update(vo)` en cada iteración para refrescar el visor 3D.
  - Al final, `cv2.destroyAllWindows()`.

- `fit_point((tw, th), (x, y, z))`
  - Mapea **x y z** de la traducción a un lienzo 2D (ignora y). Centra en `w//2` y baja el origen vertical.

> 💡 Detalle: se hace `cv2.resize(img, (width, height))` para que el procesamiento use las mismas dims que `Camera`.

---

# `odometry.py`
**Propósito:** VO (Visual Odometry) monocular muy básica: estima **R** y **t** entre frames consecutivos y acumula.

- Estados principales:
  - `cur_img`, `ref_img`: imagen actual y de referencia (gris)
  - `cur_kps/des`, `ref_kps/des`: keypoints y descriptores ORB
  - `cur_matched_kps/ref_matched_kps`: puntos matcheados (pixeles)
  - `cur_R` (3×3), `cur_t` (3×1): rotación y traslación **acumuladas**
  - `translations`: lista de `t` acumuladas; `poses`: lista de matrices 4×4 (`pose_Rt`)

- `process_frame(img, frame_id)`
  1) Convierte a gris si hace falta.
  2) Detecta y describe features.
  3) Si es el primer frame (id=0): sólo prepara `draw_img`.
  4) Si no: hace matching contra el frame previo y llama a `estimate_pose()`.
     - Filtra `mask` de inliers que devuelve `recoverPose`.
     - Dibuja matches (`draw_features`).
     - **Actualiza** pose acumulada:
       - `cur_t = cur_t + scale * cur_R.dot(t)` → compone traslación en el marco actual (usa un **escala** fija `0.8`).
       - `cur_R = cur_R.dot(R)` → compone rotación.
     - Agrega a `translations` y `poses`.
  5) Actualiza referencias para el siguiente frame.

- `estimate_pose(use_fundamental_matrix=False)`
  - Por defecto usa **Matriz Esencial** `E` (no `F`), porque primero transforma pixeles a **coordenadas normalizadas** con `Camera.denormalize_pts`.
  - `cv2.findEssentialMat(cur, ref, focal=1, pp=(0,0), RANSAC, prob=0.999, threshold=0.003)`
  - `cv2.recoverPose(E, cur, ref, focal=1, pp=(0,0))` → devuelve R, t y máscara de inliers.
  - Convierte la máscara a índices y los retorna junto a (R, t).

- `draw_features(img)`
  - Dibuja en azul los puntos y en verde las líneas entre pares (ref→cur). Devuelve un RGB para el visor 2D.

> ⚠️ Limitaciones importantes:
> - **Escala desconocida:** Monocular VO recupera t **hasta un factor de escala**. Aquí usan `self.scale = 0.8` fija → la distancia absoluta es arbitraria.
> - **Fragilidad a pocos matches:** `assert len(good) > 8` puede tirar el programa; conviene manejarlo con `if`.
> - **Sin bundle adjustment / optimización global:** la deriva crecerá con el tiempo.

---

# `util.py`
- `add_ones(x)`: añade el 1 homogéneo (a un punto o a una lista de puntos).
- `pose_Rt(R, t)`: empaqueta a **matriz 4×4** con R en la esquina y t en la última columna.

---

# Cosas a tener en cuenta / Debug rápido
- **Config correcta:** `config.yaml` debe tener `fx, fy, cx, cy, w, h` consistentes con tu fuente (video o imágenes). Si descoordinan, los matches e inliers se vuelven pobres.
- **Extensiones:** `ImageLoader` sólo toma `.png`. Si tus datos son `.jpg`, renombra o ajusta el código.
- **Rendimiento:** ORB con 2000 features por frame está OK, pero puedes bajar a 1000 si tu CPU sufre.
- **Cierre del visor:** presioná `Esc` en la ventana 3D para salir limpio (verás “Quitting viewer…” en la terminal).

---

# Mejoras fáciles (si querés iterar)
1) **Nombres claros en `Camera`:** renombrar `denormalize_pts` a `pixels_to_normalized` y arreglar comentarios.
2) **Manejar pocos matches:** si `len(good) <= 8`, saltar actualización de pose y mantener la previa.
3) **Escala dinámica:** estimar escala comparando parallax promedio o usando datos de odometría externa si existiera; si es secuencia KITTI, podrías comparar con odometría ground-truth para calibrar un factor.
4) **Soporte `.jpg`:** extender el filtro en `ImageLoader`.
5) **Threshold adaptativo:** en `findEssentialMat` usar un umbral basado en el **ruido** esperado (p. ej., proporcional a 1/focal o al tamaño de imagen).

---

# Mini checklist para reproducir lo que ves
- `uv run main --dataset=mi_video` (tras agregar bloque en `config.yaml` con la ruta a tu MP4)
- Confirmá que el visor 2D muestra **líneas verdes** entre features → significa que el matching funcionó.
- En el 3D, la franja roja (pose actual) se desplaza; la azul (trayectoria) crece.

Si querés, en la próxima vuelta te dejo **comentado en línea** cada archivo con docstrings y `#` explicativos para que lo pegues directo en tu repo.

