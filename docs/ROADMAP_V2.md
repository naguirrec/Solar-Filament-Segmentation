# Solar Filament Segmentation 2026: hoja de ruta v2 (de 0,34 a 0,36–0,38)

Documento para Nacho y para el agente de código (Cursor). Fecha: 02/11/2026. Estado de partida: `main` @ `081994c`, LB público 0,34, holdout 0,409.
Cierre de la competición: 15/11/2026 a las 23:59 hora de Chicago (06:59 del 16/11 en Madrid, ver `AGENTS.md`). Objetivo interno de entrega: sábado 14/11.

Convención: este documento está en español para que Nacho lo pueda leer entero. El código, los nombres, los comentarios, los commits y las descripciones de PR van en inglés, como pide `AGENTS.md`.

Este documento complementa a `docs/PROJECT_REPORT.md` (qué se ha hecho), `docs/RESULTS.md` (resultados) y `docs/AUDIT_2026-10-28.md` (auditoría del 28/10, que sigue vigente en lo que no contradiga a este). Si algo de aquí choca con `AGENTS.md`, manda `AGENTS.md`.

---

## 0. Reglas inamovibles (de `AGENTS.md`, resumidas)

1. Todo cambio va en un PR contra `main`. Se fusiona cuando `./scripts/check.sh` pasa en CI **y** lo aprueba un agente distinto del autor. No se hace push directo a `main`.
2. Sin el OK explícito de Nacho **no** se hace: hacer público el repo, publicar en X o LinkedIn, borrar datos, tocar credenciales. Enviar a Kaggle sigue la regla de `AGENTS.md`.
3. No se hace commit de datos de MAGFiLO, de pesos de modelos ni de CSV de envío. En `results/` solo van CSV pequeños de resultados.
4. Nada de "hackear" la métrica. Las máscaras de una misma imagen deben ser disjuntas. No se usa el test para elegir hiperparámetros. No se usa MAGFiLO público ni el leak del test (hay pruebas públicas de que el 77 % del test coincide con imágenes públicas de MAGFiLO 1.0).
5. El agente **no tiene GPU ni los datos** (`data/` está fuera de git, el portátil tiene unos 4 GB de VRAM). Todo lo que necesite GPU o datos reales se entrega como notebook de Kaggle que ejecuta Nacho, con instrucciones exactas (qué importar, qué inputs añadir, qué descargar de Output). El código se escribe con la misma separación que ya usa el repo: lógica en numpy puro con tests sintéticos en CI, y torch solo en los módulos que CI se salta.
6. Cada experimento deja una fila en `docs/RESULTS.md` y, si procede, un CSV pequeño en `results/`. Lo que no se ha medido se marca `[VERIFY]` o `[UNKNOWN]`. No se inventan números.
7. Hay que mirar el hilo de dependencias: `pyproject.toml` hoy solo tiene `matplotlib`, `numpy`, `opencv-python-headless`, `pycocotools`, `scikit-image`. No se añade torch (viene en Kaggle). Añadir `scikit-learn` (BSD-3) está permitido, pero hay que regenerar `uv.lock` y `requirements.txt`. Cualquier dependencia copyleft (por ejemplo Ultralytics, AGPL-3.0) exige parar y preguntar a Nacho.

---

## 1. Resumen ejecutivo

**Diagnóstico.** Con una U-Net semántica más componentes conexas, los equipos públicos se quedan en 0,32–0,35 de LB. Vosotros estáis en 0,34. Es el techo de ese enfoque. Los que pasan de 0,36 añaden algo a nivel de **instancia**: un selector que decide qué componentes se emiten, o un detector de instancias.

**Lo que ya está hecho y no hay que repetir:** recortes a resolución nativa, r34 de 80 épocas, inferencia por teselas con solape 128, flip TTA de 4 vistas, umbral 0,45, cierre morfológico solo para agrupar (`close_size` 11, no cambia los píxeles), `min_area` 400, guarda de checkpoints, métrica agrupada por vista.

**Lo que falta:**
- No hay puntuación por instancia: se emite todo componente que supere `min_area`.
- La LR es constante (Adam 1e-4), sin scheduler ni EMA, y el objetivo es la unión simple de los anotadores.
- Las teselas se mezclan con media uniforme (el módulo `tiling.py` ya tiene `taper_window`, pero `predict_probability` no lo usa).
- No hay caché de probabilidades: cada barrido vuelve a pasar por la GPU.
- No hay intervalos de confianza ni análisis por número de anotadores.
- Hay un hueco holdout→LB de unos 0,07 que no está explicado (sección 3).

**Plan, por ganancia esperada sobre coste** (las ganancias son estimaciones mías a partir de lo que midieron otros equipos con pipelines distintos; no hay garantía):

| # | Mejora | Coste | Ganancia esperada | Prioridad |
|---|---|---|---|---|
| A | Infraestructura: caché de probabilidades, PQ por archivo, bootstrap, análisis por nº de vistas, paridad con la métrica oficial | 1–2 días, sobre todo CPU | Indirecta: decidir bien y explicar el hueco holdout→LB | Máxima |
| B1 | Filtro por objeto (probabilidad media o máxima del componente) | Horas, CPU | +0,005 a +0,015 | Alta |
| B2 | Selector de componentes (GBM) con regla de emisión p > PQ/2 | 1–1,5 días, CPU | +0,015 a +0,025 en val; en LB un equipo pasó de 0,32 a 0,36 | Alta, con cautela |
| C | Inferencia: ventana cónica o Hann, solape 256, D4 (8 vistas), pasada única a 2048, ablación de anchura ±1 px | Horas de GPU | +0,003 a +0,01 | Media |
| D | Entrenamiento v2: AdamW + coseno + EMA, pesos por acuerdo | 2 runs de unas 3 h en T4 | +0,005 a +0,015, y sobre todo segundo miembro bueno del ensemble | Alta (hay que lanzarlo pronto) |
| E | Ensemble de r34-80ep con el run D y con el r50/effb3 que se está entrenando | Solo inferencia | +0,005 a +0,01 | Media |
| F | Detector de instancias + U-Net (RF-DETR-Seg o Mask R-CNN) | 2–3 días + 4–8 h de GPU | +0,02 a +0,05 (LB 0,37–0,41) | Decisión el 04/11 |
| G | Entrega: pesos en el repo, notebook de reproducción, `requirements.txt` completo, tabla de eficiencia, informe | 3–4 días | Es el 30 % cualitativo de la nota | Obligatoria |

**Objetivo realista.** Sin detector (A–E): 0,35–0,37 de LB. Con detector (A–F): 0,37–0,40. Con 2 decimales en el LB y un ruido de unas ±0,01, no hay que fiarse de mejoras de menos de 0,01.

**Lo que no hay que hacer** (evidencia pública de que empeora o no ayuda): CLAHE, copy-paste, pseudo-etiquetas con el test, SAM o SAM2 zero-shot, Mask2Former con pocos datos, watershed, clDice en el refinador, soft labels puras, rejillas finas en dos ejes sobre una validación pequeña, elegir la "mejor época", reentrenar con train+holdout si se usa un selector ajustado en el holdout. Detalle y fuentes en la sección 9.

---

## 2. Punto de partida (verificado en el repo)

**Datos.** JSON de entrenamiento tipo COCO con 1154 entradas de imagen y 8199 anotaciones (polígonos). No hay carpeta de máscaras. Hay `file_name` repetidos: son registros de distintos anotadores de la misma observación. El repo los colapsa a una muestra por archivo (565 de train y 142 de holdout) y conserva cada registro original como una "vista" (`TrainImage.views`; vacío si el archivo tiene una sola). 180 imágenes de test [VERIFICAR].

**Validación local.** El último 20 % de fechas de calendario (`filament.split.split_by_date`): 142 archivos y 231 vistas de anotador (1,63 vistas por archivo). PQ agrupado: se suman TP, FP, FN y sumas de IoU de todas las vistas y se calcula un único PQ (`filament.metrics.fast_view_counts`, `PqCounts`). Cuenta como TP todo par con IoU > 0,5 (no emparejamiento uno a uno). Que coincida con el notebook oficial **no está verificado** (`[VERIFICAR]` en los PR #9, #10 y #23).

**Pipeline actual** (`filament.crop_training`, `filament.tiling`, `filament.ensemble`, `filament.instances`):
1. U-Net `smp` con encoder `resnet34` (ImageNet), 3 canales (gris repetido), entrenada en recortes de 512 px de la imagen a 2048 px, muestreo sesgado a primer plano (`p_fg` 0,67), aumentos D4 (flips y rotaciones de 90°), pérdida Dice + BCE, Adam 1e-4 constante, AMP, 80 épocas, batch 8, solo se guarda la última época.
2. Inferencia: 25 teselas de 512 con solape 128 (stride 384), media uniforme, flip TTA de 4 vistas (`FLIP_NAMES = none, h, v, hv`).
3. Umbral → puerta de disco solar (`baseline.on_disk`) → componentes conexas (con cierre elíptico solo para decidir etiquetas) → se descartan componentes de menos de `min_area` → RLE por componente.

**Resultados** (PQ agrupado en el holdout; LB = leaderboard público, unos 90 de las 180 imágenes, 2 decimales):

| Modelo / configuración | Holdout | LB |
|---|---|---|
| r18, 512 px imagen entera, 12 épocas, `min_area` 400 | 0,2886 | 0,23 |
| r34 recortes nativos, 40 épocas, 0,5 / 0 / 400 | 0,3675 | 0,32 |
| r34 40 épocas, 0,45 / 15 / 400 | [UNKNOWN] | 0,33 |
| lo anterior + TTA de 4 flips | 0,403 | 0,34 |
| r34 80 épocas, 0,5 / 0 / 400, sin TTA | 0,3831 | no enviado |
| r34 80 épocas + TTA, 0,45 / 11 / 400 | 0,409 | 0,34 |
| ensemble r18 + r34 | por debajo de r34 solo [UNKNOWN] | no enviado |

Las filas de 0,33, 0,34 (TTA) y 0,409 vienen de notas de Nacho y no están en `docs/RESULTS.md`, y los CSV que eligieron esos ajustes (`holdout_postproc_r34.csv`, `holdout_tta_ensemble.csv`) no están commiteados. Hay que registrarlos (sección 7, fase G).

**Qué se está entrenando ahora en Kaggle.** [VERIFICAR con Nacho]. Lo más probable es el `r50` o el `effb3` del PR #42 (ese PR añade `MODEL = "r50"` con 120 épocas y `"effb3"` con 100). Sus tiempos (4,7–8,0 h para r50) son estimaciones, no medidas. Como las épocas difieren de las del r34, una ganancia no se puede atribuir solo al encoder.

---

## 3. Diagnóstico: por qué 0,34 y qué es el hueco holdout→LB

El holdout supera al LB en unos 0,05–0,07 en todos los pares. Cocientes LB/holdout: r18 0,80; r34-40ep 0,87; con TTA 0,84; r34-80ep + TTA 0,83. Es decir, el hueco es aproximadamente proporcional, no aditivo, y no crece con el coste de inferencia (el r18 es el más rápido y tiene el cociente más bajo). Eso apunta a un efecto **sistemático** (el test es "más difícil" para esta métrica) más que a un castigo por eficiencia. Otros equipos tienen huecos de 0,05–0,08 (biconcavelens, Solid-Snake-99) o casi nulos (rodchemist: 0,342→0,34 y 0,359→0,35). Hay tres hipótesis. Ninguna está demostrada; cada una se puede comprobar barato.

**H1. Sobreajuste del barrido al holdout.** Se han comparado más de cien configuraciones (umbral × cierre × `min_area` × TTA) sobre 142 archivos. Otro equipo (MPkS1) midió que ajustar y evaluar sobre los mismos fotogramas infla el PQ unos 0,06 (0,424 frente a 0,363 en 16 fotogramas disjuntos). Un indicio a favor en vuestros datos: el paso de 0,32 a 0,34 en LB (+0,02) se corresponde con +0,035 en holdout (0,3675→0,403), es decir, el LB recoge poco más de la mitad de la mejora. El LB con 2 decimales no permite concluir. *Test:* validación cruzada por fechas dentro del holdout (sección 5, fase A).

**H2. Mezcla de anotadores distinta en el test.** El PQ agrupado pesa cada archivo por su número de vistas, y donde los anotadores discrepan el PQ alcanzable es menor (entre dos anotadores ronda 0,33–0,34, medido por rodchemist y por ameyypawar/biconcavelens en la auditoría). El holdout tiene 1,63 vistas por archivo; kaggle.too.foo estima unas 2,44 en el test (**dato de una sola fuente, no verificado**). *Test:* calcular el PQ del holdout separado por archivos con 1 vista y con ≥2, y reponderado a 2,4 vistas por archivo. Si el subconjunto multivista da cerca de 0,34, el hueco queda explicado y el LB no es "peor", es otra mezcla.

**H3. La métrica local no es la oficial.** El notebook oficial de autoevaluación (`azimahmadzadeh/self-evaluation-notebook`) nunca se ha descargado ni comparado. Además, los organizadores dicen que la nota tiene en cuenta la eficiencia computacional, y kaggle.too.foo escribe que "el leaderboard cambió de métrica a mitad de la competición" [VERIFICAR en los anuncios de Kaggle, no lo he podido leer]. *Test:* test de paridad con el código oficial (fase A0).

**Qué NO explica el hueco:** los cocientes constantes (arriba) descartan que el TTA o la inferencia lenta se castiguen de forma notable, aunque el término de eficiencia sigue sin estar confirmado.

---

## 4. Matemática que guía la regla de emisión

Sea un candidato `c` (un componente disjunto del resto) en una observación con `R` vistas de anotador. Para cada vista `r`, si `c` empareja con un GT sin emparejar: TP +1 y FN −1, el denominador sube +0,5 y el numerador sube `IoU_r`. Si no empareja: FP +1, el denominador sube +0,5 y el numerador no cambia. Como las máscaras son disjuntas, dos predicciones no pueden tener ambas IoU > 0,5 con el mismo GT, así que el cálculo es exacto. En total el denominador sube `0,5·R` pase lo que pase.

Por tanto **emitir `c` mejora el PQ agrupado si y solo si**

```
(1/R) · Σ_r  IoU_r(c) · 1[IoU_r(c) > 0.5]   >   PQ_actual / 2
```

Con PQ ≈ 0,34 y una IoU media de los aciertos de ≈ 0,68, un candidato compensa si su probabilidad de acertar con un anotador cualquiera supera ≈ 0,25 (con PQ 0,40, ≈ 0,29). Kaggle.too.foo llega a la misma regla (umbral ≈ 0,32) y le atribuye +0,025 en validación.

Consecuencias:
- **Partir un filamento sale carísimo:** dos trozos con IoU < 0,5 cada uno cuestan 2 FP y 1 FN (+1,5 en el denominador) en vez de un TP. Agrupar fragmentos es prioritario. El repo ya agrupa con `close_size`. El oráculo de fusión de kaggle.too.foo valía +0,026 en su pipeline, así que el margen restante en agrupación es pequeño.
- **Un casi acierto (IoU 0,3–0,5) cuesta un FP y un FN.** Por eso el grosor de la máscara importa: dilatar o erosionar 1 px sobre una máscara buena cuesta del orden de 0,03 (según la auditoría del 28/10, citando a biconcavelens; no lo he re-verificado).
- **Los filamentos que marca solo 1 de 3 anotadores casi nunca compensan.** El objetivo efectivo se parece al consenso, no a la unión.
- Cuanto mayor es el PQ, más exigente es el listón de emisión (`PQ/2`): el selector se recalcula iterando (Dinkelbach).

---

## 5. Plan de trabajo, fase a fase

Leyenda: **[Agente]** lo hace el agente de código (CPU, tests sintéticos). **[Nacho]** lo ejecuta Nacho en Kaggle u ordenador. Un PR por paso. Todo test usa datos sintéticos.

### Fase A: infraestructura de decisión (días 1–2)

**A0. Paridad con la métrica oficial. [Nacho + Agente]**
- Nacho: descargar `azimahmadzadeh/self-evaluation-notebook` (con sus credenciales de Kaggle; el agente no las toca) y dejarlo en `third_party/self_eval/` (si la licencia no está clara, añadirlo a `.gitignore` y no commitearlo) [VERIFICAR licencia].
- Agente: portar su lógica de emparejamiento a una referencia (`src/filament/metrics_official.py` o un test que la use) y escribir `tests/test_metric_parity.py` que compare TP, FP, FN y suma de IoU con `fast_view_counts` sobre máscaras sintéticas con solapes, vacíos y duplicados. Si el oficial cuenta distinto (por ejemplo, emparejamiento uno a uno, o trata los vacíos de otra forma), se adapta `fast_view_counts` y se documenta en `docs/RESULTS.md`.
- Aceptación: test verde, y una nota en `docs/RESULTS.md` con "paridad confirmada" o "diferencia encontrada y corregida". Si hay un término de eficiencia, anotar su fórmula.

**A1. Caché de probabilidades. [Agente]** Módulo `src/filament/probcache.py` (numpy puro) con `save_prob_map` / `load_prob_map` en float16, un `.npy` por archivo y modelo. Se guardan dos mapas por archivo y modelo: la **media** de los 4 flips y la **desviación típica** entre flips (incertidumbre por TTA, útil como variable del selector). Tamaño: 142 × 2048² × 2 B ≈ 1,2 GB por mapa en el holdout (≈ 2,4 GB con media y std), unos 3 GB para el test. Cabe en el Output de Kaggle. Los directorios van en `outputs/` (ya ignorado por git).

**A2. Notebook de exportación. [Agente escribe, Nacho ejecuta]** `notebooks/export_probs.ipynb`: carga un checkpoint con la guarda `filament.checkpoint`, predice con teselas + flips holdout y test, escribe la caché (A1) y un `manifest.json` (hash del checkpoint, config de inferencia, segundos por imagen). Nacho lo ejecuta una vez por modelo (r34-80ep ahora, otros más adelante) y sube la salida como dataset privado de Kaggle para los notebooks de CPU. Esto elimina el pase por GPU en los barridos de las fases A–B.

**A3. PQ por archivo. [Agente]** Una función en `filament.sweep` (o módulo nuevo) que devuelva `PqCounts` **por archivo** para una configuración de postproceso, sobre mapas cacheados. Además, número de vistas por archivo (`len(views) or 1`).

**A4. Estadística. [Agente]** `src/filament/stats.py` (numpy puro):
- `bootstrap_pq(counts_by_file, n=2000, seed)` → media e IC 95 % remuestreando archivos.
- `paired_bootstrap(a, b)` → diferencia media, IC y fracción de remuestreos con b > a.
- `weighted_pq(counts_by_file, weights)` → PQ = Σ w·iou / Σ w·(tp + 0,5·fp + 0,5·fn), para reponderar a otra mezcla de vistas.
- `crossfit_select(grid_counts, folds)` → elige la configuración en una parte de las fechas y la evalúa en la otra, y reporta la media de las evaluaciones (no el máximo del barrido).
Con tests sintéticos (por ejemplo, un caso donde el pico de un barrido es ruido y el cross-fitting lo detecta).

**A5. Análisis H1 y H2. [Agente + Nacho]** Con la caché: (a) PQ del holdout por estratos de 1 vista y ≥2 vistas, y reponderado a 2,4 vistas por archivo; (b) PQ con la configuración actual elegida por cross-fitting por fechas (la diferencia con 0,409 estima el sobreajuste); (c) IC bootstrap de la configuración actual. Escribir las tres cosas en `docs/RESULTS.md`. Aceptación: se puede afirmar con números cuánto del hueco explica H1 y cuánto H2.

### Fase B: puntuar instancias (días 2–5)

**B1. Filtro por objeto (el más barato). [Agente]** Añadir a `filament.instances` (o a un módulo nuevo `filament.components`) un parámetro `min_object_prob`: se descartan los componentes cuya probabilidad media (o máxima, o percentil 90) en la caché es menor que el umbral. Se barre con cross-fitting junto a `min_area`, **un eje cada vez** (descenso por coordenadas), no una rejilla fina de dos ejes. También se amplía el barrido de umbral de píxel por encima de 0,60 (MPkS1 acabó en 0,88 con otra pérdida), pero solo con cross-fitting.

**B2. Selector de componentes. [Agente]** `src/filament/components.py` y `src/filament/selector.py`:
- *Candidatos:* componentes de la segmentación **permisiva** (umbral bajo y `min_area` bajo), no solo los que ya pasan el filtro actual.
- *Variables por candidato:* área; longitud del esqueleto (`skimage.morphology.skeletonize`) y anchura media = área / longitud; elongación y solidez (`skimage.measure.regionprops`); media, máximo y percentil 90 de la probabilidad; media de la desviación entre flips (incertidumbre TTA); distancia radial del centroide al limbo y si toca el borde del disco; contraste = mediana de gris dentro de la máscara frente a un anillo de 5–15 px alrededor.
- *Etiqueta:* `y_c = (1/R) Σ_r IoU_r(c) · 1[IoU_r(c) > 0,5]`. Se necesita `component_view_ious(labels, view_masks)` → matriz (candidatos × vistas), con la misma lógica de `bincount` que `fast_view_counts`. Test: la suma de IoU de los emparejados debe coincidir con `PqCounts` de `fast_view_counts`.
- *Modelo:* `sklearn.ensemble.HistGradientBoostingRegressor` (añadir `scikit-learn` a `pyproject.toml`, regenerar `uv.lock` y `requirements.txt`, avisar a Nacho). Empezar por una regresión logística o un árbol poco profundo como línea base.
- *Regla:* emitir `c` si `ŷ_c > λ/2`, con λ = PQ agrupado actual, iterando hasta converger (Dinkelbach, 3–5 iteraciones).
- *Evaluación:* `GroupKFold` agrupando **por fecha** sobre los 142 archivos, predicciones fuera de pliegue, repetido con 5 particiones distintas. **Solo se adopta si gana en al menos 4 de 5 particiones y el bootstrap pareado da una mejora ≥ +0,005 en ≥ 90 % de los remuestreos.** Riesgo conocido: el reescorer de biconcavelens perdió en 4 de 5 semillas, el selector de kaggle.too.foo ganó (+0,018 a +0,025 en val; de 0,32 a 0,36 en LB, aunque su página es inconsistente sobre su mejor LB y menciona un cambio de métrica a mitad de competición). Con solo ≈ 142 archivos hay del orden de 1000–1500 candidatos (estimación mía): un GBM pequeño es viable, uno grande sobreajusta.
- **Importante:** el selector se ajusta con los mapas del holdout, que el modelo no ha visto en entrenamiento. Para aplicarlo al test hay que usar **exactamente** el mismo checkpoint, TTA y postproceso. No se reentrena la red con train+holdout después (cambiaría la calibración de las probabilidades; además un equipo midió peor LB entrenando con el 100 % de los datos: 0,37→0,36).

### Fase C: inferencia (días 3–6, GPU, barato)

Notebook de Kaggle `notebooks/sweep_inference.ipynb` que evalúa variantes sobre el holdout con un postproceso **fijo** (para no sobreajustar) y mide segundos por imagen:
1. Ventana de mezcla de teselas: uniforme (actual), `taper_window(512, ramp)` (ya existe, falta pasarla desde `predict_probability`), Hann.
2. Solape 128 (actual) frente a 256 (stride 256). Ojo: el +0,03 de kaggle.too.foo mezcla pasar de "sin solape" a solape, agregación por máximo y un umbral reajustado; con vuestra base (ya hay solape y media) la ganancia esperable es mucho menor.
3. TTA D4 de 8 vistas (hay que añadir las 4 transposiciones/rotaciones a `filament.ensemble`; hoy `FLIP_NAMES` solo tiene los 4 flips).
4. Pasada única a 2048 sin teselas (una CNN lo permite; en una T4 con fp16 debería caber, hay que comprobarlo) para eliminar costuras.
5. Ablación de anchura: dilatar y erosionar 1 px las máscaras finales.
Se reportan PQ, IC bootstrap y s/imagen. Se adopta una variante solo si gana de forma consistente en los estratos de la fase A.

### Fase D: entrenamiento v2 (lanzar cuanto antes; días 1–3)

Es lo de mayor plazo de espera (≈ 3 h de T4 por run), así que el código de D va en el **primer PR**.

- **D1 (un cambio de receta, mismo dato y pérdida):** AdamW (lr 1e-4–3e-4, wd 1e-4), coseno con 1 época de calentamiento, EMA de pesos (decay 0,999), gradient clipping a 1,0, **checkpoint final = EMA de la última época** (no la "mejor"). Mantener 80 épocas para comparar con 0,3831 (sin TTA, mismo postproceso). Motivo: de 40 a 80 épocas se ganó +0,016 con LR constante, señal de que falta calendario; la LR constante deja un último ruido sin enfriar.
- **D2 (D1 + pesos por acuerdo):** el objetivo sigue siendo la unión, pero la pérdida BCE pesa más los píxeles marcados por más anotadores. Propuesta de fórmula (nuestra, no sacada de ningún repo): `w = 0,4 + 0,6·k/R` en positivos, `1` en negativos, con `k` = nº de vistas que cubren el píxel y `R` = nº de vistas del archivo. Implementación: sustituir la máscara de unión de `TrainCache` por un mapa uint8 de acuerdo `k` (la unión es `k > 0`, así que no cuesta memoria extra) y una pérdida con pesos por píxel (el `DiceLoss` de `smp` no admite pesos: aplicarlos a la BCE y dejar el Dice sin pesos). Evidencia: kaggle.too.foo, soft labels puras 0,323 frente a unión con pesos 0,377 (y 0,368 con la unión simple en otra configuración, así que la comparación no es limpia). No usar soft labels puras ni consenso (0,301).
- **D3 (opcional, baja prioridad):** Skeleton Recall Loss (Kirchhoff et al., ECCV 2024; +8 % de tiempo y +2 % de VRAM; esqueleto de la máscara dilatado con kernel de diamante de radio 2, calculado en CPU y precomputable). Los resultados públicos en esta competición son marginales y dentro del ruido (kaggle.too.foo: mejor de cuatro brazos con 0,343, pero con seis variantes dentro de ±0,02). Solo si sobra tiempo.
- **Semillas y varianza.** No hay medida de la varianza entre runs. Cualquier diferencia < 0,01 entre D1, D2 y el r34-80ep actual no es concluyente: se registra, pero la decisión de usar D1/D2 en el ensemble se toma por el bootstrap pareado, no por el valor puntual.
- **Registro:** pinear el hash del checkpoint de cada run en `filament.checkpoint` (como en #38 y #41) y añadir fila a `docs/RESULTS.md`.

### Fase E: ensemble (días 6–8)

Con la caché de la fase A2, combinar mapas de: r34-80ep, el run D (D1 o D2) y el modelo que se está entrenando ahora (r50/effb3, si mejora). Probar solo **pesos iguales** frente al mejor modelo individual (no barrer pesos: sobreajusta). El ensemble r18+r34 empeoró, pero el r18 era un modelo de 12 épocas a 512 px imagen entera con mapa reescalado, así que no es evidencia en contra de ensamblar dos modelos buenos por recortes. Se mide el coste: cada modelo extra multiplica el tiempo de inferencia (cuenta para la eficiencia).

### Fase F: detector de instancias + U-Net (decisión el 04/11)

**Qué aporta.** La U-Net sabe qué píxeles son filamento, pero no qué píxeles van juntos ni con qué confianza. Un detector da ambas cosas. Evidencia (con las salvedades de cada caso):
- biconcavelens: U-Net semántica + componentes 0,32 de LB; detectores calibrados + dedup entre ellos, 0,39 (su U-Net era más débil que la vuestra).
- Solid-Snake-99: RF-DETR-Seg-L en 4 cuadrantes reescalados a 768 y un refinador U-Net r18 sobre recortes de 256 px por caja, LB 0,410 (val 0,4871; hueco 0,077). El refinador está acreditado a trabajo público de Fususu y Nomannic19 en Kaggle.
- ranjeet258: tres U-Nets + YOLO11m + selector LightGBM, 0,39 (el repo también menciona variantes de 0,55 que incrustan un CSV de otra fuente: no son comparables ni válidas).
- Lo que **no** he podido verificar hoy: las cifras de ameyypawar (LB 0,37 con YOLO "reclamando" píxeles de la U-Net) vienen de la auditoría del 28/10; la página del repo que leí hoy mostraba una fase anterior (baseline 0,08). Hay que tomarlas como no confirmadas.

**Variante mínima recomendada ("el detector agrupa y puntúa, la U-Net pone los píxeles"):**
1. Exportar un ejemplo por vista de anotador (polígonos → formato del detector), con **el mismo split por fecha**.
2. Entrenar el detector en Kaggle (Nacho). Opciones: RF-DETR-Seg (`rfdetr`, **Apache 2.0** según el README de Roboflow para todos los tamaños de segmentación; resoluciones nativas de 312 a 768 px, así que por sí solo pierde barbs a 2048 px: de ahí que la U-Net ponga los píxeles) o Mask R-CNN de torchvision (BSD). YOLO11-seg es AGPL-3.0: parar y preguntar a Nacho antes de usarlo.
3. Fusión: ordenar las detecciones por puntuación; cada una "reclama" los píxeles de la máscara U-Net (umbral ~0,45–0,5) que caen dentro de su máscara dilatada y no reclamados aún; las máscaras salen disjuntas por construcción. Por defecto **no** convertir en instancias los trozos de U-Net sin reclamar (suele inundar de FP).
4. Calibrar la puntuación del detector con regresión isotónica ajustada con cross-fitting en el holdout, y deduplicar entre detectores con IoU 0,05 si hay más de uno (biconcavelens: la meseta es plana de 0,03 a 0,1 y se hunde por debajo de 0,01).
5. Evaluar con el protocolo de la sección 6.

**Criterio de decisión (04/11, con la fase A hecha):** ir a por F solo si (i) quedan ≥ 8 días y (ii) Nacho tiene ≥ 6–8 h de GPU libres esta semana [VERIFICAR cuota: la auditoría cita 30 h/semana de biconcavelens]. Si no, se hace A–E y se escribe el informe con la mejor configuración. Riesgo principal: biconcavelens necesitó decenas de experimentos para pasar de 0,37 a 0,39, y su ensemble mejoró en local más de lo que mejoró en LB.

**Resultado (07/11):** la variante mínima con RF-DETR-Seg L no se adopta. El mejor ajuste da 0,3603 frente a 0,4255 de la base en el holdout (-0,0651, IC 95 % -0,0900 a -0,0411) y no pasa ninguna de las 5 particiones. Detalle en `docs/RESULTS.md`. En el ajuste por defecto (0,3555), la pérdida está en RQ (0,6417 a 0,5415) con SQ casi igual, sin separar qué parte viene de partir y qué parte de descartar. La segunda regla, en la que el detector solo une componentes, tampoco se adopta: su mejor ajuste da 0,4171 (-0,0084) y no pasa ninguna partición. Se cierra la fase F y se pasa a la G.

### Fase G: entrega y higiene (días 9–13)

Lista de `AGENTS.md` que sigue pendiente:
1. **Pesos en el repo** (los exige la entrega): `state_dict` en fp16, cada uno < 100 MB (un r34 ocupa unos 49 MB; un r50, unos 65 MB), con `weights/MANIFEST.json` y el SHA-256 de cada uno, copia de respaldo en un GitHub Release. **Hacerlo y hacer público el repo requieren el OK de Nacho.**
2. **`notebooks/reproduce.ipynb`:** carga los pesos del repo, procesa el test, escribe `submission.csv`, lo valida con `filament.validate`, re-puntúa el holdout, muestra 6 superposiciones y funciona en Kaggle (T4) y en local.
3. **`requirements.txt` completo y con versiones.** Hoy solo tiene las dependencias de CPU (sin torch, smp ni, en su caso, scikit-learn ni rfdetr).
4. **Tabla de eficiencia:** segundos por imagen en T4 (y CPU si es viable) para tres variantes: "full" (todos los modelos + D4), "standard" (flips + 1 modelo o el ensemble elegido) y "lite" (1 modelo, sin TTA), cada una con su PQ de holdout. Cuenta para la nota. Hoy solo se mide el tiempo por época de entrenamiento.
5. **Informe de 4 páginas** (plantilla Overleaf): pipeline, datos y protocolo agrupado (con la derivación de la sección 4, que es un aporte propio), ablaciones con IC, SQ/RQ, distribuciones de IoU y Dice, relaciones uno a muchos y muchos a uno, análisis de errores (casi aciertos, desacuerdo entre anotadores con el techo humano de ≈ 0,34), coste computacional, limitaciones. Reutilizable para el workshop SABID [PENDIENTE decidir].
6. **RESULTS.md al día:** añadir las filas de 0,33, 0,34 y 0,409 con sus CSV (Nacho debe aportar `holdout_postproc_r34.csv` y `holdout_tta_ensemble.csv`; si no los tiene, dejar `[UNKNOWN]`).
7. **Pendientes de revisión conocidos** (de `PROJECT_REPORT.md`, sección 6): `infer_unet_crops.ipynb` solo avisa si no hay GPU (debe parar: la eficiencia se mide en GPU); `jitter` sin validar contra `crop_size`; último lote de tamaño 1 sin excluir; NaN en puntuaciones pasa en silencio (#34).
8. **Lista de comprobación antes de publicar** (cada punto con OK de Nacho): sin datos en el historial de git (`git log --all -- '*.jpg' '*.jpeg' '*.npy'`), repo público, formulario, selección de los 2 envíos finales en Kaggle [VERIFICAR si hay que elegirlos], atribución "GONG data, NSO/AURA/NSF".

---

## 6. Protocolo experimental

1. **Un cambio por run.** Cada run deja config, hash del checkpoint, commit, métricas y una fila en `docs/RESULTS.md` con PQ, IC 95 % y, si se envió, el LB.
2. **Adoptar un cambio** solo si el bootstrap pareado por archivo da diferencia > 0 en ≥ 90 % de los remuestreos **y** la mejora es ≥ +0,005 en validación. **Enviar a Kaggle** solo con ≥ +0,01 en validación o un cambio estructural (selector, detector). Sin el OK de Nacho, solo se envía con más de +0,01 (regla de `AGENTS.md`).
3. **No se ajusta nada con el LB** (2 decimales, ~90 imágenes: biconcavelens vio seis envíos distintos dar exactamente 0,39).
4. **Nada de rejillas finas de dos ejes** sobre 142 archivos. Descenso por coordenadas, elegir el **centro de una meseta** (no el pico) y reportar el valor de cross-fitting, no el máximo.
5. **Nunca se toca el test para elegir hiperparámetros** (ni contando componentes: así se eligió el `min_area` 2000 original).
6. Los números de validación del holdout están sesgados al alza por haber elegido ahí el checkpoint, el umbral, el cierre y `min_area`. Se dice en el informe.
7. Seeds y versiones fijadas. Estimar y reportar la varianza entre runs cuando haya dos con la misma receta.

---

## 7. Calendario propuesto

Hoy es lunes 02/11 por la noche. Las fechas son orientativas; los horarios de GPU dependen de la cuota de Kaggle [VERIFICAR].

| Fecha | Agente (CPU) | Nacho (Kaggle / decisiones) | Hito |
|---|---|---|---|
| Mar 03/11 | PR1: entrenamiento v2 (D1 y D2). PR2: caché y `export_probs` (A1–A2). PR3: `stats.py` y PQ por archivo (A3–A4) | Descarga el notebook oficial (A0). Lanza D1 (≈ 3 h). Ejecuta `export_probs` con r34-80ep | Caché de r34-80ep disponible |
| Mié 04/11 | A0 (paridad), A5 (H1, H2), B1 (filtro por objeto) | Revisa A5. **Decisión F: sí o no** | Sabemos cuánto del hueco es H1/H2 |
| Jue 05/11 – Vie 06/11 | B2 (selector) con cross-fitting. Si F=sí: exportación del detector y fusión | Lanza D2. Si F=sí, entrena el detector | Selector evaluado |
| Sáb 07/11 – Dom 08/11 | Fase C (notebook de inferencia) y E (ensemble) | Ejecuta `sweep_inference` y `export_probs` de los modelos nuevos | Mejor configuración candidata |
| Lun 09/11 | Congelar configuración en un archivo de configuración | **Segundo envío serio a Kaggle (OK de Nacho)** | Configuración congelada |
| Mar 10/11 – Mié 11/11 | G: pesos, `reproduce.ipynb`, `requirements.txt`, eficiencia | Sube pesos al repo (OK) | El notebook reproduce el CSV |
| Jue 12/11 – Vie 13/11 | Figuras y tablas del informe | Informe en Overleaf | PDF en borrador |
| Sáb 14/11 | Revisión final | Repo público (OK), formulario, selección de envíos | **Entrega** |
| Dom 15/11 | Colchón | | El cierre real es el 16/11 a las 06:59 en Madrid |

---

## 8. Qué se espera de cada resultado (para no hacerse ilusiones)

| Situación tras A–E | Qué significa |
|---|---|
| H2 explica casi todo el hueco | El LB de 0,34 es consistente con el holdout de 0,41 en otra mezcla de anotadores. Mejorar el holdout seguirá mejorando el LB. |
| H1 explica una parte | Parte de los 0,409 era ruido de selección. Las ganancias de postproceso pequeñas (<0,01) no son fiables. Hay que usar la configuración de cross-fitting. |
| H3 (la métrica difiere) | Se corrige `fast_view_counts` y se re-evalúa todo. Podría cambiar el orden de configuraciones. |
| Selector no gana en 4 de 5 particiones | Se queda el filtro por objeto de B1, que es más simple. |
| D1/D2 y el ensemble suben < 0,01 | El techo del enfoque semántico está cerca. Solo F queda como palanca grande. |

Techo humano de referencia: un anotador puntuado contra otro da PQ ≈ 0,33–0,34. Un modelo por encima de eso no es "mejor que los humanos": indica que las etiquetas son ruidosas. Los organizadores dicen que PQ > 0,35 tiene mucho valor y que no se persigue el primer puesto a costa de la honestidad.

---

## 9. Qué no hacer (con evidencia)

| Idea | Evidencia | Fuente |
|---|---|---|
| CLAHE | 0,4252 → 0,4193 (clip 2) y 0,4121 (clip 4) en local | biconcavelens |
| Pseudo-etiquetas con imágenes de test | 0,4171 → 0,3931 | biconcavelens |
| SAM2 automático | 0,08 de LB | biconcavelens |
| Mask2Former con unas 1000 imágenes | local 0,373, LB 0,28 | biconcavelens |
| Watershed | local 0,236, LB 0,20 | biconcavelens |
| clDice en el refinador | 0,4252 → 0,4192 | biconcavelens |
| Rescate de candidatos de baja puntuación | PQ 0,418 → 0,21 en un intento | biconcavelens |
| Soft labels puras de los anotadores | 0,323 frente a 0,377 con unión ponderada | kaggle.too.foo |
| Consenso como objetivo | 0,301 | kaggle.too.foo |
| Contexto de 768 px | 0,323, por debajo de la base | kaggle.too.foo |
| Copy-paste | empeora (0,4152 → 0,3998, según la auditoría del 28/10, no re-verificado hoy) | auditoría |
| Elegir la "mejor época" en validación | peor fuera de muestra (0,411 frente a 0,418, según la auditoría, no re-verificado hoy) | auditoría |
| Entrenar con el 100 % de los datos y ajustar un selector en el holdout | incoherente; además un equipo midió 0,37 → 0,36 | biconcavelens |
| Rejillas finas en dos ejes en 116–142 imágenes | ganancias locales que empatan o empeoran en LB | biconcavelens |
| Usar MAGFiLO público o el leak del test, o CSV ajenos | contrario a las reglas, riesgo de descalificación | anuncios / varios repos |
| Máscaras solapadas o duplicadas | infla los TP: es "hackear" | análisis de la métrica |
| Apertura morfológica | rompe barbs | morfología básica |

---

## 10. Pendientes y cosas por verificar

- **[VERIFICAR]** Qué modelo se está entrenando ahora en Kaggle y con qué épocas.
- **[VERIFICAR]** Paridad de `fast_view_counts` con el notebook oficial; si hay término de eficiencia y cuál; si la métrica cambió durante la competición (anuncios de Kaggle; yo no he podido leer el foro ni los anuncios).
- **[VERIFICAR]** Nº de vistas por imagen en el test (2,44 es una sola fuente) y si hay `file_name` repetidos en el test.
- **[VERIFICAR]** Cuota de GPU de Kaggle disponible esta semana y la próxima.
- **[VERIFICAR]** Si hay que seleccionar 2 envíos finales y el límite diario de envíos.
- **[VERIFICAR]** Si usar las espinas (`spine`) como supervisión auxiliar cuenta como "otros metadatos de ground truth" (biconcavelens las usa; la regla de los organizadores prohíbe entrenar con otros metadatos de ground truth). Mientras tanto no se usan.
- **[PENDIENTE de Nacho]** Aceptar o no una dependencia AGPL (Ultralytics). RF-DETR-Seg (Apache 2.0) y Mask R-CNN de torchvision (BSD) no tienen ese problema.
- **[PENDIENTE de Nacho]** Informe al workshop SABID; OK para publicar el repo; qué enviar a Kaggle y cuándo.
- **Limitación de este documento:** las cifras de otros equipos las he leído hoy en sus READMEs, que cambian y a veces se contradicen (biconcavelens da 0,39 y 0,40 en distintos sitios; kaggle.too.foo da 0,32 y 0,360). Se citan con esa cautela. Las cifras de ameyypawar no se han podido confirmar hoy.

---

## 11. Fuentes

- [biconcavelens/filament-segmentation-2026](https://github.com/biconcavelens/filament-segmentation-2026) (detectores calibrados, dedup, ablaciones, huecos val→LB)
- [kaggle.too.foo: memoria de solución](https://kaggle.too.foo/competitions/filament-segmentation-2026/?img=20180202130330Ch&layer=pred) (selector, regla de emisión, pesos por acuerdo, solape)
- [Solid-Snake-99/solar-comp-2026](https://github.com/Solid-Snake-99/solar-comp-2026) (RF-DETR-Seg-L por cuadrantes + refinador, LB 0,410)
- [ranjeet258/solar-filament-consensus-2026](https://github.com/ranjeet258/solar-filament-consensus-2026) (U-Nets + YOLO + selector LightGBM)
- [rodchemist/solar-filament-segmentation-2026](https://github.com/rodchemist/solar-filament-segmentation-2026) (una U-Net bien entrenada: 0,35; acuerdo entre anotadores 0,33)
- [lovephetPTL/filament-seg-2026](https://github.com/lovephetPTL/filament-seg-2026) (cierre solo para agrupar: +0,011 en val)
- [MPkS1/solar-filament](https://github.com/MPkS1/solar-filament) (umbral alto, sobreajuste de ajuste y evaluación sobre los mismos fotogramas)
- [ameyypawar/filament-seg](https://github.com/ameyypawar/filament-seg) (cifras no confirmadas hoy)
- [roboflow/rf-detr](https://github.com/roboflow/rf-detr) (licencia Apache 2.0 de los modelos de segmentación)
- [Skeleton Recall Loss, ECCV 2024](https://arxiv.org/abs/2404.03010)
- [EdgeAttNet: barbs, MAGFiLO](https://arxiv.org/abs/2509.02964v1)
- [Diercke et al., A&A 2024](https://arxiv.org/html/2402.15407v1) (detección semisupervisada; etiquetas ruidosas y filtro de detecciones pequeñas)
- [MAGFiLO, Scientific Data 2024](https://www.nature.com/articles/s41597-024-03876-y)
- Kirillov et al., Panoptic Segmentation, CVPR 2019
