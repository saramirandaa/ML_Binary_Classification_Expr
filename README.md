# Proyecto Final  
## Reconocimiento de Expresiones Faciales Gramaticales  
**Sara Rocío Miranda Mateos**

---

## • Planteamiento del problema de negocio

A partir del link adjunto, Grammatical Facial Expressions (Dataset), se encontró un dataset con coordenadas del rostro llamado Grammatical Facial Expressions, el cual contiene secuencias extraídas de videos de hablantes de la Lengua de Señas Brasileña (Libras) donde se registran coordenadas (x, y, z) de 100 puntos faciales (ojos, cejas, nariz, boca, contorno, iris) capturadas con un sensor Kinect y etiquetadas por especialistas para representar expresiones faciales gramaticales relevantes en ese lenguaje.

Este dataset fue escogido para el proyecto porque permite entrenar modelos de machine learning capaces de clasificar expresiones no manuales que conforman aspectos gramaticales claves en la comunicación de personas sordas, lo que se alinea directamente con el problema de negocio identificado:

**● Reducir la brecha de comunicación entre la comunidad sorda y la población oyente mediante tecnología que interprete patrones faciales relevantes en lengua de señas.**

La delimitación del problema se centró únicamente en detectar presencia o ausencia de expresiones faciales gramaticales, utilizando componentes faciales aislados y un modelo entrenado con webcam estándar, evitando abarcar señas completas o traducciones complejas. Este enfoque permite desarrollar un sistema especializado que complementa herramientas de traducción y atiende justo la parte que más se pierde en los métodos existentes: la gramática facial.

El valor empresarial radica en ofrecer la única solución que interpreta expresiones faciales gramaticales en tiempo real, lo que habilita aplicaciones educativas, de accesibilidad, de asistencia en comunicación y de integración laboral, generando impacto social directo y reduciendo significativamente las barreras tecnológicas actuales.

---

## • Aplicación de técnicas de ML

Para seleccionar el mejor modelo de clasificación binaria se implementó un conjunto amplio de técnicas de Machine Learning, permitiendo comparar su desempeño bajo las mismas condiciones experimentales.

Se evaluaron modelos lineales como LDA y SVM, para validar separabilidad lineal; modelos no paramétricos como KNN, para explorar las fronteras complejas basadas en similitud; y modelos generativos como Naive Bayes, que ofrecen una referencia base aunque su supuesto de independencia no se ajusta bien a datos altamente correlacionados como en este caso los 301 puntos faciales.

También se incluyeron árboles de decisión, Random Forest, Gradient Boosting y AdaBoost, los cuales capturaron relaciones no lineales y manejaron gran dimensionalidad sin requerir escalamiento. Finalmente, se probó una red neuronal multicapa, para modelar patrones no lineales de mayor complejidad.

Además, se estimó la verosimilitud mediante la optimización de la log-verosimilitud negativa para validar el comportamiento probabilístico del modelo logístico y entender cómo los parámetros maximizan la probabilidad de observar las etiquetas reales. Esta parte confirmó la coherencia estadística del problema antes de comparar clasificadores.

Tras entrenar y evaluar todos los modelos, se seleccionó el mejor algoritmo: **Random Forest**, que obtuvo el mayor accuracy y F1-score.

---

## • Interpretación y propuesta de valor

La primera visualización, correspondiente al Análisis Discriminante Lineal (LDA), muestra que al proyectar los 301 puntos faciales en un único componente discriminante, las clases 0 y 1 quedan alineadas casi completamente sobre un mismo eje, lo que indica que la separabilidad lineal entre expresiones gramaticales y no gramaticales es limitada. Aunque LDA logra capturar la dirección óptima de separación, la cercanía de los puntos sugiere que la estructura del problema requiere modelos capaces de aprender frontera
