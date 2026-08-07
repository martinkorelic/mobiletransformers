"""REMOVED — moved to ``research/tflite/tflite_builder.py`` (Migration Map S5).

TFLite export needs ``tensorflow``/``keras``/``keras_nlp``, which appear in **no** dependency profile,
so it cannot be part of the shipped package. It is exploratory code and now lives with the rest of the
research tree. There is no shim: importing it here would fail on the missing dependencies anyway.
"""

raise ModuleNotFoundError(
    "artifact.tflite_builder moved to research/tflite/tflite_builder.py (it requires tensorflow/"
    "keras/keras_nlp, which are in no dependency profile)"
)
