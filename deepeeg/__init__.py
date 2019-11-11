import tensorflow as tf

tf_version = tf.__version__
print('tensorflow version: ', tf_version)

if tf.__version__.startswith('2'):
    from tensorflow import keras
else:
    import keras