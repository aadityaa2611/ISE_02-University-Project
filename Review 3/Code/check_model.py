import tensorflow as tf
import os, json

cwd = os.path.dirname(__file__)
model_path = os.path.join(cwd, 'indian_livestock_model.h5')
classes_path = os.path.join(cwd, 'class_names.json')

model = tf.keras.models.load_model(model_path, compile=False)
print('output_shape', model.output_shape)
print('num outputs', model.output_shape[-1])

# load class names and verify
with open(classes_path, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print('class count', len(class_names))
if len(class_names) != model.output_shape[-1]:
    print('Mismatch detected')
else:
    print('Counts match perfectly')
