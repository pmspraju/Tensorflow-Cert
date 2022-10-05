import tensorflow as tf
import tensorflow_text
modelDir = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer\model'
reloaded = tf.saved_model.load(modelDir)
print(reloaded('este é o primeiro livro que eu fiz.').numpy())