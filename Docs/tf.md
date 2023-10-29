# Tensorflow

## Dataset 

tf.example - tensorflow tf.train.Example messages. When iterated over it returns these as scalar string tensors.

```
tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature0]))
tf.train.Feature(float_list=tf.train.FloatList(value=[feature1]))
tf.train.Feature(int64_list=tf.train.Int64List(value=[feature2]))

feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

Serialize:
example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
example_proto.SerializeToString()

De-serialize:
example_proto = tf.train.Example.FromString(serialized_example)
example_proto
```

from_tensor_slices() -  
``` 
X = tf.range(10)  
tf.data.Dataset.from_tensor_slices(X)  
```

TFRecordWriter() - Write a tensorflow tensor as a tfrecord shard

```
tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
```

tf.data.TFRecordDataset - read the TFRecord file shard 

```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```

tf.io.FixedLenFeature() - Used to hold feature description

```
tf.io.FixedLenFeature([], tf.int64, default_value=0)

feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}
```

tf.io.parse_single_example() - Used to parse tf.example that has been read

```
tf.io.parse_single_example(example_proto, feature_description)
```

tf.io.TFRecordWriter(filename) - Write the `tf.Example` observations to the file

```
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
```

tf.compat.v1.io.tf_record_iterator() - imports the data as is, as a tf.Example message.

```
record_iterator = tf.compat.v1.io.tf_record_iterator(path=filename)

for string_record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(string_record)
  
  print(example)
  
  # Exit after 1 iteration as this is purely demonstrative.
  break
  
print(dict(example.features.feature))
print(example.features.feature['feature3'])
print(example.features.feature['feature3'].float_list.value)

```

expand_dims()
```
image = tf.zeros([10,10,3])
tf.expand_dims(image, axis=0).shape.as_list()
tf.expand_dims(image, axis=1).shape.as_list()
tf.expand_dims(image, -1).shape.as_list()
```

as_numpy_iterator() - 
```
for item in dataset.as_numpy_iterator():
  print(item)
```

numpy_function() - Given a python function func wrap this function as an operation in a TensorFlow function. func must take numpy arrays as its arguments and return numpy arrays as its outputs. You are discouraged to use tf.numpy_function outside of prototyping and experimentation.

```
tf.numpy_function(
    func, inp, Tout, stateful=True, name=None
)

dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int64]),
          num_parallel_calls=tf.data.AUTOTUNE)

```
range() -   
``` dataset = tf.data.Dataset.range(10)```  

element_spec - 
```
dataset.element_spec
```
  
window() -   
```dataset = dataset.window(5, shift=1, drop_remainder=True)```  
  
batch() -     
flat_map() -   
```dataset = dataset.flat_map(lambda window: window.batch(5))```  
  
map() -   
```dataset = dataset.map(lambda window: (window[:-1], window[-1:]))```  
  
shuffle() -   
```dataset = dataset.shuffle(buffer_size=10)```  
  
prefetch() -   
```
dataset = dataset.batch(2).prefetch(1)  
for x,y in dataset:  
  print("x = ", x.numpy())   
  print("y = ", y.numpy())    
```    
    
reduce_sum() - Computes the sum of elements across dimensions of a tensor.

```
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)

x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x, 0).numpy()
Output - the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
```

tf.merge_dims
tf.gather
tf.reduce_join
tf.fill


