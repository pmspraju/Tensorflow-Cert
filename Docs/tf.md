# Tensorflow

## Dataset 


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
    
from_tensor_slices() -  
``` 
X = tf.range(10)  
tf.data.Dataset.from_tensor_slices(X)  
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

reduce_sum() - Computes the sum of elements across dimensions of a tensor.

```
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)

x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x, 0).numpy()
Output - the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
```

