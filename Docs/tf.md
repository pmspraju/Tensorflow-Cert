# Tensorflow

## Dataset 


range() -   
dataset = tf.data.Dataset.range(10)  
  
window() -   
dataset = dataset.window(5, shift=1, drop_remainder=True)  
  
batch() -     
flat_map() -   
dataset = dataset.flat_map(lambda window: window.batch(5))  
  
map() -   
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))  
  
shuffle() -   
dataset = dataset.shuffle(buffer_size=10)  
  
prefetch() -   
dataset = dataset.batch(2).prefetch(1) 
```  
for x,y in dataset:  
  print("x = ", x.numpy())   
  print("y = ", y.numpy())    
```    
    
from_tensor_slices() -   
X = tf.range(10)  
tf.data.Dataset.from_tensor_slices(X)  



  

