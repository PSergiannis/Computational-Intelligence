# Sanity check for validating 
# visibility of the GPUs to TF
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))