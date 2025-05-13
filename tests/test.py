import tensorflow as tf, time
# Ensure GPU is the default device
print("Device before operation:", tf.test.gpu_device_name())
# Run a simple matmul
with tf.device('/GPU:0'):
    a = tf.random.uniform((1024,1024))
    b = tf.random.uniform((1024,1024))
    t0 = time.time()
    c = tf.matmul(a, b)
    # Force evaluation
    _ = c.numpy()
print(f"MatMul on GPU succeeded in {time.time()-t0:.3f}s, result device: {c.device}")

