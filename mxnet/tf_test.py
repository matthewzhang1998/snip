import tensorflow as tf
import numpy as np

def sparse_matmul(args, sparse_matrix, scope=None, use_sparse_mul=True):
    if not isinstance(args, list):
        args = [args]

    # Now the computation.

    if len(args) == 1:
        # res = math_ops.matmul(args[0], sparse_matrix,b_is_sparse=True)
        input = args[0]
    else:
        input = tf.concat(args, 1, )
    output_shape = tf.shape(input)
    input = tf.reshape(input,
        tf.concat([[-1], [output_shape[-1]]], axis=0)
    )
    input = tf.transpose(input, perm=[1,0])

    with tf.variable_scope(scope or "Linear"):
        if use_sparse_mul:
            sparse_matrix = tf.sparse_transpose(sparse_matrix, perm=[1, 0])
            res = tf.sparse_tensor_dense_matmul(sparse_matrix, input)
            res = tf.transpose(res, perm=[1, 0])

    res = tf.reshape(res,
        tf.concat([output_shape[:-1], [-1]], axis=0)
    )
    return res

def test(arr, vec, shape):
    val = tf.Variable(arr[0], dtype=tf.float32)

    sparse_arr = tf.SparseTensor(indices=arr[1],
        values=val, dense_shape=shape)

    vec = tf.expand_dims(tf.Variable(vec, dtype=tf.float32), 1)
    res = tf.sparse_tensor_dense_matmul(sparse_arr, vec)
    return res

def main():
    arr = (np.ones((int(1e8),)),
           np.random.randint(100000, size=(int(1e8),2)))

    vec = np.ones((100000,))
    dot = test(arr, vec, shape=[100000, 100000])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    val = sess.run([dot])

    print(val)

if __name__ == '__main__':

    main()