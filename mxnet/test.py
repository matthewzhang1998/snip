import mxnet as mx
import numpy as np

def test(arr, vec):
    initializer = mx.initializer.Constant(arr)
    temp_arr = mx.ndarray.sparse.csr_matrix((arr[0], arr[1]), dtype='float64')
    sparse_arr = mx.sym.Variable('w', stype='csr', dtype='float64')

    temp_vec = mx.nd.array(vec, dtype='float64')
    sparse_vec = mx.sym.Variable('v', stype='default', dtype='float64')

    dot = mx.symbol.sparse.dot(sparse_arr, sparse_vec)

    dot_exec = dot.bind(ctx=mx.gpu(), args={'w': temp_arr, 'v': temp_vec})
    dot_exec.forward()

    return dot_exec.outputs[0]

def main():
    arr = (np.ones((int(1e8),)), (np.random.randint(100000, size=(int(1e8),)), np.random.randint(100000, size=(int(1e8),))))

    vec = np.ones((100000,1))
    dot = test(arr, vec)

    print(dot)

if __name__ == '__main__':
    main()