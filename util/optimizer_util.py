import tensorflow as tf

def get_optimizer(params):
    opt_type = params.optimizer_type.lower()
    if opt_type == 'adam':
        return tf.train.AdamOptimizer(params.learning_rate)

    elif opt_type == 'sgd':
        return tf.train.GradientDescentOptimizer(params.learning_rate)

    elif opt_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(params.learning_rate)

    elif opt_type == 'adagrad':
        return tf.train.AdagradOptimizer(params.learning_rate)

    else:
        raise ValueError("String not found.")