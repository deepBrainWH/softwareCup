import tf_model
import tensorflow as tf


def train():
    x, y, predict, loss, accuracy, merged = tf_model.build_model()
    opt = tf.train.AdamOptimizer().minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("./log", graph=sess.graph)
    

    sess.close()

if __name__ == '__main__':
    train()
    

