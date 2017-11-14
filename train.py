import tensorflow as tf
import os
from tqdm import tqdm

from config import FLAGS
from capsnet import CapsuleNet
from util import load_mnist


def main(_):

    capsNet = CapsuleNet(FLAGS.is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=FLAGS.logdir,
                             save_model_secs=0)

    path = FLAGS.results + '/accuracy.csv'
    if not os.path.exists(FLAGS.results):
        os.mkdir(FLAGS.results)
    elif os.path.exists(path):
        os.remove(path)

    fd_results = open(path, 'w')
    fd_results.write('step,test_acc\n')
    with sv.managed_session() as sess:
        num_batch = int(60000 / FLAGS.batch_size)
        num_test_batch = int(10000 / FLAGS.batch_size)
        test_image, test_label = load_mnist(FLAGS.dataset, False)
        for epoch in range(FLAGS.epoch):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                global_step = sess.run(capsNet.global_step)
                sess.run(capsNet.train_op)

                if step % FLAGS.train_sum_freq == 0:
                    _, summary_str = sess.run([capsNet.train_op, capsNet.train_summary])
                    sv.summary_writer.add_summary(summary_str, global_step)

                if (global_step + 1) % FLAGS.test_sum_freq == 0:
                    test_acc = 0
                    for i in range(num_test_batch):
                        start = i * FLAGS.batch_size
                        end = start + FLAGS.batch_size
                        test_acc += sess.run(capsNet.batch_accuracy,
                                             feed_dict={capsNet.image: test_image[start:end],
                                                        capsNet.label: test_label[start:end]})
                    test_acc = test_acc / (FLAGS.batch_size * num_test_batch)
                    fd_results.write(str(global_step + 1) + ',' + str(test_acc) + '\n')
                    fd_results.flush()
            if epoch % FLAGS.save_freq == 0:
                sv.saver.save(sess, FLAGS.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
    fd_results.close()
    tf.logging.info('Training done')


if __name__ == '__main__':
    tf.app.run()

