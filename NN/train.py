import json
import os
import time

import tensorflow as tf

from .model import CNN
from .data_loader import Loader


def train(a):
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # no need to load options from options.json
    loader = Loader(a.source_dir, a.batch_size)

    # ---------------------------------
    tf.reset_default_graph()
    train_cnn = CNN(loader.height, loader.width, loader.depth, opt=a)
    train_cnn.build_graph(False, True)

    val_cnn = CNN(loader.height, loader.width, loader.depth, opt=a)
    val_cnn.build_graph(True, False)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in train_cnn.grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    saver = tf.train.Saver(max_to_keep=50)
    # ---------------------------------

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(a.output_dir, sess.graph)

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        start = time.time()
        for epoch in range(a.max_epochs):
            def should(freq):
                return freq > 0 and ((epoch + 1) % freq == 0 or epoch == a.max_epochs - 1)

            fetches = {
                "train": train_cnn.train,
                "loss": train_cnn.loss
            }

            training_loss = 0
            for _ in range(loader.ntrain):
                X, y = loader.next_batch(0)
                results = sess.run(fetches, {train_cnn.input: X, train_cnn.target: y})
                training_loss += results['loss']
            training_loss /= loader.ntrain

            if should(a.summary_freq):
                summary = sess.run(merged, {train_cnn.input: X, train_cnn.target: y})
                writer.add_summary(summary, global_step=epoch)
                print("recording summary")
                with open(os.path.join(a.output_dir, 'loss_record.txt'), "a") as loss_file:
                    loss_file.write("%s\t%s\t%s\n" % (epoch, training_loss, validation_loss))

            if should(a.validation_freq):
                print('validating model')
                validation_loss = 0
                for _ in range(loader.nval):
                    X, y = loader.next_batch(1)
                    loss = sess.run(val_cnn.loss, {val_cnn.input: X, val_cnn.target: y})
                    validation_loss += loss
                validation_loss /= loader.nval

            if should(a.progress_freq):
                rate = (epoch + 1) / (time.time() - start)
                remaining = (a.max_epochs - 1 - epoch) / rate
                print("progress  epoch %d  remaining %dh" % (epoch, remaining / 3600))
                print("training loss", training_loss)

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=epoch)


if __name__ == '__main__':
    train()


