import os.path

import tensorflow as tf

import reader #TODO module import

class PtbReaderTest(tf.test.TestCase):

    def setUp(self):
        self._string_data = '\n'.join(['hello there i am',
                                       ' rain as day',
                                       ' want some cheesy puffs ?'])

    def testPtbRawData(self):
        tmpdir = tf.test.get_temp_dir()
        for suffix in 'train', 'valid', 'test':
            filename = os.path.join(tmpdir, 'ptb.{:s}.txt'.format(suffix))
            with tf.gfile.GFile(filename, 'w') as f:
                f.write(self._string_data)

        #smoke test
        output = reader.ptb_raw_data(tmpdir)
        self.assertEqual(len(output), 4)

    def testPtbProducer(self):
        raw_data = [4,3,2,1,0,5,6,1,1,1,1,0,3,4,1]
        batch_size = 3
        num_steps = 2
        x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess, coord=coord)
            try:
                xval, yval = sess.run([x, y])
                self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
                self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
                xval, yval = sess.run([x, y])
                self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
                self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])

            finally:
                coord.request_stop()
                coord.join()

def testPtbProducer():
    raw_data = [4,3,2,1,0,5,6,1,1,1,1,0,3,4,1]
    batch_size = 3
    num_steps = 2
    x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord=coord)
        try:
            for i in range(5):
                print(i)
                xval, yval = sess.run([x, y])
                print(xval)


        finally:
            coord.request_stop()
            coord.join()

if __name__ == '__main__':
    print('helo')
    import sys
    print(sys.path)
    # tf.test.main()

    # testPtbProducer()