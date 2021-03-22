import argparse
import importlib
import json
import math
import os
import pickle
import time

import numpy as np
import tensorflow as tf

import dataloaders
import models
import targets

FLAGS = tf.flags.FLAGS

DEFAULT_DATALOADER = 'universal_loader'
DEFAULT_MODEL = 'base_model'

if __name__ == '__main__':
  tf.flags.DEFINE_integer('batch_size', 0, 'Size of the batches for each training step.')
  tf.flags.DEFINE_integer('patch_size', 0, 'Size of each input image patch.')

  tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_integer('scale', 4, 'Scale of the input images.')
  tf.flags.DEFINE_integer('data_channels', 3, 'Number of channels of the images.')
  tf.flags.DEFINE_string('cuda_device', '0', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  tf.flags.DEFINE_string('sr_model', 'tfgraph', 'Name of the target super-resolution model.')
  tf.flags.DEFINE_string('sr_config_path', None, 'URL of the super-resolution server.')

  tf.flags.DEFINE_string('train_path', './train/', 'Base path of the trained model to be saved.')
  tf.flags.DEFINE_integer('max_steps', 50, 'The number of maximum training steps.')
  tf.flags.DEFINE_integer('log_freq', 1, 'The frequency of logging via tf.logging.')
  tf.flags.DEFINE_integer('save_max_keep', 100, '(Not used)')
  tf.flags.DEFINE_integer('summary_freq', 200, 'The frequency of logging on TensorBoard. Specify 0 to disable writing summary.')
  tf.flags.DEFINE_float('sleep_ratio', 0.05, 'The ratio of sleeping time for each training step, which prevents overheating of GPUs. Specify 0 to disable sleeping.')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('global_step', 0, 'Initial global step. Specify this to resume the training.')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.dataloader is not None):
    DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def main(unused_argv):
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.train_path)

  # data loader
  dataloader = DATALOADER_MODULE.create_loader()
  dataloader.prepare()
  
  # set batch size
  if (FLAGS.batch_size <= 0):
    num_images = dataloader.get_num_images()
    FLAGS.batch_size = num_images
  
  # SR model
  sr_model = importlib.import_module('.'+FLAGS.sr_model, 'targets').create_model()
  sr_config = None
  if (FLAGS.sr_config_path is not None):
    with open(FLAGS.sr_config_path, 'r') as f:
      sr_config = json.load(f)
      sr_config = sr_config['x%d' % (FLAGS.scale)]
  sr_model.prepare(model_type='bind', scale=FLAGS.scale, config=sr_config)

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=True, sr_model=sr_model, global_step=FLAGS.global_step)

  # model > restore
  if (FLAGS.restore_path is not None):
    model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
    tf.logging.info('restored the model')

  # model > summary
  if (FLAGS.summary_freq > 0):
    summary_path = FLAGS.train_path
    summary_writer = tf.summary.FileWriter(summary_path, graph=model.get_session().graph)
  else:
    summary_writer = None
  
  # save arguments
  arguments_path = os.path.join(FLAGS.train_path, 'arguments.json')
  with open(arguments_path, 'w') as f:
    f.write(json.dumps(FLAGS.flag_values_dict(), sort_keys=True, indent=2))

  # train
  local_train_step = 0
  while (model.global_step < FLAGS.max_steps):
    global_train_step = model.global_step + 1
    local_train_step += 1

    with_summary = True if ((FLAGS.summary_freq > 0) and (local_train_step % FLAGS.summary_freq == 0)) else False

    start_time = time.time()

    input_list, truth_list = dataloader.get_patch_batch(batch_size=FLAGS.batch_size, input_patch_size=FLAGS.patch_size)
    loss, summary = model.train_step(input_list=input_list, truth_list=truth_list, with_summary=with_summary)

    duration = time.time() - start_time
    if (FLAGS.sleep_ratio > 0 and duration > 0):
      time.sleep(min(1.0, duration*FLAGS.sleep_ratio))

    if (local_train_step % FLAGS.log_freq == 0):
      tf.logging.info('step %d, loss %.6f (%.3f sec/batch)' % (global_train_step, loss, duration))
    
    if ((summary is not None) and (summary_writer is not None)):
      summary_writer.add_summary(summary, global_step=global_train_step)

  # save final data
  tf.logging.info('saving final data')

  # image saving session
  tf_image_save_graph = tf.Graph()
  with tf_image_save_graph.as_default():
    tf_image_save_path = tf.placeholder(tf.string, [])
    tf_image_save_image = tf.placeholder(tf.float32, [None, None, FLAGS.data_channels])
    
    tf_image = tf_image_save_image
    tf_image = tf.round(tf_image)
    tf_image = tf.clip_by_value(tf_image, 0, 255)
    tf_image = tf.cast(tf_image, tf.uint8)
    
    tf_image_png = tf.image.encode_png(tf_image)
    tf_image_save_op = tf.write_file(tf_image_save_path, tf_image_png)

    tf_image_init = tf.global_variables_initializer()
    tf_image_session = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))
    tf_image_session.run(tf_image_init)
  
  num_images = dataloader.get_num_images()
  for image_index in range(num_images):
    # image
    input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index)
    if (image_name == ''):
      image_name = 'image.png'
    
    # crop center
    input_patch_size = FLAGS.patch_size
    scale = FLAGS.scale
    truth_patch_size = input_patch_size * scale
    height, width, _ = input_image.shape
    input_x = math.floor((width - input_patch_size) / 2)
    input_y = math.floor((height - input_patch_size) / 2)
    truth_x = input_x * scale
    truth_y = input_y * scale
    input_image = input_image[input_y:(input_y+input_patch_size), input_x:(input_x+input_patch_size), :]
    truth_image = truth_image[truth_y:(truth_y+truth_patch_size), truth_x:(truth_x+truth_patch_size), :]

    # get inference
    attacked_list, output_list = model.attack(input_list=[input_image])
    attacked_image = attacked_list[0]
    output_image = output_list[0]
    delta = attacked_image - input_image
    delta_image = delta + 127.5

    # save data  
    current_base_path = os.path.join(FLAGS.train_path, 'input')
    tf.gfile.MakeDirs(current_base_path)
    tf_image_session.run(tf_image_save_op, feed_dict={
      tf_image_save_path: os.path.join(current_base_path, image_name),
      tf_image_save_image: input_image
    })
    current_base_path = os.path.join(FLAGS.train_path, 'attacked')
    tf.gfile.MakeDirs(current_base_path)
    tf_image_session.run(tf_image_save_op, feed_dict={
      tf_image_save_path: os.path.join(current_base_path, image_name),
      tf_image_save_image: attacked_image
    })
    current_base_path = os.path.join(FLAGS.train_path, 'output')
    tf.gfile.MakeDirs(current_base_path)
    tf_image_session.run(tf_image_save_op, feed_dict={
      tf_image_save_path: os.path.join(current_base_path, image_name),
      tf_image_save_image: output_image
    })
    current_base_path = os.path.join(FLAGS.train_path, 'truth')
    tf.gfile.MakeDirs(current_base_path)
    tf_image_session.run(tf_image_save_op, feed_dict={
      tf_image_save_path: os.path.join(current_base_path, image_name),
      tf_image_save_image: truth_image
    })

    if (image_index == 0):
      current_base_path = os.path.join(FLAGS.train_path, 'delta')
      tf.gfile.MakeDirs(current_base_path)
      np.save(os.path.join(current_base_path, 'delta.npy'), delta)
      tf_image_session.run(tf_image_save_op, feed_dict={
        tf_image_save_path: os.path.join(current_base_path, 'delta.png'),
        tf_image_save_image: delta_image
      })
    

  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()
