import argparse
import importlib
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf

import dataloaders
import models
import targets

FLAGS = tf.flags.FLAGS

DEFAULT_DATALOADER = 'basic_loader'
DEFAULT_MODEL = 'ifgsm'

if __name__ == '__main__':
  tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_integer('scale', 4, 'Scale of the input images.')
  tf.flags.DEFINE_integer('data_channels', 3, 'Number of channels of the images.')
  tf.flags.DEFINE_string('cuda_device', '0', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  tf.flags.DEFINE_string('sr_model', 'tfgraph', 'Name of the target super-resolution model.')
  tf.flags.DEFINE_string('sr_config_path', None, 'URL of the super-resolution server.')

  tf.flags.DEFINE_string('train_path', './train_bulk/', 'Base path of the output delta and corresponding images to be saved.')
  tf.flags.DEFINE_integer('max_steps', 50, 'The number of maximum training steps.')
  tf.flags.DEFINE_integer('log_freq', 10, 'The frequency of logging via tf.logging.')
  tf.flags.DEFINE_integer('save_max_keep', 100, '(Not used)')
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
  
  # SR model
  sr_model = importlib.import_module('.'+FLAGS.sr_model, 'targets').create_model()
  sr_config = None
  if (FLAGS.sr_config_path is not None):
    with open(FLAGS.sr_config_path, 'r') as f:
      sr_config = json.load(f)
      sr_config = sr_config['x%d' % (FLAGS.scale)]
  sr_model.prepare(model_type='bind', scale=FLAGS.scale, config=sr_config)
  
  # save arguments
  arguments_path = os.path.join(FLAGS.train_path, 'arguments.json')
  with open(arguments_path, 'w') as f:
    f.write(json.dumps(FLAGS.flag_values_dict(), sort_keys=True, indent=2))

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
  
  # model
  model = None
  
  # train for each image
  num_images = dataloader.get_num_images()
  image_indices = list(range(num_images))
  
  while (len(image_indices) > 0):
    if (model != None):
      del model
      model = None
    
    remaining_image_indices = []

    for image_index in image_indices:
      # image
      input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index)
      truth_image = truth_image[0:input_image.shape[0]*4, 0:input_image.shape[1]*4, :]

      # model
      if (model == None):
        FLAGS.model_input_width = input_image.shape[1]
        FLAGS.model_input_height = input_image.shape[0]
        model = MODEL_MODULE.create_model()
        model.prepare(is_training=True, sr_model=sr_model, global_step=FLAGS.global_step)
      else:
        if ((FLAGS.model_input_width != input_image.shape[1]) or (FLAGS.model_input_height != input_image.shape[0])):
          remaining_image_indices.append(image_index)
          continue
        model.tf_session.run(model.tf_init_op)
        model.global_step = FLAGS.global_step

      # model > restore
      if (FLAGS.restore_path is not None):
        model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
        tf.logging.info('%d: restored the model' % (image_index))
      
      # train
      local_train_step = 0
      while (model.global_step < FLAGS.max_steps):
        global_train_step = model.global_step + 1
        local_train_step += 1

        start_time = time.time()

        loss, _ = model.train_step(input_list=[input_image], truth_list=[truth_image], with_summary=False)

        duration = time.time() - start_time
        if (FLAGS.sleep_ratio > 0 and duration > 0):
          time.sleep(min(1.0, duration*FLAGS.sleep_ratio))

        if (local_train_step % FLAGS.log_freq == 0):
          tf.logging.info('%d: step %d, loss %.6f (%.3f sec/batch)' % (image_index, global_train_step, loss, duration))
      
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
      current_base_path = os.path.join(FLAGS.train_path, 'delta')
      tf.gfile.MakeDirs(current_base_path)
      np.save(os.path.join(current_base_path, image_name+'.npy'), delta)
      tf_image_session.run(tf_image_save_op, feed_dict={
        tf_image_save_path: os.path.join(current_base_path, image_name),
        tf_image_save_image: delta_image
      })
    
    image_indices = remaining_image_indices

    
  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()
