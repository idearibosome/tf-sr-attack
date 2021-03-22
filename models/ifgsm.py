import math
import os
import pickle
import zlib

import numpy as np
import tensorflow as tf

from models.base_model import BaseModel

"""
Iterative FGSM-based
"""

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('model_alpha', 0.001, 'Alpha value of the model.')
tf.flags.DEFINE_float('model_eps', 0.001, 'Epsilon value of the model.')
tf.flags.DEFINE_integer('model_input_width', 0, 'Width of the input image.')
tf.flags.DEFINE_integer('model_input_height', 0, 'Height of the input image.')
tf.flags.DEFINE_integer('model_input_channels', 3, 'Number of channels of the input image.')
tf.flags.DEFINE_boolean('model_targeted', False, 'Train the model to make the SR model generates output towards the ground-truth image.')
tf.flags.DEFINE_boolean('model_random_init', False, 'Specify this to initialize the perturbation randomly (Uniform(-eps, eps)).')
tf.flags.DEFINE_boolean('model_averaged_gradients', False, 'Train the model by computing gradient for each image and then average it.')

def create_model():
  return IFGSMModel()

class IFGSMModel(BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, is_training, sr_model, global_step=0):
    # config. parameters
    self.global_step = global_step


    # tensorflow graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      
      if (is_training):
        self.tf_input = tf.placeholder(tf.float32, [None, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels], name=BaseModel.TF_INPUT_NAME)
        self.tf_sr_truth = tf.placeholder(tf.float32, [None, None, None, FLAGS.model_input_channels])

        self._create_attack_variable()

        if (FLAGS.model_averaged_gradients):
          def _loss_gradient_fn(input_image, truth_image):
            input_list = [input_image]
            output_list = self._attack(is_training=is_training, input_list=input_list)
            sr_output_list = sr_model.upscale(input_list=output_list)
            sr_loss, sr_gradient = self._calculate_loss_and_gradient(input_list=input_list, output_list=output_list, sr_list=sr_output_list, sr_truth_list=self.tf_sr_truth)
            
            sr_gradient = tf.reshape(sr_gradient, [1, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels])

            return output_list[0], sr_output_list[0], sr_loss, sr_gradient
          
          self.tf_output, self.tf_sr_output, loss_sr_list, loss_sr_gradient_list = tf.map_fn(lambda x: _loss_gradient_fn(x[0], x[1]), (self.tf_input, self.tf_sr_truth), dtype=(tf.float32, tf.float32, tf.float32, tf.float32), swap_memory=True, parallel_iterations=1)

          loss_sr = tf.reduce_mean(loss_sr_list)
          loss_sr_gradient = tf.reduce_mean(loss_sr_gradient_list, axis=0)

        else:
          self.tf_output = self._attack(is_training=is_training, input_list=self.tf_input)
          self.tf_sr_output = tf.map_fn(lambda x: sr_model.upscale(input_list=[x])[0], self.tf_output, dtype=tf.float32, swap_memory=True, parallel_iterations=1)
          loss_sr, loss_sr_gradient = self._calculate_loss_and_gradient(input_list=self.tf_input, output_list=self.tf_output, sr_list=self.tf_sr_output, sr_truth_list=self.tf_sr_truth)


        input_summary = tf.cast(tf.clip_by_value(self.tf_input, 0.0, 255.0), tf.uint8)
        tf.summary.image('input', input_summary)
        output_summary = tf.cast(tf.clip_by_value(self.tf_output, 0.0, 255.0), tf.uint8)
        tf.summary.image('output', output_summary)
        diff_summary = tf.cast(tf.clip_by_value((self.tf_output-self.tf_input+127.5), 0.0, 255.0), tf.uint8)
        tf.summary.image('diff', diff_summary)
        sr_summary = tf.cast(tf.clip_by_value(self.tf_sr_output, 0.0, 255.0), tf.uint8)
        tf.summary.image('sr', sr_summary)
        sr_truth_summary = tf.cast(tf.clip_by_value(self.tf_sr_truth, 0.0, 255.0), tf.uint8)
        tf.summary.image('sr_truth', sr_truth_summary)

        self.tf_global_step = tf.placeholder(tf.int64, [])
        self.tf_train_op, self.tf_loss = self._optimize(loss_sr=loss_sr, loss_sr_gradient=loss_sr_gradient, global_step=self.tf_global_step)

        for key, loss in self.loss_dict.items():
          tf.summary.scalar(('loss/%s' % (key)), loss)

        self.tf_saver = tf.train.Saver(max_to_keep=FLAGS.save_max_keep, var_list=self._get_attack_variables())
        self.tf_summary_op = tf.summary.merge_all()

      else:
        self._create_attack_variable()
        self.tf_input = tf.placeholder(tf.float32, [None, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels], name=BaseModel.TF_INPUT_NAME)
        self.tf_output = self._attack(is_training=False, input_list=self.tf_input)
        self.tf_sr_output = sr_model.upscale(input_list=self.tf_output)
      
      self.tf_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    # tensorflow session
    self.tf_session = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ), graph=self.tf_graph)
    self.tf_session.run(self.tf_init_op)
      
  
  def save(self, base_path):
    save_path = os.path.join(base_path, 'model.ckpt')
    self.tf_saver.save(sess=self.tf_session, save_path=save_path, global_step=self.global_step, write_meta_graph=False)


  def restore(self, ckpt_path, target=None):
    with self.tf_graph.as_default():
      restorer = tf.train.Saver(var_list=self._get_attack_variables())
      restorer.restore(sess=self.tf_session, save_path=ckpt_path)
  

  def get_session(self):
    return self.tf_session


  def train_step(self, input_list, truth_list, with_summary=False):
    feed_dict = {}
    feed_dict[self.tf_input] = input_list
    feed_dict[self.tf_sr_truth] = truth_list
    feed_dict[self.tf_global_step] = self.global_step

    summary = None

    if (with_summary):
      _, loss, summary = self.tf_session.run([self.tf_train_op, self.tf_loss, self.tf_summary_op], feed_dict=feed_dict)
    else:
      _, loss = self.tf_session.run([self.tf_train_op, self.tf_loss], feed_dict=feed_dict)

    self.global_step += 1

    return loss, summary
  
  def attack(self, input_list):
    feed_dict = {}
    feed_dict[self.tf_input] = input_list

    output_list, sr_list = self.tf_session.run([self.tf_output, self.tf_sr_output], feed_dict=feed_dict)

    return output_list, sr_list
  

  def _get_attack_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attack')
  
  def _create_attack_variable(self):
    with tf.variable_scope('attack'):
      if (FLAGS.model_random_init):
        self.delta = tf.Variable(tf.random_uniform([1, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels], minval=-FLAGS.model_eps, maxval=FLAGS.model_eps))
      else:
        self.delta = tf.Variable(tf.zeros([1, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels]))

  def _attack(self, is_training, input_list, reuse=False):
    with tf.variable_scope('attack', reuse=reuse):
      # pre-process
      input_list = tf.cast(input_list, tf.float32)
      x = input_list / 255.0

      # attack
      x = x + self.delta

      # post-process
      x = tf.clip_by_value(x, 0.0, 1.0)
      x = x * 255.0

      return x
    
  def _calculate_loss_and_gradient(self, input_list, output_list, sr_list, sr_truth_list):
    loss_sr = tf.reduce_mean(tf.pow(((sr_truth_list - sr_list) / 255.0), 2))
    if (FLAGS.model_targeted):
      loss_sr = -loss_sr
    
    loss_sr_gradient = tf.gradients(loss_sr, self.delta)

    return loss_sr, loss_sr_gradient
  
  def _optimize(self, loss_sr, loss_sr_gradient, global_step):
    loss_sr_gradient_sign = tf.stop_gradient(tf.sign(loss_sr_gradient))
    loss_sr_gradient_sign = tf.reshape(loss_sr_gradient_sign, [1, FLAGS.model_input_height, FLAGS.model_input_width, FLAGS.model_input_channels])

    loss_output = -loss_sr
    self.loss_dict['final'] = loss_output

    delta_new = self.delta + ((FLAGS.model_alpha / FLAGS.max_steps) * loss_sr_gradient_sign)
    delta_new = tf.clip_by_value(delta_new, -FLAGS.model_eps, FLAGS.model_eps)
    train_op = self.delta.assign(delta_new)
    
    return train_op, loss_output





