import math
import os

import numpy as np
import tensorflow as tf

from . import base

FLAGS = tf.flags.FLAGS

def create_model():
  return TFGraph()

class TFGraph(base.BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, model_type, scale, config=None):
    self.model_type = model_type
    if (config is None):
      raise ValueError('config should be provided')
    self.scale = scale
    self.config = config

    if (not 'channel_first' in self.config):
      self.config['channel_first'] = False
    if (not 'pixel_range' in self.config):
      self.config['pixel_range'] = 255.0
    if (not 'input_name' in self.config):
      self.config['input_name'] = 'sr_input'
    if (not 'output_name' in self.config):
      self.config['output_name'] = 'sr_output'

    if (model_type == 'standalone'):
      # tensorflow graph
      self.tf_graph = tf.Graph()
      with self.tf_graph.as_default():
        self.tf_input = tf.placeholder(tf.float32, [None, None, None, FLAGS.data_channels])
        self.tf_output = self._get_sr_output(self.tf_input)
              
        self.tf_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      # tensorflow session
      self.tf_session = tf.Session(config=tf.ConfigProto(
          log_device_placement=False,
          allow_soft_placement=True
      ), graph=self.tf_graph)
      self.tf_session.run(self.tf_init_op)
  
  def upscale(self, input_list):
    if (self.model_type == 'standalone'):
      feed_dict = {}
      feed_dict[self.tf_input] = input_list

      output_list = self.tf_session.run(self.tf_output, feed_dict=feed_dict)

      return output_list
    
    return self._get_sr_output(input_list)
  
  def _get_sr_output(self, input_list):
    with tf.gfile.GFile(self.config['model_path'], 'rb') as f:
      model_graph_def = tf.GraphDef()
      model_graph_def.ParseFromString(f.read())
    

    def _sr_graph(graph_input_list):
      sr_input = graph_input_list

      if (self.config['channel_first']):
        sr_input = tf.transpose(sr_input, [0, 3, 1, 2])
      else:
        sr_input = tf.identity(sr_input)
      
      sr_input = sr_input * (self.config['pixel_range'] / 255.0)

      sr_input_map = {self.config['input_name']+':0': sr_input}

      tf_output = tf.import_graph_def(model_graph_def, name='model', input_map=sr_input_map, return_elements=[self.config['output_name']+':0'])[0]
      
      sr_output = tf_output / (self.config['pixel_range'] / 255.0)

      if (self.config['channel_first']):
        sr_output = tf.transpose(sr_output, [0, 2, 3, 1])
      else:
        sr_output = tf.identity(sr_output)
      
      return sr_output
    
    sr_output = _sr_graph(input_list)

    return sr_output
