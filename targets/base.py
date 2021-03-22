def create_model():
  return BaseModel()

class BaseModel:

  def __init__(self):
    pass

  def prepare(self, model_type, scale, config=None):
    """
    Prepare the model to be used. This function should be called before calling any other functions.
    Args:
      model_type: Type of the model.
        'standalone': Run as a standalone model.
        'bind': Will be binded to the adversarial attack model.
      scale: Scale to be super-resolved.
      config: (Optional) Model configuration.
    """
    raise NotImplementedError
  
  def upscale(self, input_list):
    """
    Upscale the input images.
    Args:
      input_list: List of the input images.
    Returns:
      output_list: List of the upscaled images.
    """
    raise NotImplementedError