import numpy as np
import onnx as O


def bnLayer(input_bn_shape, input_str_info):
  node_lst = []
  value_info_lst = []

  # gamma
  scale_bn_info = O.helper.make_tensor_value_info('scale_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
  scale_bn_tensor = O.helper.make_tensor('scale_bn_tensor', O.TensorProto.FLOAT, 
    [input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))

  node_scale_bn = O.helper.make_node( 
    op_type='Constant',
    inputs=[],
    outputs=['scale_bn_info'],
    name='Scale_BN',
    value=scale_bn_tensor
    )
  node_lst.append(node_scale_bn)
  value_info_lst.append(scale_bn_info)


  # beta
  bias_bn_info = O.helper.make_tensor_value_info('bias_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
  bias_bn_tensor = O.helper.make_tensor('bias_bn_tensor', O.TensorProto.FLOAT, 
    [input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
  node_bias_bn = O.helper.make_node( 
    op_type='Constant',
    inputs=[],
    outputs=['bias_bn_info'],
    name='Bias_BN',
    value=bias_bn_tensor
    )
  node_lst.append(node_bias_bn)
  value_info_lst.append(bias_bn_info)


  # mean
  mean_bn_info = O.helper.make_tensor_value_info('mean_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
  mean_bn_tensor = O.helper.make_tensor('mean_bn_tensor', O.TensorProto.FLOAT, 
    [input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
  node_mean_bn = O.helper.make_node( 
    op_type='Constant',
    inputs=[],
    outputs=['mean_bn_info'],
    name='Mean_BN',
    value=mean_bn_tensor
    )

  node_lst.append(node_mean_bn)
  value_info_lst.append(mean_bn_info)

  # var
  var_bn_info = O.helper.make_tensor_value_info('var_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
  var_bn_tensor = O.helper.make_tensor('var_bn_tensor', O.TensorProto.FLOAT, 
    [input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
  node_var_bn = O.helper.make_node( 
    op_type='Constant',
    inputs=[],
    outputs=['var_bn_info'],
    name='Var_BN',
    value=var_bn_tensor
    )
  node_lst.append(node_var_bn)
  value_info_lst.append(var_bn_info)



  output_bn_shape = input_bn_shape # (N x C x H x W)
  output_bn_info = O.helper.make_tensor_value_info('output_bn_info', O.TensorProto.FLOAT, list(output_bn_shape))
  node_bn = O.helper.make_node( 
    op_type='BatchNormalization',
    inputs=[input_str_info, 'scale_bn_info', 'bias_bn_info', 'mean_bn_info', 'var_bn_info'],
    outputs=['output_bn_info'],
    name='PCONV_BN',
    # epsilon=self.layer.epsilon,
    # momentum=self.layer.momentum,
    spatial=1,
    is_test=1
    )
  node_lst.append(node_bn)
  value_info_lst.append(output_bn_info)



  return node_lst, value_info_lst, 'output_bn_info'



input_bn_shape = (1, 10, 5, 5)
input_bn_info = O.helper.make_tensor_value_info('input_bn_info', O.TensorProto.FLOAT, list(input_bn_shape))
input_str_info = 'input_bn_info'


node_list, value_info_lst, output_str_info =  batchLayer(input_bn_shape, input_str_info)

 #construct graph
graph_def = O.helper.make_graph(
  node_list,
 '_onnx',
  [input_bn_info],
  [value_info_lst[-1]],

  value_info = value_info_lst,

  )
# Create the model (ModelProto)

omodel = O.helper.make_model(graph_def, producer_name='Kneron')



O.save(omodel, 'ssss.onnx')