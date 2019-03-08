import numpy as np
import onnx as O


def reluLayer(input_relu_shape, input_relu_info):

    node_lst = []
    value_info_lst = []

    output_relu_shape = input_relu_shape
    output_relu_info = O.helper.make_tensor_value_info('output_relu_info', O.TensorProto.FLOAT, list(output_relu_shape))

    # relu_mode = cfgDict["relu_mode"]
    # relu6_clamp = cfgDict["relu6_clamp"]

    relu_mode = 3
    relu6_clamp =5

    if relu6_clamp == "MAX":
        relu6_clamp = 8

    if relu_mode == 3:
        
        node_relu = O.helper.make_node(
            'Clip',
            inputs=[input_relu_info],
            outputs=['output_relu_info'],
            min=0,
            max=relu6_clamp,
            name = 'relu6_clamp'
        )

    elif relu_mode == 2:
        slope_relu = O.helper.make_tensor_value_info('slope_relu_info', O.TensorProto.FLOAT, list(input_relu_shape[1:]))
        # channel, heigh and width, but want to share axis in w and h
        node_relu = O.helper.make_node(
            op_type='PRelu',
            inputs = [input_relu_info, 'slope_relu_info'],
            outputs = ['output_relu_info'],
            name = 'PCONV_prelu'
        )

    elif relu_mode == 1:
        node_relu = O.helper.make_node(
            op_type = 'LeakyRelu',
            inputs=[input_relu_info],
            outputs=['output_relu_info'],
            alpha=0.1,
            name = 'PCONV_leakyrelu'
        )


    elif (relu_mode == 0) or (relu_mode == "ReLU"):
        node_relu = O.helper.make_node(
          op_type='Relu',
          inputs = [input_relu_info],
          outputs = ['output_relu_info'],
          name = 'PCONV_relu'
        )

    else:
        return reluLayer = None

    node_lst.append(node_relu)
    value_info_lst.append(output_relu_info)  

    reluLayer = [node_lst, value_info_lst, 'output_relu_info']


    return reluLayer






input_relu_shape = (1, 10, 5, 5)
input_relu_info = O.helper.make_tensor_value_info('input_relu_info', O.TensorProto.FLOAT, list(input_relu_shape))
input_str_info = 'input_relu_info'


node_list, value_info_lst, output_str_info =  activateLayer(input_relu_shape, input_str_info)

 #construct graph
graph_def = O.helper.make_graph(
  node_list,
 '_onnx',
  [input_relu_info],
  [value_info_lst[-1]],

  value_info = value_info_lst,

  )
# Create the model (ModelProto)

omodel = O.helper.make_model(graph_def, producer_name='Kneron')



O.save(omodel, 'ssss2.onnx')