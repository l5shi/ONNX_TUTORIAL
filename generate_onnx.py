
import keras
from keras.models import load_model
from keras.layers import *
from keras.layers.convolutional import DepthwiseConv2D
from keras.models import Model
from keras import backend as K
from keras.models import Sequential

import onnx as O
import os
import sys
from collections import OrderedDict
from shutil import copyfile

import numpy as np
from IPython import embed

import re
import json
import xlrd




def readTestCase(testCaseFileName, row_numbers, sheet_name):
    # Read content of test case table
    data = xlrd.open_workbook(testCaseFileName)
    table = data.sheet_by_name(sheet_name)
    nrows = table.nrows
    ncols = table.ncols
    cfgKey = table.row_values(0)

    # Normalize key values for tables
    for i in range(len(cfgKey)):
        key = cfgKey[i]
        cfgKey[i] = key.lstrip(" ").rstrip(" ").replace(" ", "_").lower()

    # Make configuration list
    cfgList = []
    for r in row_numbers:
        cfgValue = table.row_values(r)
        cfgDict = dict(zip(cfgKey, cfgValue))
        cfgList.append(cfgDict)

    return cfgList

def buildSingleLayerModel(cfgDict):
    # Get the core operation
    testType = cfgDict["conv_mode_(in_model_operation)"]
    #logger.debug(cfgDict["test_case_notes"])

    # Read the input, if "dense", flatten the input to 1D
    input_shape = (int(cfgDict["input_size_h_(row)"]), int(cfgDict["input_size_w_(col)"]), int(cfgDict["input_channel_num"]))
    #if testType == "dense":
    #    input_shape = (int(cfgDict["input_size_w_(col)"]) * int(cfgDict["input_size_h_(row)"]),)

    # Generate input layer, if "add", make multiple inputs
    input = Input(shape=input_shape)
    if testType == "add":
        input = [input] + [Input(shape=input_shape)]

    # Cascade layers
    output = input

    # Build conv padding layer
    if cfgDict['conv_en']:
        paddingTop = int(cfgDict['up_padding_t'])
        paddingBottom = int(cfgDict['dn_padding_b'])
        paddingLeft = int(cfgDict['left_padding_l'])
        paddingRight = int(cfgDict['right_padding_r'])

        #print(paddingTop, paddingBottom, paddingLeft, paddingRight)
        if paddingTop or paddingBottom or paddingLeft or paddingRight:
            #print("hahaha")
            #print(paddingTop, paddingBottom, paddingLeft, paddingRight)
            output = ZeroPadding2D(padding = ((paddingTop, paddingBottom),(paddingLeft, paddingRight)))(input)


    # Build core layer (conv, deconv, dw, dense, add)
    coreLayer = None
    #filters = int(cfgDict["output_channel_num"])
    filters = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    kernel_size = (0, 0)
    if cfgDict["kernel_size_w"] != "NA":
        kernel_size = (int(cfgDict["kernel_size_w"]), int(cfgDict["kernel_size_h"]))
    strides = (int(cfgDict["conv_stride"]), int(cfgDict["conv_stride"]))
    padding = cfgDict["padding_mode"]
    activation = None
    use_bias = True
    #dense_units = int(cfgDict["output_channel_num"])
    dense_units = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    option = {
                'bias_initializer': 'RandomNormal'
            }
    #logger.debug(testType)
    if testType in[ "conv", "RGBAconv", "RGBA"]: ### named by "RGBA" not "RGBAconv"
        coreLayer = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            **option
            )
    elif testType == "deconv":
        coreLayer = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            output_padding=0,
            **option
            )
    elif testType == "dw":
        coreLayer = DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            **option
            )
    elif testType == "dense":
        if cfgDict["conv_en"] != 0:
            output = Flatten()(output)
        coreLayer = Dense(
            units=dense_units,
            activation=activation,
            use_bias=use_bias,
            **option
            )
    elif testType == "add":
        coreLayer = Add()

    #logger.debug(coreLayer)
    # Append core layer
    #logger.debug(cfgDict["conv_en"])
    if cfgDict["conv_en"] != 0:
        output = coreLayer(output)

    # Build activation layers
    relu_mode = cfgDict["relu_mode"]
    relu6_clamp = cfgDict["relu6_clamp"]
    if relu6_clamp == "MAX":
        relu6_clamp = 8
    reluLayer = None
    if relu_mode == 0 or relu_mode == "ReLU":
        reluLayer = ReLU()
    elif relu_mode == 3:
        reluLayer = ReLU(int(relu6_clamp))
    elif relu_mode == 2:
        reluLayer = PReLU(alpha_initializer='RandomNormal', shared_axes=[1,2])
    elif relu_mode == 1:
        reluLayer = LeakyReLU()

    # Hard code, all activation = relu
    #reluLayer = Activation("relu")
    # if reluLayer:
    #     output = reluLayer(output)

    # Build batchnorm layer
    bnLayer = BatchNormalization(beta_initializer='RandomNormal', gamma_initializer='RandomNormal', moving_mean_initializer='zeros', moving_variance_initializer='ones')

    #poolingLayerPair = []
    poolingLayer = None
    # Build pooling padding layer
    if cfgDict['pconv_en'] and cfgDict['pconv_pool_en']:
        paddingTop = int(cfgDict['pool_padding_up'])
        paddingBottom = int(cfgDict['pool_padding_dn'])
        paddingLeft = int(cfgDict['pool_padding_left'])
        paddingRight = int(cfgDict['pool_padding_right'])
        options = {}
        if  paddingTop or paddingBottom or paddingLeft or paddingRight:
            # poolingLayerPair.append(ZeroPadding2D(padding = ((paddingTop, paddingBottom),(paddingLeft, paddingRight))))
            options["padding"] = "same"
        # Build pooling layer
        #pool_info = cfgDict["pool_size_and_stride"]
        pool_size = cfgDict["pool_size"]
        pool_stride = cfgDict["pool_stride"]
        # if "_" in pool_info:
        #     pool_size, pool_stride = pool_info.split("_")
        #     pool_size = int(pool_size[1])
        #     pool_stride = int(pool_stride[1])
        pool_mode = cfgDict["pool_mode"]
        # poolingLayer = None
        if pool_mode == 0:
            #poolingLayerPair.append(MaxPooling2D(pool_size=int(pool_size), strides=int(pool_stride)))
            poolingLayer = MaxPooling2D(pool_size=int(pool_size), strides=int(pool_stride), **options)
        elif pool_mode == 1:
            #poolingLayerPair.append(MaxPooling2D(pool_size=int(pool_size), strides=int(pool_stride)))
            poolingLayer = MaxPooling2D(pool_size=int(pool_size), strides=int(pool_stride), **options)
        elif pool_mode == 2:
            #poolingLayerPair.append(AveragePooling2D(pool_size=int(pool_size), strides=int(pool_stride)))
            poolingLayer = AveragePooling2D(pool_size=int(pool_size), strides=int(pool_stride), **options)
        elif pool_mode == 3 or pool_mode == "global":
            #poolingLayerPair.append(GlobalAveragePooling2D())
            poolingLayer = GlobalAveragePooling2D()

    # Append post-conv layers
    layerDict = OrderedDict()
    orderDict = {0:"BRP", 1:"BPR", 2:"RPB", 3:"RBP", 4:"PBR", 5:"PRB"}
    layerDict["B"] = bnLayer if cfgDict["pconv_bn_en"] == 1 else None
    layerDict["P"] = poolingLayer if cfgDict["pconv_pool_en"] == 1 else None
    layerDict["R"] = reluLayer if cfgDict["pconv_relu_en"] == 1 else None
    #print(cfgDict["bn_relu_pool_order_(pconv_order)"])

    order = orderDict[cfgDict["bn_relu_pool_order_(pconv_order)"]]
    for c in order:
        if layerDict[c] != None:
            output = layerDict[c](output)

    model = Model(inputs=input, outputs=output)
    #model.compile()
    return model

def buildSingleLayerONNX(cfgDict):

    # Get the core operation
    testType = cfgDict["conv_mode_(in_model_operation)"]
    name = cfgDict["test_case_notes"]
    #logger.debug(cfgDict["test_case_notes"])

    # Read the input, if "dense", flatten the input to 1D
    input_shape = (1, int(cfgDict["input_channel_num"]), int(cfgDict["input_size_h_(row)"]), int(cfgDict["input_size_w_(col)"]))


    #if testType == "dense":
    #    input_shape = (int(cfgDict["input_size_w_(col)"]) * int(cfgDict["input_size_h_(row)"]),)

    # Generate input layer, if "add", make multiple inputs
    #input = Input(shape=input_shape)

    #if testType == "add":
    #    input = [input] + [Input(shape=input_shape)]

    # Cascade layers
    #output = input

    # Build conv padding layer
    if cfgDict['conv_en']:
        paddingTop = int(cfgDict['up_padding_t'])
        paddingBottom = int(cfgDict['dn_padding_b'])
        paddingLeft = int(cfgDict['left_padding_l'])
        paddingRight = int(cfgDict['right_padding_r'])

        #########################
        #         TO DO         #
        #########################




    # Build core layer (conv, deconv, dw, dense, add)
    coreLayer = None
    #filters = int(cfgDict["output_channel_num"])
    filters = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    kernel_size = (0, 0)
    if cfgDict["kernel_size_w"] != "NA":
        kernel_size = (int(cfgDict["kernel_size_w"]), int(cfgDict["kernel_size_h"]))
    strides = (int(cfgDict["conv_stride"]), int(cfgDict["conv_stride"]))
    if cfgDict["padding_mode"] == 'valid':
        padding = [0,0,0,0]
    else:
        padding = [paddingLeft,paddingTop,paddingRight,paddingBottom] ###?????????????????

    channel = input_shape[2]
    activation = None
    use_bias = True
    dense_units = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
  
    result_row = int((input_shape[2] - kernel_size[0] + paddingLeft + paddingRight)/strides[0] + 1)
    result_col = int((input_shape[3] - kernel_size[1] + paddingTop + paddingBottom)/strides[1] + 1)
    output_shape = (1,  int(cfgDict["output_channel_num"]), result_row, result_col)
     
    inputs = O.helper.make_tensor_value_info('input', O.TensorProto.FLOAT, list(input_shape))
    outputs = O.helper.make_tensor_value_info('output', O.TensorProto.FLOAT, list(output_shape))

    values_in = []
    values_out = []
    node_list = []
    values_in.append(inputs)
    values_out.append(outputs)
    #logger.debug(testType)
    if testType in[ "conv", "RGBAconv", "RGBA"]: ### named by "RGBA" not "RGBAconv"
        node = O.helper.make_node(
          'Conv2D',
          ['input'],
          ['output'],
          name = str(testType),
          kernel_shape=list(kernel_size),
          pads=padding,
          strides=list(strides),
          )
        node_list.append(node)

    elif testType == "deconv":
        node = O.helper.make_node(
          'Conv2DTranspose',
          ['input'],
          ['output'],
          name = str(testType),
          kernel_shape=list(kernel_size),
          pads=padding,
          strides=list(strides),
          )
        node_list.append(node)

    elif testType == "dw":
        node = O.helper.make_node(
          'Conv2D',
          ['input'],
          ['output'],
          name = str(testType),
          kernel_shape=list(kernel_size),
          pads=padding,
          strides=list(strides),
          group=channel,
          )
        node_list.append(node)

    elif testType == "dense":

        #########################
        #         TO DO         #
        #########################
        pass

    elif testType == "add":

        #########################
        #         TO DO         #
        #########################
        pass




    #PCONV
    #########################
    #         TO DO         #
    #########################

    # Build batchnorm layer
    def batchLayer(input_bn_shape, input_bn_info):
        node_lst = []
        value_info_lst = []

        # gamma
        scale_bn_info = O.helper.make_tensor_value_info('scale_bn_info', O.TensorProto.FLOAT, list(input_bn_shape[1]))
        scale_bn_tensor = O.helper.make_tensor('scale_bn_tensor', O.TensorProto.FLOAT, 
            np.random.normal(size =input_bn_shape[1]))
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
        bias_bn_info = O.helper.make_tensor_value_info('bias_bn_info', O.TensorProto.FLOAT, list(input_bn_shape[1]))
        bias_bn_tensor = O.helper.make_tensor('bias_bn_tensor', O.TensorProto.FLOAT, 
            np.random.normal(size =input_bn_shape[1]))
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
        mean_bn_info = O.helper.make_tensor_value_info('mean_bn_info', O.TensorProto.FLOAT, list(input_bn_shape[1]))
        mean_bn_tensor = O.helper.make_tensor('mean_bn_tensor', O.TensorProto.FLOAT, 
            np.random.normal(size =input_bn_shape[1]))
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
        var_bn_info = O.helper.make_tensor_value_info('var_bn_info', O.TensorProto.FLOAT, list(input_bn_shape[1]))
        var_bn_tensor = O.helper.make_tensor('var_bn_tensor', O.TensorProto.FLOAT, 
            np.random.normal(size =input_bn_shape[1]))
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
          inputs=[input_bn_info, 'scale_bn_info', 'bias_bn_info', 'mean_bn_info', 'var_bn_info'],
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
    

    # Build activation layers
    def activateLayer(input_relu_info, input_relu_shape)
        node_lst = []
        value_info_lst = []

        output_relu_shape = input_relu_shape
        output_relu_info = O.helper.make_tensor_value_info('output_relu_info', O.TensorProto.FLOAT, list(output_relu_shape))

        relu_mode = cfgDict["relu_mode"]
        relu6_clamp = cfgDict["relu6_clamp"]
        if relu6_clamp == "MAX":
            relu6_clamp = 8

        if relu_mode == 3:
            node_relu = onnx.helper.make_node(
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
            node_relu = onnx.helper.make_node(
                op_type = 'LeakyRelu',
                inputs=[input_relu_info],
                outputs=['output_relu_info'],
                alpha=0.1,
                name = 'PCONV_leakyrelu'
            )


         elif relu_mode == 0 or relu_mode == "ReLU":
            node_relu = O.helper.make_node(
              op_type='Relu',
              inputs = [input_relu_info],
              outputs = ['output_relu_info'],
              name = 'PCONV_relu'
            )

        else:
            node_relu = None
        node_lst.append(node_relu)
        value_info_lst.append(output_relu_info)
    return node_lst, value_info_lst, 'output_relu_info'


    # Build pooling padding layer
    def poolLayer(input_pool_info, input_pool_shape):
        node_lst = []
        value_info_lst = []

        output_pool_shape = input_pool_shape
        output_pool_info = O.helper.make_tensor_value_info('output_pool_info', O.TensorProto.FLOAT, list(output_pool_shape))

        if cfgDict['pconv_en'] and cfgDict['pconv_pool_en']:
            paddingTop = int(cfgDict['pool_padding_up'])
            paddingBottom = int(cfgDict['pool_padding_dn'])
            paddingLeft = int(cfgDict['pool_padding_left'])
            paddingRight = int(cfgDict['pool_padding_right'])
            options = {}
            if  paddingTop or paddingBottom or paddingLeft or paddingRight:
                # poolingLayerPair.append(ZeroPadding2D(padding = ((paddingTop, paddingBottom),(paddingLeft, paddingRight))))
                options["padding"] = "same"
                pads = [paddingTop, paddingLeft, paddingBottom, paddingRight]
            else:
                pads = [0, 0, 0, 0]

            # Build pooling layer
            #pool_info = cfgDict["pool_size_and_stride"]
            pool_size = cfgDict["pool_size"]
            pool_stride = cfgDict["pool_stride"]
            # if "_" in pool_info:
            #     pool_size, pool_stride = pool_info.split("_")
            #     pool_size = int(pool_size[1])
            #     pool_stride = int(pool_stride[1])
            pool_mode = cfgDict["pool_mode"]
            # poolingLayer = None
            strides = [int(pool_stride), int(pool_stride)]
            if pool_mode == 0 or pooling_mode == 1:
                node_pool = O.helper.make_node(
                  op_type='MaxPool',
                  inputs=['input_pool'],
                  outputs=['output_pool'],
                  name='PCONV_MaxPool',
                  kernel_shape=[int(pool_size), int(pool_size)],
                  pads=pads,
                  strides=strides
                )

            elif pool_mode == 2:
                node_pool = O.helper.make_node(
                  op_type='AveragePool',
                  inputs=['input_pool'],
                  outputs=['output_pool'],
                  name='PCONV_AveragePool',
                  kernel_shape=[int(pool_size), int(pool_size)],
                  pads=pads,
                  strides=strides
                )

            elif pool_mode == 3 or pool_mode == "global":
                node_pool = O.helper.make_node(
                  op_type='GlobalAveragePool',
                  inputs=['input_pool'],
                  outputs=['output_pool'],
                  name='PCONV_GlobalAveragePool',
                  kernel_shape=[int(pool_size), int(pool_size)],
                  pads=pads,
                  strides=strides
                )
            node_lst.append(node_pool)
            value_info_lst.append(output_pool_info)
            
        return node_lst, value_info_lst, 'output_pool_info'

    layerDict = OrderedDict()
    orderDict = {0:"BRP", 1:"BPR", 2:"RPB", 3:"RBP", 4:"PBR", 5:"PRB"}
    layerDict["B"] = bnLayer if cfgDict["pconv_bn_en"] == 1 else None
    layerDict["P"] = poolingLayer if cfgDict["pconv_pool_en"] == 1 else None
    layerDict["R"] = reluLayer if cfgDict["pconv_relu_en"] == 1 else None
    #print(cfgDict["bn_relu_pool_order_(pconv_order)"])

    order = orderDict[cfgDict["bn_relu_pool_order_(pconv_order)"]]
    for c in order:
        if layerDict[c] != None:
            output = layerDict[c](output)
    
    #construct graph
    graph_def = O.helper.make_graph(
	    node_list,
	    name + '_onnx',
	    values_in,
	    values_out,
	    )
    # Create the model (ModelProto)

    omodel = O.helper.make_model(graph_def, producer_name='Kneron')

	  # Create the model (ModelProto)
	  # O.checker.check_model(omodel)
	  #logger.debug("Conversion Finished. With op: " + str(ops))
    return omodel

def genTestCase(cfgList):

    for cfgDict in cfgList:

        # K.clear_session()
        # Build the Keras model
        # model = buildSingleLayerModel(cfgDict)
        # model.summary()

        # Build ONNX model
        model = buildSingleLayerONNX(cfgDict)

        O.save(model, 'test1.onnx')

def get_config(ver=5):
    config = OrderedDict()

    if ver == 5:
        config['input_gen'] ='random'
        config['weight_gen'] ='random'
        config['bias_gen'] ='random'
        config['fn_excel'] = "test_cases_red_20181114.xlsx"
        config['row_numbers'] = list(range(1, 102))
        config['sheet_name'] = 'test_case_list'
    elif ver == 111:
        config['input_gen'] ='debug'
        config['weight_gen'] ='debug'
        config['bias_gen'] ='all0'
        config['fn_excel'] = "test_cases_red_20181208_reorder.xlsx"
        config['row_numbers'] = list(range(1, 2))
        config['sheet_name'] = 'v11_8bit3x3'    
    else:
        raise NotImplemented

    assert (config['input_gen'] in ['random', 'debug', 'incre'])
    assert (config['weight_gen'] in ['random', 'debug', 'incre', 'all16'])
    assert (config['bias_gen'] in ['random', 'all0'])

    config['version'] = 'v{:03d}'.format(ver)

    return config



if __name__ == "__main__":
    np.random.seed(8)

    version = 111

    excel_path = "."
    output_path = "."


    #TODO: it should move config to JSON (don't store much data on code), and user get_config to get data from JSON
    configs = get_config(version)
    #configs['weight_gen'] = "random"
    # logger.debug(configs)

    testCasesFileName = "{}/{}".format(excel_path, configs['fn_excel'])
    cfgList = readTestCase(testCasesFileName, row_numbers=configs['row_numbers'], sheet_name=configs['sheet_name'])
    dir_output = "{}/test_cases_{}".format(output_path, configs['version'])
    genTestCase(cfgList)


