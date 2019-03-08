
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
import helper


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
    node_list = []
    values_in = []
    values_out = []
    values_info = []
    # Get the core operation
    testType = cfgDict["conv_mode_(in_model_operation)"]
    name = cfgDict["test_case_notes"]
    #logger.debug(cfgDict["test_case_notes"])

    # Read the input, if "dense", flatten the input to 1D
    input_shape = (1, int(cfgDict["input_channel_num"]), int(cfgDict["input_size_h_(row)"]), int(cfgDict["input_size_w_(col)"]))

    inputs = O.helper.make_tensor_value_info('Input', O.TensorProto.FLOAT, list(input_shape))
    values_info.append(inputs)
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
    #      build padding    #
    #########################
    padding_info = [paddingTop,paddingLeft,paddingBottom,paddingRight]
    if np.sum(padding_info) > 0:
        node = O.helper.make_node(
        'Pad', # node name
        ['Input'], # inputs
        ['padding'], # outputs
        #mode='constant', # Attributes
        name='zero_padding2d',
        pads=padding_info,
        )
        node_list.append(node)
        after_pad_row = int(input_shape[2] + paddingLeft + paddingRight)
        after_pad_col = int(input_shape[3] + paddingTop + paddingBottom)
        input_shape = [input_shape[0],input_shape[1],after_pad_row,after_pad_col]
        pad = O.helper.make_tensor_value_info('padding', O.TensorProto.FLOAT, input_shape)
        values_info.append(pad)
        input_pad = 'padding'
    else:
        input_pad = 'Input'


    #########################
    #     prepare conv      #
    #########################    
    filters = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    kernel_size = (0, 0)
    if cfgDict["kernel_size_w"] != "NA":
        kernel_size = (int(cfgDict["kernel_size_w"]), int(cfgDict["kernel_size_h"]))
    strides = (int(cfgDict["conv_stride"]), int(cfgDict["conv_stride"]))
    if cfgDict["padding_mode"] == 'valid':
        padding = [0,0,0,0]
    else:
        padding = helper.getPading(input_shape[2:4],kernel_size[0],strides[0])
    channel = input_shape[2]
    activation = None
    use_bias = True
    dense_units = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    result_row = int((input_shape[2] - kernel_size[0])/strides[0] + 1)
    result_col = int((input_shape[3] - kernel_size[1])/strides[1] + 1)
    output_shape = (1,  int(cfgDict["output_channel_num"]), result_row, result_col)
     
    #########################
    #     build weights     #
    ######################### 
    weight_shape = [output_shape[1],input_shape[1],kernel_size[0],kernel_size[1]]
    weight =  O.helper.make_tensor_value_info('weight', O.TensorProto.FLOAT, list(weight_shape))
    weight_tensor = O.helper.make_tensor('weight_tensor',O.TensorProto.FLOAT,weight_shape,np.zeros(weight_shape).ravel())
    node = O.helper.make_node(
    "Constant",
    [],
    ['weight'],
    name='weight_1',
    value=weight_tensor,
    )
    node_list.append(node)
    values_info.append(weight)

    #########################
    #      build bias       #
    ######################### 
    bias_shape = [output_shape[1]]
    bias = O.helper.make_tensor_value_info('bias', O.TensorProto.FLOAT, bias_shape)
    bias_tensor = O.helper.make_tensor('bias_tensor',O.TensorProto.FLOAT,bias_shape,np.zeros(bias_shape).ravel())
    node = O.helper.make_node(
    "Constant",
    [],
    ['bias'],
    name='bias_1',
    value=bias_tensor,
    )
    node_list.append(node)
    values_info.append(bias)

    #########################
    #      build conv       #
    ######################### 
    outputs = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(output_shape))
    values_info.append(outputs)

    #logger.debug(testType)
    if testType in[ "conv", "RGBAconv", "RGBA"]: ### named by "RGBA" not "RGBAconv"
        node = O.helper.make_node(
          'Conv2D',
          [input_pad,'weight','bias'],
          [testType+'_out'],
          name = str(testType),
          kernel_shape=list(kernel_size),
          pads=padding,
          strides=list(strides),
          )
        node_list.append(node)

    elif testType == "deconv":
        node = O.helper.make_node(
          'Conv2DTranspose',
          [input_pad,'weight','bias'],
          [testType+'_out'],
          name = str(testType),
          kernel_shape=list(kernel_size),
          pads=padding,
          strides=list(strides),
          )
        node_list.append(node)

    elif testType == "dw":
        node = O.helper.make_node(
          'Conv2D',
          [input_pad,'weight','bias'],
          [testType+'_out'],
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


    values_in.append(inputs)
    values_out.append(outputs)
    #construct graph
    graph_def = O.helper.make_graph(
        node_list,
        name + '_onnx',
        values_in,
        values_out,
        value_info=values_info,
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


