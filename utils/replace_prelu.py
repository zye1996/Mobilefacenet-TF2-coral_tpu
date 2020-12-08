import re

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

KERAS_MODEL_PATH = "../pretrained_model/training_model/inference_model_0.993.h5"

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer._name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    model_outputs = []

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            #new_layer = insert_layer_factory()
            #if insert_layer_name:
            #    new_layer._name = insert_layer_name
            #else:
            #   new_layer._name = '{}_{}'.format(layer.name,
                                                #new_layer.name)
            x = insert_layer_factory(layer_input, layer.weights) #new_layer(x)
            #print('Layer {} inserted after layer {}'.format(new_layer.name,
            #                                                layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        #if x.name in ["concatenate_3_1/Identity:0",
        #              "concatenate_4_1/Identity:0",
        #              "concatenate_5_1/Identity:0"]:
        #    model_outputs.append(x)
        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return keras.models.Model(inputs=model.inputs, outputs=x)#model_outputs)

if __name__ == "__main__":
    model = keras.models.load_model(KERAS_MODEL_PATH)
    #for i, layer in enumerate(model.layers):
    #    if re.match("p_re_lu*", layer.name):
    #        print(np.array(layer.weights))

    def prelu_layer_factory(input, weights):
        return keras.activations.relu(input) - keras.activations.relu((-1)*input) * weights


    model = insert_layer_nonseq(model, "p_re_lu*", prelu_layer_factory, position='replace')
    model.summary()
    model.save("replaced_prelu_model.h5")
