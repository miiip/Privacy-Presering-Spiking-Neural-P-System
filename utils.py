import numpy as np
import os
from PIL import Image
import snp
import pickle
import json


def get_min_var_letter(outputs, output):
    output = np.asarray(output)
    letter_output = np.asarray(outputs["a"])
    min = np.linalg.norm(output - letter_output)
    for letter in range(98, 123):
        letter_output = np.asarray(outputs[str(chr(letter))])
        dist = np.linalg.norm(output - letter_output)
        if dist < min:
            min = dist
            ans = chr(letter)
    return ans

def print_spike_train(spike_train):
    for spike in spike_train:
        print(spike, end=' ')
    print()

def spike_train_from_image(img):
    img_values = np.array(img)
    w, h, c = img_values.shape
    spike_train = np.zeros((w*h,)).astype(int)
    for i in range(w):
        for j in range(h):
            if 255 in img_values[i][j]:
                spike_train[i*h+j] = 1
    return spike_train

def spike_train_from_image_path(image_path):
    img = Image.open(image_path)
    return spike_train_from_image(img)

def connect_groups(R, source_group, destination_group):
    for source_neuron_idx in source_group:
        for destionation_neuron_idx in destination_group:
            synapse = snp.Synapse(R[source_neuron_idx], R[destionation_neuron_idx])
            synapse.set_weights(1, 0)
            R[source_neuron_idx].add_synapse(synapse)

def connect_group_to_neuron(R, source_group, neuron):
    for source_neuron_idx in source_group:
        synapse = snp.Synapse(R[source_neuron_idx], neuron)
        synapse.set_weights(1, 0)
        R[source_neuron_idx].add_synapse(synapse)

def set_network(netork_id):
    R = {}

    for i in range(35):
        R[i] = snp.Neuron(name="R" + str(i))

    inner_group = [12, 17, 22]
    middle_black = [11, 16, 21]
    middle_gray = [6, 7, 8]
    middle_orange = [13, 18, 23]
    middle_green = [26, 27, 28]
    outter_black = [5, 10, 15, 20, 25]
    outter_gray = [0, 1, 2, 3, 4]
    outter_orange = [9, 14, 19, 24, 29]
    outter_green = [30, 31, 32, 33, 34]

    out_black = snp.Neuron(name="OUT_BLACK")
    out_gray = snp.Neuron(name="OUT_GRAY")
    out_orange = snp.Neuron(name="OUT_ORANGE")
    out_green = snp.Neuron(name="OUT_GREEN")

    connect_groups(R, inner_group, middle_black)
    connect_groups(R, inner_group, middle_gray)
    connect_groups(R, inner_group, middle_orange)
    connect_groups(R, inner_group, middle_green)

    connect_groups(R, middle_black, outter_black)
    connect_groups(R, middle_gray, outter_gray)
    connect_groups(R, middle_orange, outter_orange)
    connect_groups(R, middle_green, outter_green)

    connect_group_to_neuron(R, outter_black, out_black)
    connect_group_to_neuron(R, outter_gray, out_gray)
    connect_group_to_neuron(R, outter_orange, out_orange)
    connect_group_to_neuron(R, outter_green, out_green)

    network = snp.Network(35, 4, netork_id)
    for i in range(len(R)):
        network.add_working_neuron(i, R[i])
        R[i].set_n_spikes(np.random.randint(0, 2), 0)

    out_black.set_n_spikes(0, 0)
    out_gray.set_n_spikes(0, 0)
    out_orange.set_n_spikes(0, 0)
    out_green.set_n_spikes(0, 0)

    network.add_output_neuron(0, out_black)
    network.add_output_neuron(1, out_gray)
    network.add_output_neuron(2, out_orange)
    network.add_output_neuron(3, out_green)



    return network

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_networks(networks, std_outputs):
    save_object(networks, 'networks.pkl')
    save_object(std_outputs, 'std_outputs.pkl')

def load_networks():
    networks = None
    std_outputs = None
    with open('networks.pkl', 'rb') as inp:
        networks = pickle.load(inp)
    with open('std_outputs.pkl', 'rb') as inp:
        std_outputs = pickle.load(inp)
    return networks, std_outputs

def save_network_and_outputs(networks, std_outputs):
    with open('networks.json', 'w') as f:
        network_json = {}
        for letter in range(97, 123):
            network = networks[str(chr(letter))]
            trained_weights = []
            for index in range(network.number_of_working_neurons):
                for synapse in network.working_neurons[index].synapses:
                    trained_weights.append(synapse.weight[max(synapse.weight.keys())])
            network_json[str(chr(letter))] = trained_weights
        json.dump(network_json, f)
        f.close()

    with open('std_outputs.json', 'w') as f:
        std_outputs_json = {}
        for letter in range(97, 123):
            std_output = std_outputs[str(chr(letter))]
            std_outputs_json[str(chr(letter))] = list(std_output)
        json.dump(std_outputs_json, f)
        f.close()
def load_network_and_outputs():
    with open('networks.json') as f:
        networks = {}
        networks_dictionary = json.load(f)
        for letter in range(97, 123):
            idx = 0
            network = set_network(str(chr(letter)))
            for index in range(network.number_of_working_neurons):
                for synapse in network.working_neurons[index].synapses:
                    synapse.trained_weight = networks_dictionary[str(chr(letter))][idx]
                    idx = idx + 1
            networks[str(chr(letter))] = network
        f.close()
    with open('std_outputs.json') as f:
        std_outputs = {}
        std_outputs_dictionary = json.load(f)
        for letter in range(97, 123):
            std_output = std_outputs_dictionary[str(chr(letter))]
            std_outputs[str(chr(letter))] = std_output
        f.close()

    return networks, std_outputs