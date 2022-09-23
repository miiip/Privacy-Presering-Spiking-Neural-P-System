import snp
import numpy as np
import utils
import os
from PIL import Image

def read_images_for_letter(letter):
    spike_trains = []
    folder = '/Users/mihailplesa/Documents/Doctorat/Research/Dataset/' + letter + '/train/'
    for filename in os.listdir(folder):
        if '.png' in filename:
            img = Image.open(os.path.join(folder,filename))
            if img is not None:
                spike_train = utils.spike_train_from_image(img)
                spike_trains.append(spike_train)
    return spike_trains

def get_train_dataset():
    train_dataset = {}
    for letter in range(97, 123):
        train_dataset[chr(letter)] = read_images_for_letter(chr(letter))
    return train_dataset

def train_network_for_letter(network, train_dataset, letter):
    spike_trains_letter = train_dataset[letter]
    t = 0
    network.initialize_network(spike_trains_letter[0])
    t = network.run_network(spike_trains_letter[0], True, t, load=False)
    for index in range(1, len(spike_trains_letter)):
        network.reinitialize_network_spikes(spike_trains_letter[index], t)
        t = network.run_network(spike_trains_letter[index], True, t, load=False)
    return network

def get_standard_output(network, spike_train):
    t = network.run_network(spike_train, False, 0, load=False)
    return network.get_output(t)

def train_networks():
    train_dataset = get_train_dataset()
    outputs = {}
    train_networks = {}
    for letter in range(97, 123):
        print(chr(letter))
        network_for_letter = utils.set_network(str(chr(letter)))
        trained_network = train_network_for_letter(network_for_letter, train_dataset, str(chr(letter)))
        outputs[str(chr(letter))] = get_standard_output(trained_network, train_dataset[str(chr(letter))][0])
        train_networks[str(chr(letter))] = trained_network
    return train_networks, outputs


