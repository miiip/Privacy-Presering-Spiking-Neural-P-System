import utils
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import crypto


def get_spikes_for_letter_for_noise(letter, noise_level):
    spike_trains = []
    noise_string = "_" + str(noise_level) + "_"
    folder = '/Users/mihailplesa/Documents/Doctorat/Research/Dataset/' + letter + '/test/'
    for filename in os.listdir(folder):
        if '.png' in filename:
            if noise_string in filename:
                img = Image.open(os.path.join(folder, filename))
                if img is not None:
                    spike_trains.append(utils.spike_train_from_image(img))
    return spike_trains

def get_network_output_var_for_letter(network, std_output, spike_train, load):
    t = network.run_network(spike_train, False, 0, load=load)
    letter_output = network.get_output(t)
    print(letter_output)
    return np.linalg.norm(np.asarray(std_output) - np.asarray(letter_output))

def get_network_privacy_output_for_letter(network, privacy):
    t = network.run_network(privacy=privacy)
    letter_output = network.get_output(t)
    return letter_output

def get_network_privacy_var_for_letter(std_output, encrypted_output, private_key):
    decrypted_output = [crypto.private_decrypt(ciphertext, private_key) for ciphertext in encrypted_output]
    return np.linalg.norm(np.asarray(std_output) - np.asarray(decrypted_output))

def get_accuracy_for_letter_for_noise(trained_networks, std_outputs, true_letter, noise_level, load):
    spike_trains = get_spikes_for_letter_for_noise(true_letter, noise_level)
    acc = 0
    for spike_train in spike_trains:
        min = get_network_output_var_for_letter(trained_networks['a'], std_outputs['a'], spike_train, load)
        ans_letter = 'a'
        for letter in range(98, 123):
            print('Test ', str(chr(letter)))
            var = get_network_output_var_for_letter(trained_networks[str(chr(letter))], std_outputs[str(chr(letter))], spike_train, load)
            if var < min:
                min = var
                ans_letter = str(chr(letter))
        if ans_letter == true_letter:
            acc = acc + 1
    return acc / len(spike_trains)

def get_accuracy_over_encrypted(trained_networks, std_outputs, true_letter, noise_level, public_key, private_key):
    spike_trains = get_spikes_for_letter_for_noise(true_letter, noise_level)
    acc = 0
    zero_encryption = crypto.public_encryption(0, public_key)
    for spike_train in spike_trains:
        encrypted_spike_train = [crypto.public_encryption(int(spike), public_key) for spike in spike_train]
        privacy = (encrypted_spike_train, zero_encryption)
        encrypted_outputs = get_network_privacy_output_for_letter(trained_networks['a'], privacy)
        min = get_network_privacy_var_for_letter(std_outputs['a'], encrypted_outputs, private_key)
        ans_letter = 'a'
        for letter in range(98, 123):
            print('Test ', str(chr(letter)))
            encrypted_outputs = get_network_privacy_output_for_letter(trained_networks[str(chr(letter))], privacy)
            var = get_network_privacy_var_for_letter(std_outputs[str(chr(letter))], encrypted_outputs, private_key)
            if var < min:
                min = var
                ans_letter = str(chr(letter))
        if ans_letter == true_letter:
            acc = acc + 1
    return acc / len(spike_trains)

