import train
import test
import utils
import crypto

def train_and_save():
    trained_networks, outputs = train.train_networks()
    print("Ok")
    acc = test.get_accuracy_for_letter_for_noise(trained_networks, outputs, 'a', 0, load=False)
    print(acc)
    print("Ok")
    utils.save_network_and_outputs(trained_networks, outputs)


def load_and_test(letter, noise_level):
    trained_networks, outputs = utils.load_network_and_outputs()
    acc = test.get_accuracy_for_letter_for_noise(trained_networks, outputs, letter, noise_level, load=True)
    print(acc)

def load_and_test_privacy(letter, noise_level):
    public_key, private_key = crypto.generate_key_pair()
    trained_networks, outputs = utils.load_network_and_outputs()
    acc = test.get_accuracy_over_encrypted(trained_networks, outputs, letter, noise_level, public_key, private_key)
    print(acc)

#train_and_save()

#load_and_test('a', 1)

load_and_test_privacy('a', 1)
