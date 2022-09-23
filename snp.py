import numpy as np
import crypto

class Synapse:
    def __init__(self, input_n, output_n):
        self.weight = {}
        self.input_n = input_n
        self.output_n = output_n
        self.trained_weight = 1

    def set_weights(self, weight, t):
        self.weight[t] = weight

    def update_weight(self, t):
        self.weight[t] = self.weight[t - 1] + 1

class Neuron:
    def __init__(self, name):
        self.name = name
        self.n_spikes = {}
        self.synapses = []
        self.n_spikes_trained = 0

    def set_n_spikes(self, n_spikes, t):
        self.n_spikes[t] = n_spikes

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

class Network:
    def __init__(self, number_of_working_neurons, number_of_output_neurons, network_id):
        self.network_id = network_id
        self.working_neurons = {}
        self.output_neurons = {}
        self.number_of_working_neurons = number_of_working_neurons
        self.number_of_output_neurons = number_of_output_neurons


    def add_working_neuron(self, index, neuron):
        self.working_neurons[index] = neuron

    def add_output_neuron(self, index, neuron):
        self.output_neurons[index] = neuron

    def initialize_network(self, spike_train):
        for index in range(self.number_of_working_neurons):
            self.working_neurons[index].n_spikes[0] = spike_train[index]
            for synapse in self.working_neurons[index].synapses:
                synapse.weight[0] = 1
        for index in range(self.number_of_output_neurons):
            self.output_neurons[index].n_spikes[0] = 0

    def reinitialize_network_spikes(self, spike_train, t):
        for index in range(self.number_of_working_neurons):
            if self.working_neurons[index].n_spikes[t] != 0:
                print("Wrong number of spikes")
                exit(0)
            self.working_neurons[index].n_spikes[t] = spike_train[index]

    def initialize_network_prediction(self, spike_train, load):
        for index in range(self.number_of_working_neurons):
            self.working_neurons[index].n_spikes = {}
            self.working_neurons[index].n_spikes[0] = spike_train[index]
            if load == False:
                for synapse in self.working_neurons[index].synapses:
                    synapse.trained_weight = synapse.weight[max(synapse.weight.keys())]

        for index in range(self.number_of_output_neurons):
            self.output_neurons[index].n_spikes = {}
            self.output_neurons[index].n_spikes[0] = 0


    def initialize_forward_synapse_weights(self, t):
        for index in range(self.number_of_working_neurons):
            for synapse in self.working_neurons[index].synapses:
                synapse.weight[t + 1] = synapse.weight[t]

    def initialize_forward_n_spikes_neurons(self, t):
        for index in range(self.number_of_working_neurons):
            self.working_neurons[index].n_spikes[t + 1] = self.working_neurons[index].n_spikes[t]
        for index in range(self.number_of_output_neurons):
            self.output_neurons[index].n_spikes[t + 1] = self.output_neurons[index].n_spikes[t]


    def forward_time_t(self, t):
        self.initialize_forward_synapse_weights(t)
        self.initialize_forward_n_spikes_neurons(t)
        for index in range(self.number_of_working_neurons):
            if self.working_neurons[index].n_spikes[t] > 0:
                for synapse in self.working_neurons[index].synapses:
                    synapse.output_n.n_spikes[t + 1] = synapse.output_n.n_spikes[t + 1] + 1
                    #Learning rule
                    synapse.weight[t + 1] = synapse.weight[t + 1] + 1
                self.working_neurons[index].n_spikes[t + 1] = self.working_neurons[index].n_spikes[t + 1] - 1

    def forward_time_t_prediction(self, t):
        self.initialize_forward_n_spikes_neurons(t)
        for index in range(self.number_of_working_neurons):
            if self.working_neurons[index].n_spikes[t] > 0:
                for synapse in self.working_neurons[index].synapses:
                    synapse.output_n.n_spikes[t + 1] = synapse.output_n.n_spikes[t + 1] + synapse.trained_weight
                self.working_neurons[index].n_spikes[t + 1] = self.working_neurons[index].n_spikes[t + 1] - 1

    def forward_time_t_privacy_prediction(self, t, zero_encryption, one_encryption, public_key):
        self.initialize_forward_output_neurons(t)
        for index in range(self.number_of_working_neurons):
            self.working_neurons[index].n_spikes[t + 1] = self.working_neurons[index].n_spikes[t]
            for synapse in self.working_neurons[index].synapses:
                synapse.output_n.n_spikes[t + 1] = synapse.output_n.n_spikes.get(t + 1, zero_encryption) + crypto.public_encryption(synapse.trained_weight, public_key)
            synapse.input_n.n_spikes[t + 1] = synapse.input_n.n_spikes.get(t + 1, zero_encryption) - one_encryption




    def check_if_halts(self, t):
        for index in range(self.number_of_working_neurons):
            if self.working_neurons[index].n_spikes[t] > 0:
                return False
        return True

    def run_training(self, spike_train, t):
        halts = self.check_if_halts(t)
        while halts == False:
            self.forward_time_t(t)
            t = t + 1
            halts = self.check_if_halts(t)
        return t

    def run_predicting(self, spike_train, load):
        self.initialize_network_prediction(spike_train, load)
        t = 0
        halts = self.check_if_halts(t)
        while halts == False:
            self.forward_time_t_prediction(t)
            t = t + 1
            halts = self.check_if_halts(t)
        print(t)
        return t

    def initialize_weights_for_one_time_prediction(self):
        for index in range(self.number_of_working_neurons):
            for synapse in self.working_neurons[index].synapses:
                synapse.trained_weight = synapse.weight[max(synapse.weight.keys())]

    def initialize_output_group(self, value):
        for index in range(self.number_of_output_neurons):
            self.output_neurons[index].n_spikes_trained = value
            self.output_neurons[index].n_spikes = {}

    def initialize_group(self, neurons_group, spike_train):
        for index in neurons_group:
            self.working_neurons[index].n_spikes_trained = spike_train[index]
            self.working_neurons[index].n_spikes = {}

    def create_groups(self):
        inner_group = [12, 17, 22]
        middle_black = [11, 16, 21]
        middle_gray = [6, 7, 8]
        middle_orange = [13, 18, 23]
        middle_green = [26, 27, 28]
        middle_group = middle_black + middle_gray + middle_orange + middle_green
        outter_black = [5, 10, 15, 20, 25]
        outter_gray = [0, 1, 2, 3, 4]
        outter_orange = [9, 14, 19, 24, 29]
        outter_green = [30, 31, 32, 33, 34]
        outter_group = outter_black + outter_gray + outter_orange + outter_green
        return inner_group, middle_group, outter_group

    def initialize_all_groups(self, spike_train, value):
        inner_group, middle_group, outter_group = self.create_groups()
        self.initialize_group(inner_group, spike_train)
        self.initialize_group(middle_group, spike_train)
        self.initialize_group(outter_group, spike_train)
        self.initialize_output_group(value)

    def run_group(self, neurons_group):
        for index in neurons_group:
            for synapse in self.working_neurons[index].synapses:
                synapse.output_n.n_spikes_trained = synapse.output_n.n_spikes_trained + synapse.trained_weight * synapse.input_n.n_spikes_trained
        self.working_neurons[index].n_spikes_trained = 0

    def run_all_groups(self):
        inner_group, middle_group, outter_group = self.create_groups()
        self.run_group(inner_group)
        self.run_group(middle_group)
        self.run_group(outter_group)

    def run_predicting_one_time(self, spike_train, load):
        self.initialize_all_groups(spike_train, 0)
        if load == False:
            self.initialize_weights_for_one_time_prediction()
        self.run_all_groups()

        return -1



    def run_privacy_predicting(self, encrypted_spike_train, zero_encryption):
        self.initialize_all_groups(encrypted_spike_train, zero_encryption)
        self.run_all_groups()
        return -1

    def run_network(self, spike_train=None, train=False, t=0, load=False, privacy=None):
        if train == True:
            return self.run_training(spike_train, t)
        else:
            if privacy is None:
                return self.run_predicting_one_time(spike_train, load)
            else:
                return self.run_privacy_predicting(privacy[0], \
                                                   privacy[1])

    def print_network(self, t):
        for index in range(self.number_of_working_neurons):
            print("Neuron ", self.working_neurons[index].name, "(", str(self.working_neurons[index].n_spikes), " spikes) ", "is connected to:")
            for synapse in self.working_neurons[index].synapses:
                print("\t ", synapse.output_n.name, " with weight ", synapse.weight[t])

    def print_spikes(self, t):
        if t == -1:
            for index in range(self.number_of_working_neurons):
                print(self.working_neurons[index].name, self.working_neurons[index].n_spikes_trained, end=' /')
            for index in range(self.number_of_output_neurons):
                print(self.output_neurons[index].name, self.output_neurons[index].n_spikes_trained, end=' /')
        else:
            for index in range(self.number_of_working_neurons):
                print(self.working_neurons[index].name, self.working_neurons[index].n_spikes[t], end=' /')
            for index in range(self.number_of_output_neurons):
                print(self.output_neurons[index].name, self.output_neurons[index].n_spikes[t], end=' /')
        print()


    def get_output(self, t):
        output_spikes = []
        for index in range(self.number_of_output_neurons):
            if t == -1:
                output_spikes.append(self.output_neurons[index].n_spikes_trained)
            else:
                output_spikes.append(self.output_neurons[index].n_spikes[t])
        return output_spikes

