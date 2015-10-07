import numpy as np


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
	s = sigmoid(z)
	return s * (1 - s)


def batch(iterable, n=1):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]


class FullyConnectedNeuralNet(object):
	def __init__(self):
		self.layers = []
	
	def create_fully_connected_network(self, network_definition):
		for i in range(1, len(network_definition)):
			number_of_nodes = network_definition[i]
			layer = 2 * np.random.rand(number_of_nodes, network_definition[i - 1] + 1) - 1
			self.layers.append(layer)
	
	def evaluate_network(self, input_data):
		previous_outputs = input_data
		network_activations = []
		for layer in self.layers:
			current_outputs = []
			node_activations = []
			for n in range(len(layer)):
				node = layer[n]
				#assert (len(previous_outputs) == len(node) - 1)
				activation = np.dot([1] + previous_outputs, node)
				node_activations.append(activation)
				current_outputs.append(sigmoid(activation))
				
			network_activations.append(node_activations)
			previous_outputs = current_outputs
		return (previous_outputs, network_activations)
	
	def train(self, train_set, val_set, learning_rate=0.1, epochs=100, minibatch_size=1):
		num_layers = len(self.layers)
		
		for epoch in range(epochs):
			
			for minibatch in batch(train_set, minibatch_size):
				
				average_gradient = []  # Gradient: the PD for each weight of each node of each layer
				for layer in self.layers:
					average_gradient.append(np.zeros(np.shape(layer)))
				
				for (input, expected) in minibatch:
					(result, network_activations) = self.evaluate_network(input)
					
					next_layer_node_error_terms = []
					
					output_error = np.array(result) - np.array(expected)
					for layer_i in range(num_layers - 1, -1, -1):
						
						if layer_i > 0:
							previous_layer_output = np.array(
								[1] + [sigmoid(z) for z in network_activations[layer_i - 1]])
						else:
							previous_layer_output = np.array([1] + input)
						
						num_nodes = len(network_activations[layer_i])
						
						layer_gradient = np.zeros((num_nodes, len(previous_layer_output)))
						
						layer_nodes_error_terms = np.zeros((num_nodes, 1))
						
						for n in range(num_nodes):
							activation_derivative = sigmoid_derivative(network_activations[layer_i][n])
							
							if layer_i == num_layers - 1:
								node_error_term = activation_derivative * output_error[n]
							else:
								# this node appears for 1 weight in subsequent nodes, find it
								node_error_term = activation_derivative * np.dot(np.transpose(self.layers[layer_i + 1])[n],
								                                                 next_layer_node_error_terms)
							
							layer_nodes_error_terms[n] = node_error_term
							
							layer_gradient[n] = node_error_term * previous_layer_output
						
						next_layer_node_error_terms = layer_nodes_error_terms
						average_gradient[layer_i] += layer_gradient / float(minibatch_size)
				
				for l in range(1, len(self.layers)):
					self.layers[l] = self.layers[l] - learning_rate * average_gradient[l]
			
			# Compute Validation Error...
			total_error = 0
			for (x, y) in val_set:
				error = np.array(self.evaluate_network(x)[0]) - np.array(y)
				total_error += np.dot(error, error)
			
			total_error /= float(len(val_set))
			
			percent_done = str(100 * epoch / float(epochs))
			
			message_for_the_impatients = "The machine learning: " + percent_done + "% done... Val Error: " + str(
				total_error) + "."
			
			print(message_for_the_impatients)  # , sep=' ', end='', flush=True)


nn = FullyConnectedNeuralNet()

nn.create_fully_connected_network([1, 10, 10, 10, 1])

PI = 3.14159265359

train_set = []
for l in range(10000):
	x = np.random.rand() * 2 * PI
	train_set.append(([x], [np.sin(x)]))

val_set = []
for l in range(50):
	x = np.random.rand() * 2 * PI
	val_set.append(([x], [np.sin(x)]))

nn.train(train_set, val_set, 1.0, 150, 100)

print("Testing network....")

for l in range(10):
	x = np.random.rand() * 2 * PI
	
	print("x = " + str(x) + ", sin(x)=" + str(np.sin(x)) + ", nn= " + str(nn.evaluate_network([x])[0]))

