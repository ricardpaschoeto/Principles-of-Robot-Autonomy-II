import numpy as np
import tensorflow as tf
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        initializer = tf.keras.initializers.GlorotUniform(seed=0)
        # Input Network units
        in_hidden_1 = 64
        in_hidden_2 = 32
        in_hidden_3 = 16

        # Branch Network 01 units
        branch1_hidden_1 = 16
        branch1_hidden_2 = 8

        # Branch Network 02 units
        branch2_hidden_1 = 16
        branch2_hidden_2 = 8

        # Branch Network 03 units
        branch3_hidden_1 = 16
        branch3_hidden_2 = 8

       # Input Network layer 1
        self.input_layer1_w = tf.Variable(initializer(shape=(in_size, in_hidden_1)))
        self.input_layer1_bias = tf.Variable(tf.zeros([in_hidden_1]))

       # Input Network layer 2
        self.input_layer2_w = tf.Variable(initializer(shape=(in_hidden_1, in_hidden_2)))
        self.input_layer2_bias = tf.Variable(tf.zeros([in_hidden_2]))

       # Input Network layer 3
        self.input_layer3_w = tf.Variable(initializer(shape=(in_hidden_2, in_hidden_3)))
        self.input_layer3_bias = tf.Variable(tf.zeros([in_hidden_3]))

        # Branch Network 1 layer 1
        self.branch1_layer1_w = tf.Variable(initializer(shape=(in_hidden_3, branch1_hidden_1)))
        self.branch1_layer1_bias = tf.Variable(tf.zeros([branch1_hidden_1]))

        # Branch Network 1 layer 2
        self.branch1_layer2_w = tf.Variable(initializer(shape=(branch1_hidden_1, branch1_hidden_2)))
        self.branch1_layer2_bias = tf.Variable(tf.zeros([branch1_hidden_2]))

        # Branch Network 1 output
        self.branch1_output_w = tf.Variable(initializer(shape=(branch1_hidden_2, out_size)))
        self.branch1_output_bias = tf.Variable(tf.zeros([out_size]))

        # Branch Network 2 layer 1
        self.branch2_layer1_w = tf.Variable(initializer(shape=(in_hidden_3, branch2_hidden_1)))
        self.branch2_layer1_bias = tf.Variable(tf.zeros([branch2_hidden_1]))

        # Branch Network 2 layer 2
        self.branch2_layer2_w = tf.Variable(initializer(shape=(branch2_hidden_1, branch2_hidden_2)))
        self.branch2_layer2_bias = tf.Variable(tf.zeros([branch2_hidden_2]))

        # Branch Network 2 output
        self.branch2_output_w = tf.Variable(initializer(shape=(branch2_hidden_2, out_size)))
        self.branch2_output_bias = tf.Variable(tf.zeros([out_size]))

        # Branch Network 3 layer 1
        self.branch3_layer1_w = tf.Variable(initializer(shape=(in_hidden_3, branch3_hidden_1)))
        self.branch3_layer1_bias = tf.Variable(tf.zeros([branch3_hidden_1]))

        # Branch Network 3 layer 2
        self.branch3_layer2_w = tf.Variable(initializer(shape=(branch3_hidden_1, branch3_hidden_2)))
        self.branch3_layer2_bias = tf.Variable(tf.zeros([branch3_hidden_2]))

        # Branch Network 3 output
        self.branch3_output_w = tf.Variable(initializer(shape=(branch3_hidden_2, out_size)))
        self.branch3_output_bias = tf.Variable(tf.zeros([out_size]))
        ########## Your code ends here ##########

    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.int8)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (?, |O|) tensor that keeps a batch of observations
        # - u is a (?, 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # FYI: For the intersection scenario, u=0 means the goal is to turn left, u=1 straight, and u=2 right. 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found tf.math.equal and tf.cast useful. This is not necessarily a requirement though.

        # Input layer
        in_layer1_out = tf.nn.tanh(tf.add(tf.matmul(x, self.input_layer1_w), self.input_layer1_bias))
        in_layer2_out = tf.nn.tanh(tf.add(tf.matmul(in_layer1_out, self.input_layer2_w), self.input_layer2_bias))
        in_layer3_out = tf.nn.tanh(tf.add(tf.matmul(in_layer2_out, self.input_layer3_w), self.input_layer3_bias))

        # Branch 1
        branch1_h1 = tf.nn.tanh(tf.add(tf.matmul(in_layer3_out, self.branch1_layer1_w), self.branch1_layer1_bias))
        branch1_h2 = tf.nn.tanh(tf.add(tf.matmul(branch1_h1, self.branch1_layer2_w), self.branch1_layer2_bias))
        y_est1 = tf.add(tf.matmul(branch1_h2, self.branch1_output_w), self.branch1_output_bias)

        # Branch 2
        branch2_h1 = tf.nn.tanh(tf.add(tf.matmul(in_layer3_out, self.branch2_layer1_w), self.branch2_layer1_bias))
        branch2_h2 = tf.nn.tanh(tf.add(tf.matmul(branch2_h1, self.branch2_layer2_w), self.branch2_layer2_bias))
        y_est2 = tf.add(tf.matmul(branch2_h2, self.branch2_output_w), self.branch2_output_bias)

        # Branch 3
        branch3_h1 = tf.nn.tanh(tf.add(tf.matmul(in_layer3_out, self.branch3_layer1_w), self.branch3_layer1_bias))
        branch3_h2 = tf.nn.tanh(tf.add(tf.matmul(branch3_h1, self.branch3_layer2_w), self.branch3_layer2_bias))
        y_est3 = tf.add(tf.matmul(branch3_h2, self.branch3_output_w), self.branch3_output_bias)

        action_final = y_est1 * tf.cast(tf.equal(u, 0), dtype = tf.float32) + y_est2 * tf.cast(tf.equal(u, 1), dtype = tf.float32) + y_est3 * tf.cast(tf.equal(u, 2), dtype = tf.float32)

        return action_final
        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    weight_1 = 1.0
    weight_2 = 1.0

    l_steering = tf.math.sqrt(tf.nn.l2_loss((y_est[:, 0] - y[:, 0]))) * weight_1
    l_throttle = tf.math.sqrt(tf.nn.l2_loss((y_est[:, 1] - y[:, 1]))) * weight_2

    l = l_steering + l_throttle

    return l
    ########## Your code ends here ##########
   

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y, u):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model (note both x and u are inputs now)
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        with tf.GradientTape() as tape:
            y_est = nn_model(x, u)
            current_loss = loss(y_est, y)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y, u in train_data:
            train_step(x, y, u)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'], data['u_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = 'all'
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
