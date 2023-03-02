import numpy as np
import argparse
import tensorflow as tf
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        initializer = tf.keras.initializers.GlorotUniform(seed=0)
        self.h0 = tf.keras.layers.Dense(units=in_size, activation='relu', kernel_initializer=initializer)
        self.h1 = tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer=initializer)
        self.h2 = tf.keras.layers.Dropout(0.2)
        self.h3 = tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer=initializer)
        self.h4 = tf.keras.layers.Dropout(0.2)
        self.h5 = tf.keras.layers.Dense(name='output', units=out_size, activation='sigmoid', kernel_initializer=initializer)     
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?,|O|) tensor that keeps a batch of observations
        x_ = self.h0(x)
        x_ = self.h1(x_)
        x_ = self.h2(x_)
        x_ = self.h3(x_)
        x_ = self.h4(x_)
        action_pred = self.h5(x_)

        return action_pred
        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    pen_steering = 1.0
    pen_throttle = 1.2
    l = tf.math.sqrt(tf.nn.l2_loss(pen_steering * (y_est[0] - y[0]) + pen_throttle * (y_est[1] - y[1])))

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
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        #nn_model.optimizer = optimizer
        with tf.GradientTape() as tape:
            y_est = nn_model(x)
            current_loss = loss(y_est, y)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        tf.keras.optimizers.Optimizer.apply_gradients(optimizer, zip(grads, nn_model.trainable_variables))
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
