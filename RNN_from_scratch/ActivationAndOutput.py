#this file has our activation function, our output logistic wrapper, and 
#their derivatives. used for the rnn node maths
import numpy as np

#set of functions use for outputting machine outputs and getting error
class Softmax:

    #get the outputs in 0-1 probs 
    #param machine_output   : the output of our machine to scale between 0-1
    #return                 : the outputs scaled between 0-1
    def scale(machine_output):
        exp_scores  = np.exp( machine_output )
        return      exp_scores / np.sum( exp_scores )

    #get cross entropy loss - just used for output info, not for deriving loss
    #param machine_output   : the raw output of our machine 
    #param correct_output   : what the machine should have predicted
    #return                 : cross entropy loss ( a single number for loss )
    def cross_entropy(machine_output, correct_output):
        eps = 1e-9
        machine_output = Softmax.scale( machine_output )
        return -np.sum(correct_output*np.log(machine_output+eps))\
                    /len(correct_output)

    #get the difference between our output and the correct output
    #param machine_output   : the raw output of our machine (not softmaxed)
    #param correct_output   : the correct output to get error on
    #return                 : our outputs but correct one-hot idx is -1 for err 
    def raw_err(machine_output, correct_output):
        probs = Softmax.scale(machine_output)
        return probs - correct_output

#our activation function and it's derivative
class Tanh: 

    #param x : the input to get tanh on (wrapper that just calls numpy)
    def Tanh( x ):
        return np.tanh(x)

    #param x : the input to get tanh derivative on
    def Tanh_derivative( x ):
        return 1-(Tanh.Tanh(x)**2)
