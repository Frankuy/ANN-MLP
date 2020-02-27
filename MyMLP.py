import numpy as np

def MyMLP(data_x, data_y, mini_data=1, learning_rate=0.4, hidden_layer_unit=100, output_layer_unit=2, max_epoch=100000):
    '''
    Using mini-batch gradient descent with backpropagation algorithm
    function sigmoid as activation.
    output layer unit should same as number of unique data in data_y

    RETURN model (weight of the MLP)
    '''

    ### CHECK DATA IS EMPTY
    if len(data_x) == 0:
        print("Your data is empty")
        return

    # Initilize weight and error
    weight = initWeight(len(data_x[0]), hidden_layer_unit, output_layer_unit)
    error = 999999
    epoch = 1
    while (error > 0.05 and epoch != max_epoch):
        # Initialize delta weight
        delta_weights = initDeltaWeight(len(data_x[0]), hidden_layer_unit, output_layer_unit)
        count_processed_data = 0
        final_output = 0

        print('EPOCH #',epoch)
        print('ERROR', error)

        for idx in range(0, len(data_x)):
            # Feed Forward Phase
            ## Count all output
            ### HIDDEN LAYER
            output_for_hidden = []
            for i in range(0, hidden_layer_unit):
                output_for_hidden.append(sigmoid(nett(data_x[idx], weight['hidden-input'][i])))
            
            ### OUTPUT LAYER
            output = []
            for idx_output in range(0, output_layer_unit):
                local_output = 0
                for idx_hidden in range(0, len(output_for_hidden)):
                    local_output += weight['output-hidden'][idx_output][idx_hidden] * output_for_hidden[idx_hidden]
                local_output += weight['output-hidden'][idx_output][len(output_for_hidden)] * 1 # ADD BIAS
                local_output = sigmoid(local_output)
                output.append(local_output)

            # Backward Phase
            ## Count delta
            ### OUTPUT LAYER
            delta_output = []
            for i in range(0, output_layer_unit):
                if (data_y[idx] == i): # Class value is equal to output i-th, which means the actual target is 1
                    delta_output.append(deltaO(output[i], 1))
                else:
                    delta_output.append(deltaO(output[i], 0))

            ### HIDDEN LAYER
            delta_hidden = []
            for idx_hidden in range(0, hidden_layer_unit):
                delta_hidden.append(deltaH(output_for_hidden[idx_hidden], weight['output-hidden'], delta_output, idx_hidden))

            # Weight Changer
            ## Update delta weight
            ### HIDDEN-INPUT LAYER
            for i in range(0, len(delta_weights['hidden-input'])):
                # Hidden i-th
                for j in range(0, len(delta_weights['hidden-input'][i])):
                    # Input j-th to Hidden i-th
                    if (j == len(delta_weights['hidden-input'][i]) - 1): # BIAS
                        delta_weights['hidden-input'][i][j] += deltaWeight(learning_rate, delta_hidden[i], 1)    
                    else: 
                        delta_weights['hidden-input'][i][j] += deltaWeight(learning_rate, delta_hidden[i], data_x[idx][j])

            ### OUTPUT-HIDDEN LAYER
            for i in range(0, len(delta_weights['output-hidden'])):
                # Output ke i-th 
                for j in range(0, len(delta_weights['output-hidden'][i])):
                    # Hidden j-th to Output i-th
                    if (j == len(delta_weights['output-hidden'][i]) - 1): # BIAS
                        delta_weights['output-hidden'][i][j] += deltaWeight(learning_rate, delta_output[i], 1)
                    else:
                        delta_weights['output-hidden'][i][j] += deltaWeight(learning_rate, delta_output[i], output_for_hidden[j])
            
            count_processed_data += 1
            if idx == len(data_x) - 1 or count_processed_data == mini_data: # Already in last data OR in minimal data for mini-batch
                count_processed_data = 0
                ## Use delta weight to update weight
                ### OUTPUT-HIDDEN
                for i in range(0, len(weight['output-hidden'])):
                    for j in range(0, len(weight['output-hidden'][i])):
                        weight['output-hidden'][i][j] += delta_weights['output-hidden'][i][j]
                ### HIDDEN-INPUT
                for i in range(0, len(weight['hidden-input'])):
                    for j in range(0, len(weight['hidden-input'][i])):
                        weight['hidden-input'][i][j] += delta_weights['hidden-input'][i][j]

                delta_weights = initDeltaWeight(len(data_x[0]), hidden_layer_unit, output_layer_unit)
                if idx == len(data_x) - 1:
                    final_output = output
        # END FOR (ALL DATA HAVE BEEN PROCESSED)
        epoch += 1
        error = calculateError(final_output, data_y[len(data_y) - 1])

    return weight

def initWeight(number_input_unit, number_hidden_unit, number_output_unit):
    '''
    Initialize weight by random number in range [0..1]

    the structure weight is divided by 2:
    1. Hidden-input
    EX. [[0.4, 0.6],[0.2, 0.3]] which means [0.4, 0.6] is weight for hidden-0 that 0.4 is from input-0 and 0.6 from input-1
    2. Output-hidden
    EX. [[0.4, 0.5], [0.3, 0.7]] which means 0.4 is from hidden-0 to output-0 and 0.3 is from hidden-0 to output 1
    '''
    weight = dict()
    weight_hidden_input = list()
    for i in range(0, number_hidden_unit):
        local_weight = list(np.random.uniform(size=number_input_unit))
        local_weight.append(0) # add 1 zero weight for bias
        weight_hidden_input.append(local_weight)
    weight['hidden-input'] = weight_hidden_input

    weight_output_hidden = list()
    for i in range(0, number_output_unit):
        local_weight = list(np.random.uniform(size=number_hidden_unit))
        local_weight.append(0) # add 1 zero weight for bias
        weight_output_hidden.append(local_weight)
    weight['output-hidden'] = weight_output_hidden

    return weight

def initDeltaWeight(number_input_unit, number_hidden_unit, number_output_unit):
    delta_weight = dict()
    weight_hidden_input = list()
    for i in range(0, number_hidden_unit):
        local_weight = list(np.zeros(shape=number_input_unit + 1)) # add 1 zero weight for bias
        weight_hidden_input.append(local_weight)
    delta_weight['hidden-input'] = weight_hidden_input

    weight_output_hidden = list()
    for i in range(0, number_output_unit):
        local_weight = list(np.zeros(shape=number_hidden_unit + 1)) # add 1 zero weight for bias
        weight_output_hidden.append(local_weight)
    delta_weight['output-hidden'] = weight_output_hidden

    return delta_weight
        

def nett(data_x, weight):
    '''
    Add all data_x multiply weight

    EX. 
    var data_x = [3,2,5]
    var weight = [0.3, 0.1, 0.5, 0.7] # 0.7 is for bias

    nett(data_x, weight) = 3*0.3 + 2*0.1 + 5*0.5 + 0.7 = 4.3 
    '''
    i = 0
    jumlah = 0
    for x in data_x:
        jumlah += x * weight[i]
        i += 1
    
    jumlah += weight[i]

    return jumlah

def sigmoid(nett):
    '''
    Return the result of activation sigmoid function from nett

    EX.
    var nett = 3.6

    sigmoid(nett) = 1/(1+e^-nett) = 1/(1+e^(-3.6)) = 0.9734
    '''
    return 1/(1+math.e**(-nett))

def deltaO(output, target):
    '''
    Count delta of an output layer unit

    EX.
    deltaO(0.7, 0.8) = 0.7*(1-0.7)*(0.8-0.7)
    '''
    return output*(1-output)*(target-output)

def deltaH(output, weight, deltaO, index_hidden):
    '''
    Count delta of a hidden layer unit

    EX.
    deltaH = output*(1-output)*sigma
    where sigma = sum(weight[index_output][index_hidden]*deltaO[index_output])
    '''
    sigma = 0
    for idx_output in range(0, len(deltaO)):
        sigma += weight[idx_output][index_hidden]*deltaO[idx_output]

    return output*(1-output)*sigma

def deltaWeight(learning_rate, delta, x):
    '''
    Count delta weight using learning_rate*delta*x
    '''
    return learning_rate*delta*x

def calculateError(output, target):
    '''
    Error for determining terminate state

    Error = 1/2(target-output)^2
    '''
    total_error = 0
    for idx_output in range(0, len(output)):
        if (target == idx_output):
            total_error += ((1-output[idx_output])**2)/2
        else:
            total_error += ((0-output[idx_output])**2)/2

    return total_error

def predict(model, data_x):
    '''
    Predict the output using model that contains weight
    '''
    pass

##### TESTING #####
from sklearn.datasets import load_iris
import math

data = load_iris().data
target = load_iris().target
number_unique_output = len(np.unique(target))

weight = MyMLP(data, target, learning_rate=0.4, mini_data=20, hidden_layer_unit=4, output_layer_unit=number_unique_output, max_epoch=100000)