def MyMLP(data_x, data_y, mini_data=1, learning_rate=0.1, hidden_layer_unit=1):
    '''
    Using mini-batch gradient descent with backpropagation algorithm
    function sigmoid as activation.

    RETURN model (weight of the MLP)
    '''

    ### CHECK DATA IS EMPTY
    if len(data_x) == 0:
        print("Your data is empty")
        return

    count_processed_data = 0
    weight = initWeight(len(data_x[0]), hidden_layer_unit)

    for idx in range(0, len(data_x)):
        # Feed Forward Phase
        ## Count all output
        ### HIDDEN LAYER
        output_for_hidden = []
        for i in range(0, hidden_layer_unit):
            output_for_hidden.append(sigmoid(nett(data_x[idx], weight['hidden-input'][i])))
        
        ### OUTPUT LAYER
        output = 0
        for idx_hidden in range(0, len(output_for_hidden)):
            output += weight['output-hidden'][idx_hidden] * output_for_hidden[idx_hidden]
        output += weight['output-hidden'][len(output_for_hidden)] * 1; # ADD BIAS
        output = sigmoid(output)

        # Backward Phase
        ## Count delta
        ### OUTPUT LAYER
        ### HIDDEN LAYER

        # Weight Changer
        ## Change weight based on delta
        ### Update delta weight
        # DELTA_WEIGHT = DELTA_WEIGHT + deltaWeight() 

        count_processed_data += 1
        if idx == len(data_x) - 1 or count_processed_data == mini_data: # Already in last data OR in minimal data for mini-batch
            count_processed_data = 0
            ### Use delta weight to update weight
            # WEIGHT = updateWeight(WEIGHT, DELTA_WEIGHT)
            ### Reset delta weight
            # DELTA WEIGHT = [0,0,0,0]

    return weight

def initWeight(number_input_unit, number_hidden_unit):
    '''
    Initialize weight with 0

    the structure weight is divided by 2:
    1. Hidden-input
    EX. [[0.4, 0.6],[0.2, 0.3]] which means [0.4, 0.6] is weight for hidden-0 that 0.4 is from input-0 and 0.6 from input-1
    2. Output-hidden
    EX. [0.4, 0.5] which means 0.4 is from hidden-0 and 0.5 is from hidden-1
    '''
    weight = dict()
    weight_hidden_input = [0 for i in range(0, number_input_unit + 1)] # + 1 for bias
    weight['hidden-input'] = [weight_hidden_input for i in range(0, number_hidden_unit)]
    weight_output_hidden = [0 for i in range(0, number_hidden_unit + 1)] # + 1 for bias
    weight['output-hidden'] = weight_output_hidden

    return weight
        

def nett(data_x, weight):
    '''
    Add all data_x multiply weight

    EX. 
    var data_x = [3,2,5]
    var weight = [0.3, 0.1, 0.5, 0.7] # 0.7 is for bias

    nett(data_x, weight) = 3*0.3 + 2*0.1 + 5*0.5 + 0.7 = 4.3 
    '''
    return 0

def sigmoid(nett):
    '''
    Return the result of activation sigmoid function from nett

    EX.
    var nett = 3.6

    sigmoid(nett) = 1/(1+e^-nett) = 1/(1+e^(-3.6)) = 0.9734
    '''
    return 0

def deltaO(output, target):
    '''
    Count delta of an output layer unit

    EX.
    deltaO(0.7, 0.8) = 0.7*(1-0.7)*(0.8-0.7)
    '''
    pass

def deltaH(output, list_deltaO):
    '''
    Count delta of an hidden layer unit

    EX.
    
    '''
    pass

def deltaWeight(learning_rate, delta, x):
    '''
    Count delta weight using learning_rate*delta*x
    '''
    pass

def calculateError(output, target):
    '''
    Error for determining terminate state

    Error = 1/2(target-output)^2
    '''
    pass

def predict(model, data_x):
    '''
    Predict the output using model that contains weight
    '''
    pass

def updateWeight(weight, delta_weight):
    '''
    Update current weight added with delta_weight
    '''
    pass

##### TESTING #####
from sklearn.datasets import load_iris

data = load_iris().data
target = load_iris().target

MyMLP(data, target, 10, hidden_layer_unit=2)
