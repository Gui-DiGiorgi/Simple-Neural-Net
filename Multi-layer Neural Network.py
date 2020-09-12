import numpy as np
import matplotlib.pyplot as plt
from random import randint

# BNN - Bigger Neural Network (at least compared to my last project)

class BNN():
    def __init__(self, links, values, positive_parameters_initialization = True, n_g_min = -0.5, leak_rate = 0.01,
                 hidden_f = "Leaky ReLU", output_f = "Linear", error_f = "Squared Error", in_norm_f = "Global Normalization",
                 out_norm_f = "Global Normalization"):
        
        import numpy as np
        from random import randint
        
        self.n_g_min = n_g_min
        self.leak_rate = leak_rate
        
        functions = {"Squared Error": BNN.sq_error, "Sigmoid": BNN.sigmoid, "Tanh": BNN.tanh,
                     "ReLU": BNN.ReLU, "Leaky ReLU": self.leaky_ReLU, "Linear": BNN.linear}
        
        functions_der = {"Squared Error": BNN.sq_error_der, "Sigmoid": BNN.sigmoid_der, "Tanh": BNN.tanh_der,
                         "ReLU": BNN.ReLU_der, "Leaky ReLU": self.leaky_ReLU_der, "Linear": BNN.linear_der}
        
        functions_in_norm = {"Global Normalization": BNN.full_normie, "Local Normalization": BNN.single_normie}
        
#         functions_in_post_norm = {"Global Normalization": BNN.post_full_normie, "Local Normalization": BNN.single_normie}
        
        functions_out_norm = {"Global Normalization": BNN.full_normie, "Local Normalization": BNN.hyper_full_normie}
        
        
        
        self.hidden_func = functions[hidden_f]
        self.hidden_func_der = functions_der[hidden_f]
        self.output_func = functions[output_f]
        self.output_func_der = functions_der[output_f]
        self.error_func = functions[error_f]
        self.error_func_der = functions_der[error_f]
        
        
        inputs = []
        outputs = []

        for i in values:
            
            inputs.append(np.array(i[0]))
            outputs.append(np.array(i[1]))
            
        self.inputs_data = functions_in_norm[in_norm_f](inputs, n_g_min)
        self.outputs_data = functions_out_norm[out_norm_f](outputs, n_g_min)
        
        self.in_norm_mode = in_norm_f
        self.out_rev_norm_func = BNN.rev_full_normie
            

        weights = {}
        bias = {}
        usable_derivs = {}
        wei_derivs = {}
        cell_values = {"I":None}

        for i in range(len(links)-1):
            if i == len(links)-2:
                name = "O"
            else:
                name = "H{}".format(i)

            weights[name] = []
            for j in range(links[i]):
                slots = np.random.normal(0, 1/np.sqrt(links[i]), links[i+1])
                if positive_parameters_initialization:
                    slots = abs(slots)
                weights[name].append(slots)
                
            weights[name] = np.array(weights[name])
            bias[name] = np.random.normal(0, 1/np.sqrt(links[i]), links[i+1])
            if positive_parameters_initialization:
                bias[name] = abs(bias[name])

            cell_values[name] = None
            usable_derivs[name] = None
            wei_derivs[name] = None
            
            
        self.weights = weights
        self.bias = bias
        self.cell_values = cell_values
        self.usable_derivs = usable_derivs
        self.wei_derivs = wei_derivs
        
        self.keywords = list(self.cell_values)
        self.keywords_rev = self.keywords.copy()
        self.keywords_rev.reverse()
        
# Beginning of Functions and Functions derivatives

# Squared Error: Classic error function for training
        
    def sq_error(y,z):
        return (1/2)*((y-z)**2)

    def sq_error_der(y,z):
        return (z-y)
    
# Sigmoid: Squishy treatment type function that acts nice, but can make earlier neurons have less impact on
#          the network as a whole, it acts on the 0 to 1 range. The z in its derivative is not the raw number,
#          but the z = sigmoid(t), where t is the true number here

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def sigmoid_der(z):
        return (z*(1-z))
    
# Tanh: Friend of Sigmoid, but it acts on the -1 to 1 range and it has a steeper curve on the -1 to 1 transition

    def tanh(z):
        return (2/(1+np.exp(-2*z)) - 1)

    def tanh_der(z):
        return (1 - z**2)
    
# ReLU: Relevancy treatment type function, it blocks useless neurons, but it can kill most of the network if not
#       careful with the learning rate
    
    def ReLU(x):
        if type(x) == int or type(x) == float:
            return max(0,x)
        elif type(x) == np.ndarray:
            ma = x.copy()
            ma[ma<0] = 0
            return ma

    def ReLU_der(x):
        if type(x) == int or type(x) == float:
            if x>0:
                return 1
            else:
                return 0
        elif type(x) == np.ndarray:
            ma = x.copy()
            ma[ma>0] = 1
            ma[ma<0] = 0
            return ma
        
# Leaky ReLU: Almost like ReLU, except that it doesn't turn negative results into zero, but into very small
#             versions of themselves, preventing death
        
    def leaky_ReLU(self, x, leak = None):
        if leak == None:
            leak = self.leak_rate
        if type(x) == int or type(x) == float:
            return max(0.01*x,x)
        elif type(x) == np.ndarray:
            ma = x.copy()
            ma[ma<0] = ma[ma<0]*leak
            return ma

    def leaky_ReLU_der(self, x, leak = None):
        if leak == None:
            leak = self.leak_rate
        if type(x) == int or type(x) == float:
            if x>0:
                return 1
            else:
                return leakyness
        elif type(x) == np.ndarray:
            ma = x.copy()
            ma[ma>0] = 1
            ma[ma<0] = leak
            return ma
    
# Linear: a final function, similar to ReLU, but it leaves the values as it is, treat it as a non-treatment
#         type function that doesn't do anything with the values it gets from the previous layer

    def linear(x):
        return x

    def linear_der(x):
        if type(x) == np.ndarray:
            ma = x.copy()
            ma[True] = 1
            return ma
    
# End of Functions and Functions derivatives

# Treatments for data

    def full_normie(x, g_min = 0):
        normalization = np.array(x, dtype = float)

        for i in range(len(normalization)):
            normalization[i] = np.array(normalization[i],dtype = float)

        norm_len = len(normalization[0])

        max_value = np.array([-np.inf for i in range(norm_len)],dtype = float)
        min_value = np.array([np.inf for i in range(norm_len)],dtype = float)

        for i in normalization:
            for j in range(norm_len):
                if i[j]>max_value[j]:
                    max_value[j] = i[j]
                if i[j]<min_value[j]:
                    min_value[j] = i[j]

        max_min_const = max_value - min_value
            
        for i in range(len(max_min_const)):
            if max_min_const[i] == 0:
                max_min_const[i] = max_value[i]

        for i in range(len(normalization)):
            normalization[i] = (normalization[i] - min_value)/max_min_const + g_min

        return {"Normie Array": normalization,"Min Value": min_value, "Max Min Const": max_min_const}
    
    def hyper_full_normie(x, g_min = 0):

        normalization = np.array(x, dtype = float)

        max_v = np.amax(normalization[0])
        min_v = np.amin(normalization[0])

        for i in range(len(normalization)):
            max_v = max(max_v,np.amax(normalization[i]))
            min_v = min(min_v,np.amin(normalization[i]))

        max_min_v = max_v - min_v

        if max_min_v == 0:
            max_min_v = max_v

        for i in range(len(normalization)):
            normalization[i] = (normalization[i] - min_v)/max_min_v + g_min

        return {"Normie Array": normalization,"Min Value": min_v, "Max Min Const": max_min_v}
    
    def post_full_normie(values, min_value, max_min_const, g_min = 0):
        v = np.array(values)
        v = (v - min_value)/max_min_const + g_min
        return v

    def rev_full_normie(values, min_value, max_min_const, g_min = 0):
        v = np.array(values)
        v = (v - g_min)*max_min_const + min_value
        return v
    
    def single_normie(x, g_min = 0):
        normalization = np.array(x, dtype = float)

        max_v = []
        min_v = []
        max_min_v = []

        for i in range(len(normalization)):
            max_v.append(np.amax(normalization[i]))
            min_v.append(np.amin(normalization[i]))
            max_min_v.append(max_v[i]-min_v[i])

        max_v = np.array(max_v)
        min_v = np.array(min_v)
        max_min_v = np.array(max_min_v)

        for i in range(len(max_min_v)):
            if max_min_v[i] == 0:
                max_min_v[i] = max_v[i]

        for i in range(len(normalization)):
            normalization[i] = (normalization[i] - min_v[i])/max_min_v[i] + g_min

        return {"Normie Array": normalization,"Min Value": min_v, "Max Min Const": max_min_v}
    
    def post_single_normie(values, min_value, max_min_const, g_min = 0):
        v = np.array(values)
        min_v = np.array(min_value)
        max_min_v = np.array(max_min_const)
        for i in range(len(v)):
            v[i] = (v[i] - min_v[i])/max_min_v[i] + g_min
        return v

    def rev_single_normie(values, min_value, max_min_const, g_min = 0):
        v = np.array(values)
        min_v = np.array(min_value)
        max_min_v = np.array(max_min_const)
        for i in range(len(v)):
            v[i] = (v[i] - g_min) * max_min_v[i] + min_v[i]
        return v
    
    def shuffle(self, shuffles):

        luckyyy_in = []
        luckyyy_out = []

        for i in range(shuffles):

            magic_numbaa = randint(0,len(self.inputs_data["Normie Array"])-1)

            luckyyy_in.append(self.inputs_data["Normie Array"][magic_numbaa])
            luckyyy_out.append(self.outputs_data["Normie Array"][magic_numbaa])

        luckyyy_in = np.array(luckyyy_in)
        luckyyy_out = np.array(luckyyy_out)

        return {"I":luckyyy_in, "O":luckyyy_out}
                
# Insert a new value and let the machine guess
    
    def predict(self, data):
        
        if self.in_norm_mode == "Global Normalization":
        
            self.cell_values["I"] = BNN.post_full_normie(data,self.inputs_data["Min Value"],self.inputs_data["Max Min Const"],
                                                  self.n_g_min)
            
        elif self.in_norm_mode == "Local Normalization":
            
            self.cell_values["I"] = BNN.single_normie(data,self.n_g_min)["Normie Array"]
        
        i = -1
    
        for i in range(len(self.keywords)-2):
            self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                          self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
            self.cell_values[self.keywords[i+1]] = self.hidden_func(self.cell_values[self.keywords[i+1]])
            
        if i >= -1:
            i += 1
            
        self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                      self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
        self.cell_values[self.keywords[i+1]] = self.output_func(self.cell_values[self.keywords[i+1]])
        
        output_treatment = self.out_rev_norm_func(self.cell_values["O"],self.outputs_data["Min Value"],
                                        self.outputs_data["Max Min Const"],self.n_g_min)

        return output_treatment
        
# Training Bits
    
    def training_predict(self, data):
            
        self.cell_values["I"] = data
        
        i = -1
    
        for i in range(len(self.keywords)-2):
            self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                          self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
            self.cell_values[self.keywords[i+1]] = self.hidden_func(self.cell_values[self.keywords[i+1]])
            
        if i >= -1:
            i += 1
            
        self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                      self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
        self.cell_values[self.keywords[i+1]] = self.output_func(self.cell_values[self.keywords[i+1]])
                
    def settings_training(self, normie_outputs):
        
        error_d = self.error_func_der(normie_outputs, self.cell_values["O"])
        squishy_d = self.output_func_der(self.cell_values["O"])
        self.usable_derivs["O"] = error_d*squishy_d
        
        for i in range(1,len(self.keywords_rev)-1):
            
            error_d = np.dot(self.usable_derivs[self.keywords_rev[i-1]],self.weights[self.keywords_rev[i-1]].T)
            squishy_d = self.hidden_func_der(self.cell_values[self.keywords_rev[i]])
            self.usable_derivs[self.keywords_rev[i]] = error_d*squishy_d
            
        for k in range(len(self.keywords_rev)-1):
            
            self.wei_derivs[self.keywords_rev[k]] = np.array([np.array(np.dot(np.matrix(i).T,[j])) for i,j in \
                                                    zip(self.cell_values[self.keywords_rev[k+1]],\
                                                        self.usable_derivs[self.keywords_rev[k]])])
            
    def e_analysis_training_predict(self, data):
            
        self.cell_values["I"] = data
        
        i = -1
    
        for i in range(len(self.keywords)-2):
            self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                          self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
            self.cell_values[self.keywords[i+1]] = self.hidden_func(self.cell_values[self.keywords[i+1]])
            
        if i >= -1:
            i += 1
            
        self.cell_values[self.keywords[i+1]] = np.dot(self.cell_values[self.keywords[i]],
                                                      self.weights[self.keywords[i+1]]) + self.bias[self.keywords[i+1]]
        self.cell_values[self.keywords[i+1]] = self.output_func(self.cell_values[self.keywords[i+1]])
            
        return self.cell_values["O"]
            
    def avg_growth(x):
    
        size = len(x)

        if type(x) == np.ndarray:

            initial = x[0].copy()

            for i in range(1,size):
                add = x[i].copy()
                initial += add

            return initial/size

        else:

            initial = np.array(x[0].copy())

            for i in range(1,size):
                add = np.array(x[i].copy())
                initial += add

            return initial/size

    def error(self):
        
        pred = self.e_analysis_training_predict(self.inputs_data["Normie Array"])
        errors = self.error_func(self.outputs_data["Normie Array"],pred)
        
        t_error = []
        for i in errors:
            t_error.append(sum(i))
            
        return sum(np.array(t_error))/len(t_error)
    
    def training(self, learning_rate, n_shuffle_members, n_shuffles, t_cycles_per_shuffle):
        
        if n_shuffle_members > len(self.inputs_data["Normie Array"]):
            n_shuffle_members = len(self.inputs_data["Normie Array"])
                
        c_shuffles = 0
                
        while c_shuffles < n_shuffles:
        
            training_times = 0

            shuffle_gang = self.shuffle(n_shuffle_members)

            while training_times < t_cycles_per_shuffle:

                self.training_predict(shuffle_gang["I"])
                self.settings_training(shuffle_gang["O"])
                
                for i in range(1,len(self.keywords)):
                    self.weights[self.keywords[i]] -= learning_rate*BNN.avg_growth(self.wei_derivs[self.keywords[i]])
                    self.bias[self.keywords[i]] -= learning_rate*BNN.avg_growth(self.usable_derivs[self.keywords[i]])
                    
                training_times += 1
                    
            c_shuffles += 1
            
            print("Cycle {1}: Currently with {0} of error".format(self.error(), c_shuffles))
            
        print("Done!")
