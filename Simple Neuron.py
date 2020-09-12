# Here's a class to find the weight and base values for a simple neural network providing it training data

import numpy
import random
import math

class NN_equation_type(object):
    
    # variable_n is the number of variables/inputs
    # output_n is the number of answers/outputs
    # variable_list is a list of lists, with all the variables in all cases
    # answer_list is a list of lists, with all the expected answers in all cases
    # variable_names is a list with the name for each variable
    # pred_names is a list with the name for each prediction
    # treat_type is the kind of treatment it will have on the predicted answer
    # its default is 0, it's not going to treat the answer
    # there are other functions, such as ReLU and sigmoid
    # more info on the treatment functions are found above the predict function
    
    def __init__(self, variable_n, answer_n, variable_list, answer_list, variable_names=None, answer_names=None,treat_type=0):
        
        # The variables will have a label detailing what each variable is for
        # In case of testing, the name will be named "a","b","c"... and so on
        # Same for predictions, but those will use numbers
        
        if variable_names==None:
            
            new_names = NN_equation_type.naming(variable_n)
            
            variable_names=[]
            for i in new_names:
                variable_names.append(i)
        
        if answer_names==None:
            answer_names=[]
            for i in range(answer_n):
                answer_names.append(i)
                
        # The number of names and expected variables will be checked here
    
        if len(variable_names)==variable_n:
            self.variable_names = variable_names
            variable_dict = {}
            for i, j in zip(variable_names,range(variable_n)):
                variable_dict[i] = j
            self.variable_dict = variable_dict
        else:
            print("Your input naming does not match in the system")
       
        # The number of names and expected answers will be checked here
    
        if len(answer_names)==answer_n:
            self.answer_names = answer_names
            answer_dict = {}
            for i, j in zip(answer_names,range(answer_n)):
                answer_dict[i] = j
            self.answer_dict = answer_dict
        else:
            print("Your output naming does not match in the system")
            

        # The random values for each relation will be created here
        
        self.variable_n = variable_n
        self.answer_n = answer_n
        settings_list = []
        
        for i in range(answer_n):
            settings_list.append(list())
            for j in range(variable_n+1):
                settings_list[i].append(numpy.random.randn())
            
        self.settings_list = settings_list
            
            
        self.variable_list = variable_list
        self.ok1 = True
        
        for var in self.variable_list:
            if len(var)!=self.variable_n:
                print("Variables are missing!")
                self.ok1 = False
                
        self.answer_list = answer_list
        self.ok2 = True
        
        for var in self.answer_list:
            if len(var)!=self.answer_n:
                print("Answers are missing!")
                self.ok2 = False
                
        if len(self.answer_list)!=len(self.variable_list):
            if len(self.answer_list)>len(self.variable_list):
                print("There are more answers than variable sets")
            else:
                print("There are more variable sets than answers")
            self.ok2 = False
            
        self.treat_type=treat_type
            
    def naming(names_n):

        alp = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

        names = []

        num = names_n
        fk_num = num
        loops=0

        while fk_num>=1:
            fk_num/=len(alp.copy())
            loops+=1

        resources=[]
        for nam in range(loops):
            resources+=[alp.copy()]


        for n in range(num):
            title=""
            for i in range(loops):
                for ind,r in zip(range(loops-1),resources):
                    if r==[]:
                        resources[ind]=alp.copy()
                        resources[ind+1].remove(resources[ind+1][0])

                title+=resources[loops-1-i][0]

            resources[0].remove(resources[0][0])
            names.append(title)

        return names   
    
    def rel_dict_naming(var,ans,rel):
    
        rel_dict={}

        pseudo_var = var.copy()
        pseudo_var.append("bias")


        for a in ans:
            naming_part1 = "->"+a
            for v in pseudo_var:
                naming_part2 = v+naming_part1
                rel_dict[naming_part2]=rel[ans.index(a)][pseudo_var.index(v)]

        return  rel_dict
    
    # Next there are the possible functions to use to treat the prediction sum
    
    # passes the number as it is, good for a raw equation
    def untreated(x):
        return x
    
    # passes the number as a yes (with 1) or no (with a 0) answer, some uncertainty in between might be present
    def sigmoid(x):
        return 1/(1 + numpy.exp(-x))
    
    # passes the number as it is if positive, 0 if it is negative:
    def ReLU(x):
        return max(0,x)
    
    
    def predict(self):
        
        # treatments will select which type of function to apply for the predicted value
        
        treatments = {0:NN_equation_type.untreated, 1:NN_equation_type.sigmoid, 2:NN_equation_type.ReLU}
        
        treat_func = treatments[self.treat_type]
        
        pred_list = []
        
        if self.ok1:
            for data_size in range(len(self.variable_list)):
                pred = []
                for i in range(self.answer_n):
                    p = 0
                    for j,k in zip(self.settings_list[i],self.variable_list[data_size]):
                        p += j*k
                    p += self.settings_list[i][-1]
                    pred.append(treat_func(p))
                pred_list.append(pred)
            return pred_list
    
    # the next functions work in order to give the error
    # the first was originally going to be the error function, but I decided to separate them as I was getting confused...
    # ...with all the variables going around here, so the first is the generalized squared difference function
    # the second is just the first, but working with the lists I wants
    
    def avg_sqrd_diff(guessed,expected):

        diffs = []
        base = []
        for i in range(len(guessed[0])):
            base += [0]
        diffs+=base

        for j in range(len(guessed[0])):
            sd=0
            for k in range(len(guessed)):
                g = guessed[k][j]
                e = expected[k][j]
                sd+=(g-e)**2
            diffs[j]=sd
            diffs[j]/=len(guessed)
            
        return diffs
    
    def avg_error_func(self):
                
        if self.ok2:
            pred_list=self.predict()
            error_list = NN_equation_type.avg_sqrd_diff(pred_list,self.answer_list)
            
            return error_list
        
    # Next there are the derivatives of the functions used to treat the prediction sum
    
    # derivative of the untreated fucntion
    def d_untreated(x):
        der = 1
        return der
    
    # passes the number as a yes (with 1) or no (with a 0) answer, some uncertainty in between might be present
    def d_sigmoid(x):
        der = (NN_equation_type.sigmoid(x))*(1-NN_equation_type.sigmoid(x))
        return der
    
    # passes the number as it is if positive, 0 if it is negative:
    def d_ReLU(x):
        if x>0:
            der = 1
        else:
            der = 0
               
        return der
               
    # The universal derivative for the weights, it returns the derivative value for each one as a list of lists
    def univ_sqrd_diff_w_der(variables, answers, settings, func, d_func):

        deriv_frost_list = []

        for m in range(len(settings)):
            frost_list=[]
            for k in range(len(settings[0])-1):
                frost_force = 0
                for j in range(len(variables)):
                    frost_deriv = 0
                    frost1 = 0
                    frost2 = 0
                    for i in range(len(variables[0])):
                        frost1 += settings[m][i]*variables[j][i]
                    frost1+=settings[m][-1]
                    frost2 = func(frost1)
                    frost_deriv = 2*(frost2-answers[j][m])
                    frost_deriv *= d_func(frost1)
                    frost_deriv *= variables[j][k]
                    frost_force+=frost_deriv
                frost_list.append(frost_force)
            deriv_frost_list.append(frost_list)

        return deriv_frost_list
               
    # The universal derivative for the bias, it returns the derivative value for each one as a list
    def univ_sqrd_diff_b_der(variables, answers, settings, func, d_func):

        deriv_flame_list = []

        for m in range(len(settings)):
            flame_force=0
            for j in range(len(variables)):
                flame_deriv = 0
                flame1 = 0
                flame2 = 0
                for i in range(len(variables[0])):
                    flame1 += settings[m][i]*variables[j][i]
                flame1+=settings[m][-1]
                flame2 = func(flame1)
                flame_deriv = 2*(flame2-answers[j][m])
                flame_deriv *= d_func(flame1)
                flame_force+=flame_deriv
            deriv_flame_list.append(flame_force)

        return deriv_flame_list
            
    # the next function updates the weight and bias
    # the update will be proportional to their derivatives on the system
    # the function will loop until either the stop point or the lowest error have been reached
    # the variable explanation below and its set values are assuming an untreated function
    # if will probably have those varies across answ
    # the stop point is defaulted as 0.001, but it is changeable
    # the function also a variable called learning rate for how much proportional updating to do on the setting values
    # the optimal learning rate is currently 0.00001, below that the convergence is too slow, and above, it doesn't converge
    
    def train(self,stop_point=0.001,learning_rate=0.00001):
    
        # treatments_v2 will select which type of function and its derivative to apply for the predicted value
               
        treatments_v2 = {0: [NN_equation_type.untreated,NN_equation_type.d_untreated],
                      1: [NN_equation_type.sigmoid,NN_equation_type.d_sigmoid],
                      2: [NN_equation_type.ReLU,NN_equation_type.d_ReLU]}
            
        treat_func = treatments_v2[self.treat_type][0]
        d_treat_func = treatments_v2[self.treat_type][1] 
            
        variable_list = self.variable_list
        answer_list = self.answer_list
        loop_count = 0
               
        while True:
            setting_list = self.settings_list
            
            # error_danger1 checks the previous error
            
            error_danger1 = self.avg_error_func()
            
            w_der_list = NN_equation_type.univ_sqrd_diff_w_der(variable_list, answer_list, setting_list, treat_func, d_treat_func)
            b_der_list = NN_equation_type.univ_sqrd_diff_b_der(variable_list, answer_list, setting_list, treat_func, d_treat_func)
               
            for m in range(len(w_der_list)):
                for k in range(len(w_der_list[0])):
                    self.settings_list[m][k] -= learning_rate*w_der_list[m][k]
               
            for m in range(len(b_der_list)):
                self.settings_list[m][-1] -= learning_rate*b_der_list[m]
            
            # error_danger2 checks the new error
            
            error_danger2 = self.avg_error_func()
                
            if error_danger2[0]<stop_point:
                print("Stopping the process...")
                print("You reached a very low standard deviation: {}".format(math.sqrt(error_danger2[0])))
                print("Your final settings: {}".format(self.settings_list))
                break
                
            # it prints the current error when it reaches 10000 loops
            
            if loop_count%10000==0:
                print("Current standard deviation: {}".format(math.sqrt(error_danger2[0])))
            
            if 0.9999*error_danger1[0]<error_danger2[0]:
                print("Current final standard deviation: {}".format(math.sqrt(error_danger2[0])))
                print("No further training recommended\nHere's your final settings: {}".format(self.settings_list))
                break
                
            loop_count+=1
               
