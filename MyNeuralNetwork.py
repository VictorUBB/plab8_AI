import numpy as np
class MyNeuralNetwork:
    def __init__(self,hiddenLayers,maxIter=1000,learningRate=.001):
        self.hiddenLayer=hiddenLayers
        self.maxIter=maxIter
        self.learningRate=learningRate
        self.weights=[]

    def softmax(self, x):
        exp_vector = np.exp(x)
        return exp_vector / exp_vector.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fit(self,x,y):
        nr_features=len(x[0])
        nr_out=len(set(y))
        new_y=np.zeros((len(y),nr_out))
        for i in range(len(y)):
            new_y[i,y[i]]=1
        y= new_y
        #se initilizeaza ponderile
        w_in_hidden=np.random.rand(nr_features,self.hiddenLayer)
        w_hidden_out=np.random.rand(self.hiddenLayer,nr_out)
        coef_in_hidden=np.random.randn(self.hiddenLayer)
        coef_hidden_out=np.random.randn(nr_out)
        for epoca in range(self.maxIter):

            #propagare informatii
            y_in_hidden=np.dot(x,w_in_hidden)+coef_in_hidden

            #activare
            o_in_hidden=self.sigmoid(y_in_hidden)

            #outputurile
            out = np.dot(o_in_hidden,w_hidden_out)+coef_hidden_out

            #se transforma outpututile prin softmax->0-1
            out_softmax=self.softmax(out)

            #propagare erori
            err=out_softmax-y
            err_w_hidden_out=np.dot(o_in_hidden.T,err)
            err_coef=err
            error_dah=np.dot(err,w_hidden_out.T)
            sig_der=self.sigmoid_derivative(y_in_hidden)
            cp_x=x
            w_err_in_hidden=np.dot(cp_x.T,sig_der*error_dah)
            err_corf_in_hid=error_dah*sig_der
            w_in_hidden-=self.learningRate*w_err_in_hidden
            coef_in_hidden -=self.learningRate*err_corf_in_hid.sum(axis=0)
            w_hidden_out-=self.learningRate*err_w_hidden_out
            coef_hidden_out-=self.learningRate*err_coef.sum(axis=0)
        self.weights=[w_in_hidden,coef_in_hidden,w_hidden_out,coef_hidden_out]

    def predict(self,x):
        w_in_hidden,coef_in_hidden,w_hidden_out,coef_hidden_out=self.weights
        y_in_hidden=np.dot(x,w_in_hidden)+coef_in_hidden
        y_in_hid_sig=self.sigmoid(y_in_hidden)
        y_out=np.dot(y_in_hid_sig,w_hidden_out)+coef_hidden_out
        y_out_softmax=self.softmax(y_out)
        comp_out=[list(out).index(max(out)) for out in y_out_softmax]
        return comp_out