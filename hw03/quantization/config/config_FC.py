class FC_config():
    def __init__(self,
                 input_size=784,
                 hidden_size=[512,128,64],#1024,
                 #num_hidden_layers=,
                 num_classes=10,
                 hidden_states=True,
                 ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.hidden_states = hidden_states
        #self.ID = args.ID