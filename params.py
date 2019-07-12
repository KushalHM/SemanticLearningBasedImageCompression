from pprint import pprint

class HyperParams() :
    def __init__(self, verbose):
        # Hard params and magic numbers
        self.sparse      = True
        self.model_path  = 'models/model'
        self.n_labels    = 257
        self.top_k       = 5  
        self.stddev      = 0.2
        self.fine_tuning = False 
        self.image_h     = 224
        self.image_w     = 224
        self.image_c     = 3 
        self.filter_h    = 3
        self.filter_w    = 3

        if verbose:
            pprint(self.__dict__)
        
class TrainingParams():
    def __init__(self, verbose):
        self.model_path         = './models/'
        self.num_epochs         = 200
        self.learning_rate      = 0.002
        self.weight_decay_rate  = 0.0005
        self.momentum           = 0.9
        self.batch_size         = 32
        self.max_iters          = 200000
        self.test_every_iter    = 200
        self.data_train_path    = './data/train.pickle'
        self.data_test_path     = './data/test.pickle'
        self.images             = './data/images'
        self.resume_training    = True
        self.on_resume_fix_lr   = False
        self.change_lr_env      = False
        self.optimizer          = 'Adam' # 'Adam', 'Rmsprop', 'Ftlr'

        if verbose:
            pprint(self.__dict__)

class CNNParams():
    def __init__(self, verbose):
        self.pool_window   = [1, 2, 2, 1]
        self.pool_stride   = [1, 2, 2, 1]
        self.last_features = 1024
        self.conv_filters  = [2048]
        self.depth_filters = [32]
        self.layer_shapes  = self.get_layer_shapes()

        if verbose:
            pprint(self.__dict__)

    def get_layer_shapes(self):
        shapes = {}
        hyper = HyperParams(verbose=False)
        l = self.last_features
        f = self.conv_filters
        d = self.depth_filters[-1]

        shapes['convFeatures/W'] = (hyper.filter_h, hyper.filter_w, f[0], d)
        shapes['convFeatures/b'] = (d,)
        shapes['depth/W']   = (hyper.filter_h, hyper.filter_w, d,d)
        shapes['depth/b']   = (l, )
        shapes['convReplacement/W']   = (hyper.filter_h, hyper.filter_w, l, l)
        shapes['convReplacement/b']   = (l,)
        shapes['GAP/W']     = (l, hyper.n_labels)
        return shapes


