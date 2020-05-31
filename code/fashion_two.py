''' author: samtenka
    change: 2020-05-29
    create: 2019-05-29
    descrp: instantiate abstract class `Landscape` for tiny toy Fashion model
'''


from utils import device, prod, secs_endured, megs_alloced, CC, pre, reseed

from landscape import PointedLandscape, FixedInitsLandscape

import tqdm
import numpy as np
import torch
from torch import conv2d, matmul, tanh
from torch.nn.functional import log_softmax, nll_loss 
from torchvision import datasets, transforms
import torchvision

pre(str(torchvision.__version__) in ('0.2.1', '0.3.0'), 
    'torchvision version should be 0.2.1 or 0.3.0'
)

#=============================================================================#
#           0. Fashion as a DATASET                                           #
#=============================================================================#

class Fashion(PointedLandscape):
    '''
        Load specified class_nms of Fashion, e.g. just 0s and 1s for binary
        classification subtask.  Implements PointedLandscape's `sample_data`
        but not its `update_weight`, `get_weight`, `set_weight`,
        `get_loss_stalk`, or `nabla`.
    '''

    CLASS_NMS = (
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    )

    def __init__(self, class_nms=CLASS_NMS): 
        '''
            Load Fashion images and labels, downloading as needed.  Only keep the
            classes named.
        '''

        #---------------------------------------------------------------------#
        #           0.0. load all of Fashion                                  #
        #---------------------------------------------------------------------#

        train_set, test_set = (
            datasets.FashionMNIST(
                '../data',
                train=train_flag,
                download=True,
                transform=transforms.ToTensor()
            )
            for train_flag in (True, False) 
        )

        if str(torchvision.__version__)=='0.3.0':
            pre(train_set.classes == test_set.classes,
                'test and train class index-name correspondences should agree!'
            )
        self.imgs = np.concatenate(
            [train_set.data , test_set.data]
            if str(torchvision.__version__) == '0.3.0' else
            [train_set.train_data , test_set.test_data]
        , axis=0) / 255.0
        self.lbls = np.concatenate(
            [train_set.targets, test_set.targets]
            if str(torchvision.__version__) == '0.3.0' else
            [train_set.train_labels, test_set.test_labels]
        , axis=0)

        #---------------------------------------------------------------------#
        #           0.1. filter for requested classes                         #
        #---------------------------------------------------------------------#

        class_nm_from_lbl = (
            train_set.classes
            if str(torchvision.__version__)=='0.3.0' else
            Fashion.CLASS_NMS
        )

        idxs_to_keep = np.array([
            i for i, lbl in enumerate(self.lbls)
            if class_nm_from_lbl[lbl] in class_nms
        ]).astype(np.int32) 

        unflipped = self.imgs[idxs_to_keep]
        flipped = self.imgs[idxs_to_keep, :, ::-1] # reverse order of columns
        intensity = np.linalg.norm(unflipped, axis=(1,2))
        asymmetry = np.linalg.norm(flipped-unflipped, axis=(1,2)) #/ intensity
        intensity /= np.amax(intensity)
        symmetry = 1.0 - asymmetry / np.amax(asymmetry)
        intensity -= np.mean(intensity)
        symmetry -= np.mean(symmetry)
        print(intensity)
        print(symmetry)

        self.imgs = torch.Tensor(
            [[ii,ss] for ii, ss in zip(intensity, symmetry)]
        ).view(-1, 2)
        self.lbls = torch.Tensor([
            class_nms.index(class_nm_from_lbl[l]) 
            for l in self.lbls[idxs_to_keep]
        ]).view(-1).long()

        #---------------------------------------------------------------------#
        #           0.2. record index bounds and range                        #
        #---------------------------------------------------------------------#

        self.nb_classes = len(class_nms)
        self.nb_datapts = len(idxs_to_keep)
        self.idxs = np.arange(self.nb_datapts)
        print(CC + '@Y {} @D classes; @M {} @D samples'.format(
            self.nb_classes, self.nb_datapts
        ))

    def sample_data(self, N, seed):
        '''
            Sample data i.i.d. from Fashion's 60000-mass distribution.  So
            repeats will occur with positive probability.  Samples idxs,
            so lightweight.
        '''
        reseed(seed)
        return np.random.choice(self.idxs, N, replace=True)



#=============================================================================#
#           1. CONNECT LANDSCAPE INTERFACE TO PYTORCH                         #
#=============================================================================#

class FashionAbstractArchitecture(Fashion, FixedInitsLandscape):
    ''' 
        Partial implementation of Landscape class for neural networks that
        classify Fashion.  The method `logits_and_labels` remains to be
        implemented by descendants of this class.  The attribute
        `subweight_shapes` remains to be defined in a descendant's constructor; 
        immediately after this definition, `prepare_architecture` should be
        called.
    '''
    def __init__(self, class_nms=Fashion.CLASS_NMS,
                 weight_scale=1.0, init_scale=1.0):
        '''
            Load Fashion data.
        '''
        super().__init__(class_nms)
        self.weight_scale = weight_scale

    #-------------------------------------------------------------------------#
    #               1.0. weight slicing, scaling, getting, and setting        #
    #-------------------------------------------------------------------------#

    def prepare_architecture(self): 
        '''
            Partition weight vector into weight layers and memoize for each
            layer the proper Xavier scaling.
        '''
        self.subweight_offsets = [
            sum(prod(shape) for shape in self.subweight_shapes[:depth])
            for depth in range(len(self.subweight_shapes)+1) 
        ]
        self.subweight_scales = [
            self.weight_scale * (2.0 / (shape[0] + prod(shape[1:])))**0.5
            for shape in self.subweight_shapes
        ]

    def subweight(self, depth):
        '''
            Select (sliced and scaled) weight layer from total weight vector.
            Here, `depth` is an index that specifies a layer.
        '''
        return (
            self.subweight_scales[depth] * 
            self.weight[
                self.subweight_offsets[depth]:
                self.subweight_offsets[depth+1]
            ]
        ).view(self.subweight_shapes[depth])

    def sample_weight(self, seed):
        '''
            Sample an (unscaled, unsliced) total weight vector from a standard
            normal, for use as an initialization before descent.  The Xavier
            initialization we implement in the architecture (so changing
            Xavier would change the metric) instead of in this sampling
            method.
        '''
        reseed(seed)
        return 0.0 * np.random.randn(self.subweight_offsets[-1])

    def get_weight(self):
        '''
            Return a numpy value of the total weight vector. 
        '''
        return self.weight.detach().numpy()

    def set_weight(self, weights):
        '''
            Set the total weight vector from the given numpy value.
        '''
        self.weight = torch.autograd.Variable(
            torch.Tensor(weights),
            requires_grad=True
        )

    def update_weight(self, displacement):
        '''
            Add the given numpy displacement to the current weight.
        '''
        self.weight.data += displacement.detach().data

    #-------------------------------------------------------------------------#
    #               1.1. the subroutines and diagnostics of descent           #
    #-------------------------------------------------------------------------#

    def nabla(self, scalar_stalk, create_graph=True):
        '''
            Differentiate a stalk, assumed to be at the current weight, with
            respect to this weight.
        '''
        return torch.autograd.grad(
            scalar_stalk,
            self.weight,
            create_graph=create_graph,
        )[0] 

    def get_loss_stalk(self, data_idxs):
        '''
            Compute cross entropy loss on provided data indices; return as a
            pytorch tensor.  Calls the method `logits_and_labels` to be
            implemented.
        '''
        logits, labels = self.logits_and_labels(data_idxs)
        return nll_loss(logits, labels)

    def get_metrics(self, data_idxs):
        '''
            Compute cross entropy loss and classification accuracy on provided
            data indices; return as a dictionary of floats.  Calls the method
            `logits_and_labels` to be implemented.
        '''
        logits, labels = self.logits_and_labels(data_idxs)
        _, argmax = logits.max(1) 
        return {
            'loss': nll_loss(logits, labels).detach().numpy(),
            'acc':  float(argmax.eq(labels).sum()) / labels.shape[0]
        }

#=============================================================================#
#           2. DEFINE ARCHITECTURES FOR Fashion CLASSIFICATION                #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                   2.0. shallow or                                           #
#-----------------------------------------------------------------------------#

class FashionShallowOr(FashionAbstractArchitecture):
    ''' 
    '''
    def __init__(self, weight_scale=1.0, verbose=False, seed=0):
        '''
            Define tensor shape of network and initialize weight vector.
        '''
        class_nms=('Pullover', 'Bag')
        super().__init__(class_nms, weight_scale)
        self.subweight_shapes = [
            (1,),
            (1,),
        ]
        self.prepare_architecture()

        self.set_weight(self.sample_weight(seed))

        if verbose:
            print('Shallow Or has {} parameters'.format(
                sum(prod(w) for w in self.subweight_shapes)
            ))

    def logits_and_labels(self, data_idxs):
        '''
            Implement a linear model.
        '''
        x, y = self.imgs[data_idxs], self.lbls[data_idxs]

        x = x.view(-1, 2)
        x = torch.log((torch.exp(self.subweight(0) * x[:, 0])+  
                       torch.exp(self.subweight(1) * x[:, 1]) )/2)
        x = x.view(-1, 1)
        x = torch.cat((x, torch.zeros(x.shape)), 1)

        logits = log_softmax(x, dim=1)
        return logits, y

#=============================================================================#
#           3. DEMONSTRATE INTERFACE by REPORTING GRAD STATS during DESCENT   #
#=============================================================================#

if __name__=='__main__':

    #-------------------------------------------------------------------------#
    #               3.0. descent hyperparameters                              #
    #-------------------------------------------------------------------------#

    N = 8192#256 
    BATCH = 256#64
    TIME = 5000
    STEP =  200 
    LRATE = 0.00#1.0
    pre(N%BATCH==0,
        'batch size must divide train size!'
    )

    #-------------------------------------------------------------------------#
    #               3.1 specify and load model                                #
    #-------------------------------------------------------------------------#

    model_nm = 'SHALLOWOR'
    nb_inits = 1
    model_idx = 0 

    file_nm = 'saved-weights/fashion-{}.npy'.format(model_nm.lower())
    model = {'SHALLOWOR':FashionShallowOr}[model_nm]
    ML = model(verbose=True, seed=0)
    ML.load_from(file_nm, nb_inits=nb_inits, seed=0)
    ML.switch_to(model_idx)
    #ML.set_weight(np.load('saved-weights/fashion-{}-trained.npy'.format(model_nm.lower()))[0])

    D = ML.sample_data(N=N, seed=0) 
    for t in range(TIME):
        #---------------------------------------------------------------------#
        #           3.2 perform one descent step                              #
        #---------------------------------------------------------------------#

        L = ML.get_loss_stalk(D[(BATCH*t)%N:(BATCH*(t+1)-1)%N+1])
        G = ML.nabla(L)
        ML.update_weight(-LRATE * G)

        #---------------------------------------------------------------------#
        #           3.3 compute and display gradient statistics               #
        #---------------------------------------------------------------------#

        if (t+1)%STEP: continue

        L_train= ML.get_loss_stalk(D)
        acc_train = ML.get_metrics(D)['acc']

        data = ML.sample_data(N=3000, seed=1)
        L_test = ML.get_loss_stalk(data[:1500])
        L_test_= ML.get_loss_stalk(data[1500:])
        acc = ML.get_metrics(data)['acc']

        print(CC+' @D \t'.join([
            'after @M {:4d} @D steps'.format(t+1),
            'grad2 @G {:.2e}'.format(
                ML.nabla(L_test).dot(ML.nabla(L_test_)).detach().numpy()
            ),
            'train acc @O {:.2f}'.format(acc_train),
            'train loss @Y {:.2f}'.format(L_train.detach().numpy()),
            'test loss @L {:.2f}'.format(
                (L_test + L_test_).detach().numpy()/2.0
            ),
            'test acc @O {:.2f}'.format(acc),
            'weight<intensity> @M {:.3f}'.format(ML.get_weight()[0]),
            'weight<symmetry> @M {:.3f}'.format(ML.get_weight()[1]),
        '']))

    #np.save('saved-weights/fashion-{}-trained.npy'.format(
    #    model_nm.lower()
    #), [ML.get_weight()])
