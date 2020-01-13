''' author: samtenka
    change: 2020-01-12
    create: 2019-06-10
    descrp: define loss landscape type 
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np
import os.path

from utils import CC, pre, reseed
import random



#=============================================================================#
#           0. DECLARE LANDSCAPE INTERFACE                                    #
#=============================================================================#

class PointedLandscape(ABC):
    '''
        Interface for a stochastic loss landscape.  In particular, presents a
        distribution over smooth loss functions on weight space by providing: 

            get_loss_stalk : [Datapoint] --> Weight* --> Real

            sample_data : Nat --> Seed --> [Datapoint]

            nabla : ( Weight* --> Real ) --> ( Weight* --> Real )

        This landscape class is *pointed*, meaning that it maintains a weight
        via the lens

            get_weight: Landscape --> Weight

            set_weight: Landscape --> Weight --> Landscape

        --- a weight sampled according to a distribution 

            sample_weight: Seed --> Weight 
        
        It is at this weight that `get_loss_stalk` and `nabla` act.  Likewise,
        via
            
            update_weight: Landscape --> TangentSpace(Weight) --> Landscape

        gradient descent is cleanly implementable.  Above, the type `Weight*`
        actually indicates the weights in an infinitesimal neighborhood around
        the weight maintained by the Landscape at the time of execution of that
        function; we read the type `Weight* --> Real` as the type of (Taylor)
        germs of smooth functions at the landscape's weight.
    '''

    #-------------------------------------------------------------------------#
    #               0.0 declare samplers for data and weights                 #
    #-------------------------------------------------------------------------#

    @abstractmethod
    def sample_data(self, N, seed): 
        '''
            Sample N datapoints (i.e. memory-light objects indexing
            deterministic loss landscapes) independently and identically
            distributed from the population.
        '''
        pass

    @abstractmethod
    def sample_weight(self, seed):
        '''
            Sample from a (potentially dirac) distribution that is a property
            of this Pointed Landscape class (but not conceptually a property of
            the mathematical object).  This method should be automatically
            called at initialization.
        '''
        pass

    #-------------------------------------------------------------------------#
    #               0.1 declare lens for current weights                      #
    #-------------------------------------------------------------------------#

    @abstractmethod
    def get_weight(self):
        '''
            Return the numpy value of current weight.
        '''
        pass

    @abstractmethod
    def set_weight(self, weight):
        '''
            Overwrite the current weight with the given numpy value.
        '''
        pass

    @abstractmethod
    def update_weight(self, displacement):
        '''
            Add displacement to weight (on a curved space, the displacement is
            actually a vector, and instead of addition one uses the Riemannian
            exponential).
        '''
        pass

    #-------------------------------------------------------------------------#
    #               0.2 declare loss stalk and its derivative operator        #
    #-------------------------------------------------------------------------#

    @abstractmethod
    def get_loss_stalk(self, data_idxs):
        '''
            Present loss, averaged over given data, as a deterministic scalar
            stalk.
        '''
        pass

    @abstractmethod
    def nabla(self, scalar_stalk):
        '''
            Differentiate the given deterministic scalar stalk (assumed to be
            on the current weight), returning a deterministic covector stalk
            (in implementation, with the same shape as weights have).
        '''
        pass

    @abstractmethod
    def get_metrics(self, data_idxs):
        '''
            Compute metrics (perhaps loss and accuracy) on provided data
            indices.  Return as a dictionary of numeric values by names.
        '''
        pass



#=============================================================================#
#           1. IMPLEMENT READING/WRITING OF WEIGHT INITIALIZATIONS            #
#=============================================================================#

class FixedInitsLandscape(PointedLandscape):
    '''
        For reduced estimation-error, we may choose to initialize only at a few
        points.  This wrapper on the PointedLandscape class implements some
        handy methods for this purpose.  
    '''

    def sample_to(self, file_nm, nb_inits, seed=0):
        '''
            Save a random list of weight initializations to the file named.
        '''
        pre(file_nm.endswith('.npy'),
            'file name must end with .npy'
        )
        pre(not os.path.isfile(file_nm),
            'avoided overwriting {}'.format(file_nm)
        )
        reseed(seed)
        self.inits = [random.randint(0, 2**32) for _ in range(nb_inits)] 
        np.save(file_nm, self.inits)
        print(CC + 'saved @R {} @D initial weights to @M {} @D '.format(
            len(self.inits), file_nm
        ))
        self.switch_to(0)

    def load_from(self, file_nm, nb_inits=None, seed=0):
        '''
            Load a set of weight initializations from the file named.
        '''
        pre(file_nm.endswith('.npy'),
            'file name must end with .npy'
        )
        if not os.path.isfile(file_nm):
            pre(nb_inits is not None,
                'attempted resample before load: nb_inits is unspecified!'
            )
            self.sample_to(file_nm, nb_inits, seed)
        else:
            self.inits = np.load(file_nm)
        print(CC + 'loaded @R {} @D initial weights from @M {} @D '.format(
            len(self.inits), file_nm
        ))
        self.switch_to(0)

    def switch_to(self, init_idx):  
        '''
            Switch the current weight to that of the given index.
        '''
        self.set_weight(self.sample_weight(self.inits[init_idx]))
