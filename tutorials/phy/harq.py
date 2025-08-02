# %% [markdown]
# # 5G LDPC HARQ Performance in AWGN
# 
# In this notebook, you will learn how to implement 5G LDPC HARQ retransmissions with different redundancy versions using Sionna.
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [BLER Performance of 5G LDPC with HARQ](#BLER-Performance-of-5G-LDPC-HARQ)

# %% [markdown]
# ## GPU Configuration and Imports

# %%
import os
import tensorflow as tf

# Configure GPU/CPU automatically
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA GPU detected - using GPU for computation")
    # Use first GPU and allow memory growth
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("No CUDA GPU detected - using CPU for computation")
    tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e 

# For testing in eager mode
tf.config.run_functions_eagerly(True)

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.channel import AWGN

# %%
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements

# %% [markdown]
# ## BLER Performance of 5G LDPC HARQ
# 
# For k = 64 and n = 150, we simulate 4 scenarios with increasing numbers of retransmissions:
# - RV0 only (no retransmissions)
# - RV0 + RV2
# - RV0 + RV2 + RV3
# - RV0 + RV2 + RV3 + RV1

# %% [markdown]
# Let us define the system model first. We use encoder and decoder as input parameter such that the model remains flexible w.r.t. the coding scheme.

# %%
class System_Model(Block):
    """System model for channel coding BER simulations.
    
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to 
    initialize the model.
    
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        
        n: int 
            codeword length.
        
        num_bits_per_symbol: int
            number of bits per QAM symbol.
            
        rv_set: list of str
            A list of redundancy versions (RV) to be used for the simulation.
            The RVs are used to set the FEC encoder and decoder to the correct
            redundancy version. Defaults to ['rv0'].
            
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
            
        sim_esno: bool  
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        
        ebno_db: float or tf.float
            A float defining the simulation SNR.
            
    Output
    ------
        (u, u_hat):
            Tuple:
        
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 rv_set=['rv0'],
                 demapping_method="app",
                 sim_esno=False):

        super().__init__()
        
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        
        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # init components
        self.source = BinarySource()
       
        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        
        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()

        # 5G LDPC codes with 20 BP iterations (prune_pcm not yet supported
        # in HARQ mode)
        self.encoder = LDPC5GEncoder(k=k, n=n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20,
                                     prune_pcm=False, harq_mode=True)

        # redundancy version set
        self.rv_set = rv_set

    @tf.function() # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):

        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else: 
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)            

        u = self.source([batch_size, self.k]) # generate random data
        
        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u, rv=self.rv_set) # explicitly encode
        x = self.mapper(c) # map c to symbols x
        y = self.channel(x, no) # transmit over AWGN channel
        llr_ch = self.demapper(y, no) # demap y to LLRs
        u_hat = self.decoder(llr_ch, rv=self.rv_set)  # decoder

        return u, u_hat

# %% [markdown]
# Run the BLER simulation for each scenario, with progressively more retransmissions. And plot the results.

# %%
k = 64 # number of information bits per codeword
n = 150 # desired codeword length
rv_set = ['rv0', 'rv2', 'rv3', 'rv1']  # redundancy version set for HARQ
rv_set_upper = [rv.upper() for rv in rv_set]  # upper-case for plotting

num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 4, 0.5) # sim SNR range 
ebno_offset = np.array([0.0, -4.0, -6.0, -7.0]) # offset for each number of retries

bler_plot128 = PlotBER(f"5G LDPC BP-20 - HARQ Performance (k={k}, n={n})")

for i in range(len(rv_set)):
    # run BLER simulations for each number of retransmission attempts:
    # rv0; rv0+rv2; rv0+rv2+rv3; rv0+rv2+rv3+rv1.
    # In each scenario, the first i+1 redundancy versions are used.
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         rv_set=rv_set[0:i+1])  # use the first i+1 redundancy versions

    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    bler_plot128.simulate(model, # the function have defined previously
                          ebno_dbs=ebno_db + ebno_offset[i], # SNR to simulate
                          legend=" + ".join(rv_set_upper[0:i+1]), # legend string for plotting
                          max_mc_iter=100, # run 100 Monte Carlo runs per SNR point
                          num_target_block_errors=100, # continue with next SNR point after 1000 bit errors
                          batch_size=10000, # batch-size per Monte Carlo run
                          soft_estimates=False, # the model returns hard-estimates
                          early_stop=True, # stop simulation if no error has been detected at current SNR point
                          show_fig=False, # we show the figure after all results are simulated
                          add_bler=True, # in case BLER is also interesting
                          forward_keyboard_interrupt=True); # should be True in a loop

# and show the figure
bler_plot128(ylim=(1e-3, 1), show_ber=False, show_bler=True)


