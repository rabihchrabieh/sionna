import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

#tf.config.run_functions_eagerly(True)

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.fec.linear import OSDecoder
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.channel import AWGN

# %%
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements

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
            
        encoder: Sionna Block
            A Sionna Block that encodes information bit tensors.
            
        decoder: Sionna Block
            A Sionna Block layer that decodes llr tensors.
            
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
            
        sim_esno: bool  
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

         cw_estiamtes: bool  
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
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
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False,
                 cw_estimates=False):

        super().__init__()
        
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
        
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

        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder
        
        z = self.encoder.z
        self.encoder.set_circ_buff_start(2*z)
        self.decoder.set_circ_buff_start(2*z)
        
        
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

        # Reset to RV0 (from a previous loop)
        self.encoder.set_rv('rv0')
        self.decoder.set_rv('rv0')
        
        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode
        
        x = self.mapper(c) # map c to symbols x

        y = self.channel(x, no) # transmit over AWGN channel

        llr_ch = self.demapper(y, no) # demap y to LLRs

        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)

        if self.decoder._harq_mode:
            if True:
                # Second HARQ round
                circ_buff = u_hat[1]
                self.encoder.set_rv('rv2')
                self.decoder.set_rv('rv2')
                c2 = self.encoder(u) # explicitly encode
                x2 = self.mapper(c2) # map c to symbols x
                y2 = self.channel(x2, no) # transmit over AWGN channel
                llr_ch2 = self.demapper(y2, no) # demap y to LLRs
                u_hat = self.decoder(llr_ch2, circ_buff=circ_buff) # run FEC decoder
            u_hat = u_hat[0]
            
        if self.cw_estimates:
            return c, u_hat
            
        return u, u_hat

# %%
# code parameters
k = 64 # number of information bits per codeword
n = 128 # desired codeword length

# Create list of encoder/decoder pairs to be analyzed.
# This allows automated evaluation of the whole list later.
codes_under_test = []

# 5G LDPC codes with 20 BP iterations
enc = LDPC5GEncoder(k=k, n=n)
dec = LDPC5GDecoder(enc, num_iter=20, prune_pcm=True, harq_mode=True)
name = "5G LDPC BP-20"

ber_plot128 = PlotBER(f"Performance of Short Length Codes (k={k}, n={n})")

# %% [markdown]
# And run the BER simulation for each code.

# %%
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 5, 0.5) # sim SNR range 

# run ber simulations for each code we have added to the list
# generate a new model with the given encoder/decoder
model = System_Model(k=k,
                        n=n,
                        num_bits_per_symbol=num_bits_per_symbol,
                        encoder=enc,
                        decoder=dec)

# the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
ber_plot128.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db, # SNR to simulate
                        legend=name, # legend string for plotting
                        max_mc_iter=100, # run 100 Monte Carlo runs per SNR point
                        num_target_block_errors=100, # continue with next SNR point after 1000 bit errors
                        batch_size=10000, # batch-size per Monte Carlo run
                        soft_estimates=False, # the model returns hard-estimates
                        early_stop=True, # stop simulation if no error has been detected at current SNR point
                        show_fig=False, # we show the figure after all results are simulated
                        add_bler=True, # in case BLER is also interesting
                        forward_keyboard_interrupt=True); # should be True in a loop

# and show the figure
ber_plot128(ylim=(1e-5, 1), show_bler=False) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.
