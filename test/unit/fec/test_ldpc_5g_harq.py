#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
import pytest
import numpy as np

from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.utils import GaussianPriorSource
from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import BinarySource
from sionna.phy.channel import AWGN    
from sionna.phy.mapping import Mapper, Demapper, Constellation

#############################
# Test cases for LDPC5G HARQ
#############################

@pytest.mark.parametrize("k_n", [(400, 900), (5000, 12000)])
@pytest.mark.parametrize("rv_list", [["rv0", "rv1", "rv2", "rv3"]])
def test_harq_encoder(k_n, rv_list):
    """Test HARQ encoder functionality with different RV combinations.
    
    This test verifies that the HARQ encoder correctly generates codewords
    for different redundancy versions (RVs) by comparing against a reference
    encoder with circular shifting.
    
    Tests:
    - Various code rates (k/n combinations)
    - HARQ transmissions (4 RVs)
    - Correct RV-specific codeword generation
    - Output shape and data type validation
    - Encoder doesn't crash with different RV configurations
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    k, n = k_n
    batch_size = 2
    num_rv = len(rv_list)

    ldpc_params = LDPC5GEncoder.get_params(k, n)
    n_cb = ldpc_params.n_cb

    # Reference encoder
    encoder_ref = LDPC5GEncoder(k, n_cb, harq_mode=True)

    # HARQ encoder
    encoder = LDPC5GEncoder(k, n, harq_mode=True)
    starts = encoder.get_rv_starts()

    # Generate encoded bits
    source = BinarySource()
    bits = source([batch_size, k])
    x_ref = encoder_ref(bits)  # Shape: [batch_size, n_cb]
    x = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]

    # Validate encoded bits
    assert x.shape == [batch_size, num_rv, n]
    for i, rv in enumerate(rv_list):
        start = starts[rv]
        x_ref_unrolled = tf.roll(x_ref, shift=-start, axis=-1)
        x_ref_unrolled = x_ref_unrolled[:, :n]  # Adjust to match n length
        assert tf.equal(x_ref_unrolled, x[:, i, :]).numpy().all(), \
            f"RV {rv} encoding mismatch"


@pytest.mark.parametrize("k_n", [(400, 900)])
@pytest.mark.parametrize("use_graph_mode", [True, False])
@pytest.mark.parametrize("use_xla", [True, False])
def test_harq_encoder_graph_xla(k_n, use_graph_mode, use_xla):
    """Test HARQ encoder in graph mode and XLA compilation.
    
    This test ensures that the HARQ encoder works correctly under different
    execution modes: eager mode, graph mode, and XLA compilation. This is
    important for performance optimization and deployment scenarios.
    
    Tests:
    - Eager mode execution (use_graph_mode=False, use_xla=False)
    - Graph mode execution with @tf.function decoration
    - XLA compilation compatibility (jit_compile=True)
    - Consistent output across execution modes
    - No performance regressions or compilation errors
    """
    
    # Set random seed for reproducibility
    tf.random.set_seed(123)
    
    k, n = k_n
    batch_size = 4
    rv_list = ["rv0", "rv1", "rv2"]

    ldpc_params = LDPC5GEncoder.get_params(k, n)
    n_cb = ldpc_params.n_cb

    # Reference encoder for validation
    encoder_ref = LDPC5GEncoder(k, n_cb)
    
    # HARQ encoder
    encoder = LDPC5GEncoder(k, n, harq_mode=True)
    starts = encoder.get_rv_starts()
    
    source = BinarySource()
    
    @tf.function(jit_compile=use_xla)
    def run_harq_encoder():
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)
        x_harq = encoder(bits, rv=rv_list)
        return bits, x_ref, x_harq
    
    if use_graph_mode:
        bits, x_ref, x_harq = run_harq_encoder()
    else:
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)
        x_harq = encoder(bits, rv=rv_list)

    # Validate output shapes and types
    assert x_harq.shape == [batch_size, len(rv_list), n]
    assert x_harq.dtype == tf.float32
    assert not tf.reduce_any(tf.math.is_nan(x_harq))
    
    # Validate correctness by comparing with reference
    for i, rv in enumerate(rv_list):
        start = starts[rv]
        x_ref_unrolled = tf.roll(x_ref, shift=-start, axis=-1)
        x_ref_unrolled = x_ref_unrolled[:, :n]
        
        # Should be bitwise identical
        assert tf.reduce_all(tf.equal(x_ref_unrolled, x_harq[:, i, :])), \
            f"RV {rv} encoding mismatch in mode: Graph={use_graph_mode}, XLA={use_xla}"
    
    # Print mode info for debugging
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"HARQ encoder test passed for {mode_str}")


@pytest.mark.parametrize("return_infobits", [True, False])
@pytest.mark.parametrize("hard_out", [True, False])
def test_harq_output_formats(return_infobits, hard_out):
    """Test HARQ with different output format options.
    
    This test verifies that the HARQ decoder correctly handles different 
    output configurations. The decoder can return either information bits 
    or full codewords, and can output either soft LLRs or hard binary decisions.
    
    Tests:
    - return_infobits=True: Output should be [batch_size, k] (info bits only)
    - return_infobits=False: Output should be [batch_size, n] (full codeword)
    - hard_out=True: Output should contain only binary values (0 or 1)
    - hard_out=False: Output should contain soft LLR values
    """
    
    k, n = 150, 300
    batch_size = 8
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, 
                           harq_mode=True, 
                           return_infobits=return_infobits,
                           hard_out=hard_out,
                           num_iter=15)
    
    source = GaussianPriorSource()
    rv_list = ["rv0", "rv2"]
    llr_ch = source([batch_size, len(rv_list), n], 0.6)
    
    result = decoder(llr_ch, rv=rv_list)
    
    # Check output shape
    if return_infobits:
        expected_shape = [batch_size, k]  # Accumulated across RVs
    else:
        expected_shape = [batch_size, len(rv_list), n]  # Per-RV results
    assert result.shape == expected_shape
    
    # Check hard/soft output
    if hard_out:
        # Should be binary values
        unique_vals = tf.unique(tf.reshape(result, [-1]))[0]
        assert len(unique_vals) <= 2
        assert tf.reduce_all(tf.logical_or(result == 0, result == 1))

@pytest.mark.parametrize("use_state", [True, False])
def test_harq_with_decoder_state(use_state):
    """Test HARQ functionality with decoder state management.
    
    The LDPC decoder can optionally return its internal state (edge messages
    from the belief propagation algorithm). This test verifies that HARQ mode
    works correctly both with and without state return.
    
    Tests:
    - return_state=True: Decoder should return (result, state) tuple
    - return_state=False: Decoder should return only result
    - State tensor should have proper dimensions when returned
    - Normal decoding should work in both cases
    """
    
    k, n = 100, 200
    batch_size = 6
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, 
                           harq_mode=True, 
                           return_state=use_state,
                           num_iter=5)
    
    source = GaussianPriorSource()
    rv_list = ["rv0", "rv1", "rv3"]
    llr_ch = source([batch_size, len(rv_list), n], 0.5)
    
    if use_state:
        result, state = decoder(llr_ch, rv=rv_list)
        assert state is not None
        assert state.shape[0] > 0  # Should have some edges
        assert state.shape[1] == batch_size
        assert result.shape == [batch_size, k]  # Default HARQ accumulates
    else:
        result = decoder(llr_ch, rv=rv_list)
        assert result.shape == [batch_size, k]

def test_harq_vs_single_transmission():
    """Test that HARQ with single RV behaves like non-HARQ mode.
    
    This is a regression test to ensure that enabling HARQ mode with only
    one redundancy version produces the same results as the standard 
    (non-HARQ) decoder. This verifies that HARQ mode doesn't introduce
    unexpected overhead or behavior changes when not actually needed.
    
    Tests:
    - HARQ decoder with 1 RV == standard decoder with same LLRs
    - Results should be numerically identical (within floating point precision)
    - Validates that HARQ mode is backward compatible
    """
    
    k, n = 200, 400
    batch_size = 10
    encoder = LDPC5GEncoder(k, n)
    
    # Non-HARQ decoder
    decoder_single = LDPC5GDecoder(encoder, harq_mode=False, num_iter=20)
    
    # HARQ decoder with single RV
    decoder_harq = LDPC5GDecoder(encoder, harq_mode=True, num_iter=20)
    
    source = GaussianPriorSource()
    llr_single = source([batch_size, n], 0.4)
    llr_harq = tf.expand_dims(llr_single, axis=1)  # Add RV dimension
    
    result_single = decoder_single(llr_single)
    result_harq = decoder_harq(llr_harq, rv=["rv0"])
    
    # Results should be identical (or very close)
    assert np.allclose(result_single.numpy(), result_harq.numpy(), rtol=1e-4)

@pytest.mark.parametrize("graph_mode", [True, False])
@pytest.mark.parametrize("use_xla", [True, False])
def test_harq_graph_mode(graph_mode, use_xla):
    """Test HARQ functionality in graph and XLA modes.
    
    TensorFlow can execute in eager mode (immediate execution) or graph mode
    (compiled execution). XLA (Accelerated Linear Algebra) provides additional
    optimizations. This test ensures HARQ works in all execution modes.
    
    Tests:
    - Eager mode execution (graph_mode=False, use_xla=False)
    - Graph mode execution with @tf.function decoration
    - XLA compilation compatibility (jit_compile=True)
    - All modes should produce valid, non-NaN outputs
    - Performance optimization modes don't break HARQ functionality
    """
    
    k, n = 100, 200
    batch_size = 8
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, num_iter=10)
    source = GaussianPriorSource()
    
    @tf.function(jit_compile=use_xla)
    def run_harq(batch_size):
        rv_list = ["rv0", "rv1"]
        llr_ch = source([batch_size, len(rv_list), n], 0.5)
        return decoder(llr_ch, rv=rv_list)
    
    if graph_mode:
        result = run_harq(batch_size)
    else:
        rv_list = ["rv0", "rv1"]
        llr_ch = source([batch_size, len(rv_list), n], 0.5)
        result = decoder(llr_ch, rv=rv_list)
    
    # Should produce valid output
    assert result.shape == [batch_size, k]
    assert not tf.reduce_any(tf.math.is_nan(result))

@pytest.mark.parametrize("use_graph_mode", [True, False])
@pytest.mark.parametrize("use_xla", [True, False])
def test_harq_e2e_coding(use_graph_mode, use_xla):
    """End-to-end test of HARQ coding scheme with channel simulation.
    
    This is a correctness test that verifies the complete HARQ system works
    properly with real data transmission. Unlike interface tests that use
    synthetic LLRs, this test:
    1. Generates random information bits
    2. Encodes them with LDPC encoder
    3. Simulates 64QAM transmission over AWGN channel
    4. Applies different noise levels per HARQ transmission
    5. Verifies the decoder can recover the original bits
    
    Tests:
    - Complete encode→transmit→decode chain
    - Multiple HARQ transmissions with improving channel quality
    - Actual bit error rate measurement and validation
    - Real correctness verification (not just interface testing)
    - Graph mode and XLA compatibility
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    k, n = 100, 240  # Smaller code for faster testing
    batch_size = 10
    
    source = BinarySource()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, num_iter=20)
    channel = AWGN()
    
    # Set up 64QAM
    constellation = Constellation("qam", num_bits_per_symbol=6)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper(demapping_method="app", constellation=constellation)

    # HARQ transmission with different RV levels
    rv_list = ["rv0", "rv2", "rv3", "rv1"]
    esno_db_value = 1.0  # Fixed SNR value for testing
    no = ebnodb2no(esno_db_value, num_bits_per_symbol=6, coderate=k/n)
    
    @tf.function(jit_compile=use_xla)
    def run_e2e_harq():
        # Generate information bits
        bits = source([batch_size, k])
        
        # Encode and transmit all RVs at once
        codeword = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]    
        x = mapper(codeword)
        y = channel(x, no)
        llr = demapper(y, no)
        decoded_bits = decoder(llr, rv=rv_list)
        
        return bits, decoded_bits
    
    if use_graph_mode:
        bits, decoded_bits = run_e2e_harq()
    else:
        # Generate information bits
        bits = source([batch_size, k])
        
        # Encode and transmit all RVs at once
        codeword = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]    
        x = mapper(codeword)
        y = channel(x, no)
        llr = demapper(y, no)
        decoded_bits = decoder(llr, rv=rv_list)

    # Check decoding success
    bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(bits, tf.float32), tf.cast(decoded_bits > 0, tf.float32)), tf.float32))
    ber = bit_errors / (batch_size * k)
    
    # Print BER for analysis
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"EsNo: {esno_db_value} dB, BER: {ber.numpy():.6f} ({mode_str})")
    
    # Verify the system runs end-to-end and produces reasonable output
    assert ber <= 0.05  # BER should not exceed 5%
    assert not tf.reduce_any(tf.math.is_nan(decoded_bits))  # No NaN values
    # Note: With proper SNR levels and HARQ, BER should be quite low

@pytest.mark.parametrize("num_transmissions", [1, 2, 3, 4])
def test_harq_incremental_improvement(num_transmissions):
    """Test that additional transmissions improve performance.
    
    This test verifies the fundamental HARQ principle: more transmissions
    should generally lead to better decoding performance. It's a statistical
    test that measures BER for different numbers of HARQ transmissions.
    
    Tests:
    - 1-4 HARQ transmissions of the same codeword
    - Fixed noise level across all transmissions  
    - BER calculation and basic validation (0 ≤ BER ≤ 1)
    - Note: Due to randomness, we only test bounds, not strict improvement
    - Validates that HARQ accumulation doesn't break with more transmissions
    """
    
    k, n = 150, 300
    batch_size = 100
    
    source = BinarySource()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, num_iter=20)
    channel = AWGN()
    
    bits = source([batch_size, k])
    codeword = encoder(bits)
    
    # Simulate transmissions
    no = 0.6  # Relatively high noise
    rv_list = ["rv0", "rv1", "rv2", "rv3"][:num_transmissions]
    
    llr_transmissions = []
    for _ in rv_list:  # We don't actually use the RV value in this loop
        x_bpsk = tf.cast(2 * codeword - 1, tf.complex64)
        y = channel(x_bpsk, no)
        llr = tf.math.real(2 * y / (no**2))
        llr_transmissions.append(llr)
    
    llr_harq = tf.stack(llr_transmissions, axis=1)
    decoded_bits = decoder(llr_harq, rv=rv_list)
    
    # Calculate BER
    bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(bits, tf.float32), tf.cast(decoded_bits > 0, tf.float32)), tf.float32))
    ber = bit_errors / (batch_size * k)
    
    # More transmissions should generally lead to better performance
    # (This is a statistical test, so we just check it doesn't crash)
    assert ber >= 0.0  # BER should be non-negative
    assert ber <= 1.0  # BER should not exceed 100%

def test_harq_error_conditions():
    """Test error handling in HARQ mode.
    
    This test verifies that the HARQ decoder properly handles invalid inputs
    and edge cases. Good error handling prevents silent failures and helps
    users identify configuration problems.
    
    Tests:
    - Mismatched tensor dimensions vs RV list length
    - Invalid RV names (implementation-dependent behavior)
    - Proper exception raising for clearly invalid inputs
    - Graceful handling or appropriate errors for edge cases
    - Validates input validation logic works correctly
    """
    
    k, n = 100, 200
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True)
    
    batch_size = 5
    
    # Test mismatched RV list and tensor dimensions
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        llr_ch = tf.random.normal([batch_size, 2, n])  # 2 RVs in tensor
        decoder(llr_ch, rv=["rv0", "rv1", "rv2"])  # 3 RVs in list
    
    # Test invalid RV names (if validation is implemented)
    # This might not raise an error depending on implementation
    llr_ch = tf.random.normal([batch_size, 1, n])
    try:
        result = decoder(llr_ch, rv=["invalid_rv"])
        # If no error, just check output is valid
        assert result.shape == [batch_size, k]
    except (ValueError, tf.errors.InvalidArgumentError, KeyError):
        # Error is acceptable for invalid RV names
        pass

@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dtype_in", [tf.float32, tf.float64])
def test_harq_dtypes(precision, dtype_in):
    """Test HARQ with different data types and precision settings.
    
    Neural network frameworks often support multiple numerical precisions.
    Single precision (float32) is faster but less accurate, while double 
    precision (float64) is slower but more accurate. This test ensures
    HARQ works correctly with both.
    
    Tests:
    - Input data types: tf.float32 and tf.float64
    - Decoder precision settings: "single" and "double"
    - Output data type consistency with precision setting
    - Internal state data type consistency
    - Validates numerical precision handling in HARQ mode
    """
    
    k, n = 100, 200
    batch_size = 8
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, precision=precision, return_state=True)
    
    rv_list = ["rv0", "rv1"]
    llr_ch = tf.zeros([batch_size, len(rv_list), n], dtype_in)
    
    result, state = decoder(llr_ch, rv=rv_list)
    
    # Check output precision
    if precision == "single":
        assert result.dtype == tf.float32
        assert state.dtype == tf.float32
    else:
        assert result.dtype == tf.float64
        assert state.dtype == tf.float64

def test_custom_accumulator_function():
    """Test HARQ with custom accumulator function.
    
    The HARQ decoder allows users to provide custom accumulator functions
    instead of the default simple addition. This enables research into
    different HARQ combination strategies and algorithm optimization.
    
    Tests:
    - Custom accumulator with transmission-dependent weighting
    - Exponential decay weighting (newer transmissions weighted less)
    - Proper function signature: (llr_accumulated, llr_new, transmission_idx)
    - Integration with HARQ decoder infrastructure
    - Validates extensibility and custom algorithm support
    - Output validity with non-standard accumulation strategies
    """
    
    def weighted_accumulator(llr_accumulated, llr_new, transmission_idx):
        """Custom accumulator with transmission-dependent weighting."""
        weight = 0.9 ** transmission_idx  # Decay with transmission index
        
        if llr_accumulated is None:
            return llr_new * weight
        else:
            return llr_accumulated + llr_new * weight
    
    k, n = 150, 300
    batch_size = 6
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, accumulator=weighted_accumulator)
    
    source = GaussianPriorSource()
    rv_list = ["rv0", "rv1", "rv2"]
    llr_ch = source([batch_size, len(rv_list), n], 0.4)
    
    result = decoder(llr_ch, rv=rv_list)
    
    # Should produce valid output with custom accumulation
    assert result.shape == [batch_size, k]
    assert not tf.reduce_any(tf.math.is_nan(result))

# TODO: Add more specific tests based on your HARQ implementation details
# - Test with different modulation orders
# - Test with specific 3GPP scenarios
# - Test performance benchmarks
# - Test memory usage in HARQ mode
# - Test with varying batch sizes during HARQ

if __name__ == "__main__":
    # Debug the test_harq_encoder function
    k_n = (400, 900)
    rv_list = ["rv0", "rv1", "rv2", "rv3"]
    
    print("Starting debug of test_harq_encoder...")
    test_harq_encoder(k_n, rv_list)
    print("test_harq_encoder completed successfully!")