.. _appendix:

Appendix
========

This appendix contains supplementary material, including detailed performance benchmarks and instructions for citing the HEonGPU library in academic work.

Performance Benchmarks
----------------------

All benchmarks listed below were performed on an **NVIDIA RTX 4090 GPU** and are sourced from the project's main repository. Performance is highly dependent on the chosen parameters and hardware configuration.

**TFHE Unsigned Integer Arithmetic**

This table compares the latency (in milliseconds) of homomorphic unsigned integer addition between HEonGPU, TFHE-rs, and results from recent academic literature.

.. list-table:: TFHE Unsigned Integer Arithmetic Latency (ms)
   :widths: 14 14 14 14 14 14 14
   :header-rows: 1

   * - Library
     - uint8
     - uint16
     - uint32
     - uint64
     - uint128
     - uint256
   * - `TFHE-rs <https://github.com/zama-ai/tfhe-rs>`_
     - 31.53
     - 31.54
     - 31.55
     - 32.03
     - 33.74
     - 58.32
   * - `Literature <https://tches.iacr.org/index.php/TCHES/article/view/11931>`_
     - 18.63
     - 18.61
     - 18.87
     - 24.23
     - 29.97
     - 58.30
   * - **HEonGPU**
     - **12.72**
     - **12.75**
     - **13.60**
     - **15.88**
     - **23.10**
     - **38.24**

**CKKS Bootstrapping**

This table details the execution times for various CKKS bootstrapping operations implemented in HEonGPU.

* **LKM (Less Key Mode)**: An optimization that reduces the number of required Galois keys by 30% at the cost of a 15-20% performance decrease. This is useful when GPU memory is a constraint.
* **Amortized Time**: The total execution time divided by the number of slots, providing a per-slot performance metric.

.. list-table:: CKKS Bootstrapping Performance
   :widths: 20 10 15 10 15 15 15
   :header-rows: 1

   * - Bootstrapping Type
     - :math:`N`
     - Slot Count
     - LKM
     - Remaining Level
     - Total Time
     - Amortized Time
   * - **Slim**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - ON
     - 4
     - 164.20 ms
     - 5.01 µs
   * - **Bit**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - OFF
     - 6
     - 55.66 ms
     - 3.40 µs
   * - **Bit**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - OFF
     - 4
     - 115.88 ms
     - 3.53 µs
   * - **Gate\***
     - :math:`2^{15}`
     - :math:`2^{14}`
     - OFF
     - 0
     - 27.03 ms
     - 1.64 µs
   * - **Gate\***
     - :math:`2^{16}`
     - :math:`2^{15}`
     - OFF
     - 0
     - 70.73 ms
     - 2.16 µs

\*For all logic gates.

How to Cite HEonGPU
-------------------

Please use the following BibTeX entries to cite HEonGPU and its associated research in academic papers.

Main Library Paper
^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

    @misc{cryptoeprint:2024/1543,
          author = {Ali Şah Özcan and Erkay Savaş},
          title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
          howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
          year = {2024},
          url = {https://eprint.iacr.org/2024/1543}
    }

Key-Switching Optimizations Paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

    @misc{cryptoeprint:2025/124,
          author = {Ali Şah Özcan and Erkay Savaş},
          title = {{GPU} Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes},
          howpublished = {Cryptology {ePrint} Archive, Paper 2025/124},
          year = {2025},
          url = {https://eprint.iacr.org/2025/124}
    }
