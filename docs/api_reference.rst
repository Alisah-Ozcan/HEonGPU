.. _api_reference:

API Reference
=============

This section provides a high-level overview of the public C++ API of HEonGPU. The design is intended to be stable and intuitive, abstracting all underlying CUDA complexity and drawing inspiration from the user-friendly design of Microsoft SEAL. The API is organized within the ``heongpu`` namespace and is heavily templated to support different FHE schemes.

Core Enumerations
-----------------

**Scheme Selection**
    Used to specify the desired homomorphic encryption scheme during context creation.

    .. code-block:: cpp

        namespace heongpu {
            enum class Scheme { 
                BFV, 
                CKKS, 
                TFHE 
            };
        }

**Key-Switching Method**
    Selects the algorithm used for key-switching operations like relinearization and rotation.

    .. code-block:: cpp

        namespace heongpu {
            enum class keyswitching_type {
                KEYSWITCHING_METHOD_I,  // Based on Fan-Vercauteren
                KEYSWITCHING_METHOD_II, // Hybrid method
                KEYSWITCHING_METHOD_III // Key decomposition method (relinearization only)
            };
        }

Context and Parameter Classes
-----------------------------

**HEContext**
    The central class that manages the entire cryptographic environment. It must be the first object instantiated. It holds the encryption parameters, validates them, and pre-computes necessary data on the GPU.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEContext {
        public:
            // Constructor requires selecting a key-switching method.
            HEContext(keyswitching_type method);

            // Setters for core encryption parameters.
            void set_poly_modulus_degree(size_t degree);
            void set_coeff_modulus_default_values(int level);
            void set_plain_modulus(int plain_modulus); // For BFV

            // Finalizes the context and uploads pre-computations to the GPU.
            void generate();
        };

Key Management Classes
----------------------

**HEKeyGenerator**
    Generates all necessary cryptographic keys from a valid ``HEContext``.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEKeyGenerator {
        public:
            HEKeyGenerator(const HEContext<scheme_type>& context);

            void generate_secret_key(Secretkey<scheme_type>& sk);
            void generate_public_key(Publickey<scheme_type>& pk, const Secretkey<scheme_type>& sk);
            void generate_relin_key(Relinkey<scheme_type>& rlk, const Secretkey<scheme_type>& sk);
            void generate_galois_key(/* ... */); // For rotations
        };

**Key Storage Classes**
    These classes are containers for the key data, which resides primarily on the GPU.

    .. code-block:: cpp

        template <Scheme scheme_type> class Secretkey;
        template <Scheme scheme_type> class Publickey;
        template <Scheme scheme_type> class Relinkey;
        template <Scheme scheme_type> class Galoiskey;

Data and Cryptographic Classes
------------------------------

**Plaintext and Ciphertext**
    These classes hold the plaintext (encoded) and ciphertext data.

    .. code-block:: cpp

        template <Scheme scheme_type> class Plaintext;
        template <Scheme scheme_type> class Ciphertext;

**HEEncoder**
    Handles the conversion of user data into the plaintext polynomial format required for encryption.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEEncoder {
        public:
            HEEncoder(const HEContext<scheme_type>& context);

            // BFV: Encodes a vector of integers.
            void encode(Plaintext<Scheme::BFV>& ptxt, const std::vector<uint64_t>& message);

            // CKKS: Encodes a vector of doubles with a given scale.
            void encode(Plaintext<Scheme::CKKS>& ptxt, const Message<Scheme::CKKS>& msg, double scale);
            
            // Corresponding decode functions are also provided.
            void decode(std::vector<uint64_t>& message, const Plaintext<Scheme::BFV>& ptxt);
            void decode(Message<Scheme::CKKS>& msg, const Plaintext<Scheme::CKKS>& ptxt);
        };

**HEEncryptor and HEDecryptor**
    Perform the core cryptographic transformations.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEEncryptor {
        public:
            HEEncryptor(const HEContext<scheme_type>& context, const Publickey<scheme_type>& pk);
            void encrypt(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);
        };

        template <Scheme scheme_type>
        class HEDecryptor {
        public:
            HEDecryptor(const HEContext<scheme_type>& context, const Secretkey<scheme_type>& sk);
            void decrypt(Plaintext<scheme_type>& ptxt, const Ciphertext<scheme_type>& ctxt);
        };

**HEArithmeticOperator**
    The primary class for executing all homomorphic operations on the GPU.

    .. code-block:: cpp
       :linenos:

        template <Scheme scheme_type>
        class HEArithmeticOperator {
        public:
            HEArithmeticOperator(const HEContext<scheme_type>& context, const HEEncoder<scheme_type>& encoder);

            // In-place and out-of-place arithmetic operations
            void add(const Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2, Ciphertext<scheme_type>& result);
            void add_inplace(Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2);
            void multiply(const Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2, Ciphertext<scheme_type>& result);
            void multiply_inplace(Ciphertext<scheme_type>& ctxt1, const Ciphertext<scheme_type>& ctxt2);
            
            // Operations with plaintexts
            void add_plain_inplace(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);
            void multiply_plain_inplace(Ciphertext<scheme_type>& ctxt, const Plaintext<scheme_type>& ptxt);

            // Key-switching and modulus management operations
            void relinearize_inplace(Ciphertext<scheme_type>& ctxt, const Relinkey<scheme_type>& rlk);
            void rotate_rows_inplace(Ciphertext<scheme_type>& ctxt, int steps, const Galoiskey<scheme_type>& gk);
            void rescale_inplace(Ciphertext<Scheme::CKKS>& ctxt); // CKKS only
        };
