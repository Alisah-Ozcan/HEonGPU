#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

// Forward declarations
heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator);

heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator);

heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector,
    int vec_len,
    heongpu::HEKeyGenerator<Scheme>& keygen,
    heongpu::Secretkey<Scheme>& secret_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale);

/**
 * @brief GPU-aware timer using CUDA Events for accurate GPU timing
 */
class GPUTimer {
    cudaEvent_t start, stop;
public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start);
    }

    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

int main()
{
    cudaSetDevice(0);

    // HE Context initialisieren
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();
    // Definiert die Menge an Werten, die verschlüsselt werden können. (polymod_degree / 2)
    const size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60}); // Definiert die multplikative Tiefe. Hier 3 Mult. möglich.
    double scale = pow(2.0, 30); // parameter defining encoding precision
    context->generate();

    // Schlüsselerzeugung
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Encoder, Encryptor, Decryptor
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    // Vektor befüllen mit maximal möglichen Elementen.
    // Rest soll dynamisch an vec_len angepasst werden.
    const int vec_len = sqrt(poly_modulus_degree / 2);
    std::vector<double> input(vec_len);
    for(int i = 0; i < vec_len; i++) {
        input[i] = i + 1;  // [1, 2, 3, ..., 64]
    }

    // Calculate total slots needed for the matrix
    int matrix_slots = vec_len * vec_len;  // n = √n × √n

    // Validate that matrix fits in available slots
    int available_slots = poly_modulus_degree / 2;
    if (matrix_slots > available_slots) {
        std::cerr << "Error: Matrix size " << vec_len << "x" << vec_len
                << " (" << matrix_slots << " slots) exceeds available slots ("
                << available_slots << ")" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Matrix size: " << vec_len << "x" << vec_len
              << " (using " << matrix_slots << "/" << available_slots << " slots)\n";

    // Generate Galois keys for all row rotations needed
    // For logarithmic row replication: need rotations -(vec_len/2)*vec_len, ..., -2*vec_len, -vec_len
    std::vector<int> row_galois_shifts;
    for (int i = vec_len/2; i > 0; i = i/2) {
        row_galois_shifts.push_back(-(i * vec_len));
    }

    std::cout << "Generating " << row_galois_shifts.size() << " Galois keys for row rotations...\n";

    GPUTimer keygen_timer;
    keygen_timer.startTimer();
    heongpu::Galoiskey<Scheme> row_galois_key(context, row_galois_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);
    float keygen_time = keygen_timer.stopTimer();
    std::cout << "row Galois key generation took: " << keygen_time << " ms\n";


    // Generate Galois keys for all column rotations needed 
    // For parallel column replication: need rotations -1, -2, ..., -(vec_len-1)
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i++) {
        col_galois_shifts.push_back(-(i));
    }

    std::cout << "Generating " << col_galois_shifts.size() << " Galois keys for  column rotations...\n";

    GPUTimer keygen_timer_col;
    keygen_timer_col.startTimer();
    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);
    float keygen_time_col = keygen_timer_col.stopTimer();
    std::cout << "column Galois key generation took: " << keygen_time_col << " ms\n";

    // we adopt the row-by-row approach [20],
    // which consists of concatenating each row into a single vector and then encrypting it. For a square matrix of size N, we have the requirement that N2 ≤ n/2, where n
    // is the ring dimension, otherwise multiple ciphertexts are needed to store the entire matrix.

    // ===== Row Replication via Homomorphic Rotations =====
    std::cout << "\n=== Homomorphic Row Replication ===\n";
    std::cout << "Original input vector: ";
    display_vector(input, vec_len);

    // Prepare initial row vector: first vec_len slots contain input, rest is zeros
    std::vector<double> row_initial(poly_modulus_degree / 2, 0.0);
    for (size_t i = 0; i < vec_len; i++) {
        row_initial[i] = input[i];
    }

    // Encode and encrypt the initial row
    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    // Benchmark the row replication
    GPUTimer timer;
    timer.startTimer();
    heongpu::Ciphertext<Scheme> row_replicated = replicateRow(ciphertext, vec_len, row_galois_key, evaluator);
    float time_ms = timer.stopTimer();
    std::cout << "row replication took: " << time_ms << " ms\n";

    // Decrypt and verify the row replication
    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, row_replicated);
    std::vector<double> row_result;
    encoder.decode(row_result, decrypted_ciphertext);

    std::cout << "Replicated row vector:\n";
    display_vector(row_result, vec_len * vec_len);

    // ===== Test transposeRowToColumn =====
    std::cout << "\n=== Testing transposeRowToColumn ===\n";

    // Create a simple test row vector for easy verification
    // Use first 8 elements for clarity: [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, ...]
    std::vector<double> test_row(poly_modulus_degree / 2, 0.0);
    int test_len = std::min(8, vec_len);  // Use 8 elements or vec_len, whichever is smaller
    for (int i = 0; i < test_len; i++) {
        test_row[i] = i + 1;
    }

    std::cout << "Test input (first " << test_len << " elements): ";
    display_vector(test_row, test_len);

    // Encode and encrypt
    heongpu::Plaintext<Scheme> test_plaintext(context);
    encoder.encode(test_plaintext, test_row, scale);
    heongpu::Ciphertext<Scheme> test_ciphertext(context);
    encryptor.encrypt(test_ciphertext, test_plaintext);

    // Apply transposeRowToColumn
    GPUTimer transpose_timer;
    transpose_timer.startTimer();
    heongpu::Ciphertext<Scheme> transposed = transposeRowToColumn(
        test_ciphertext, vec_len, keygen, secret_key, evaluator, encoder, context, scale);
    float transpose_time = transpose_timer.stopTimer();
    std::cout << "transposeRowToColumn took: " << transpose_time << " ms\n";

    // Decrypt and verify
    heongpu::Plaintext<Scheme> transposed_plaintext(context);
    decryptor.decrypt(transposed_plaintext, transposed);
    std::vector<double> transposed_result;
    encoder.decode(transposed_result, transposed_plaintext);

    // Display result - should show column pattern: elements at positions 0, N, 2N, 3N, ...
    std::cout << "Transposed result (should show column pattern):\n";
    std::cout << "First " << vec_len * test_len << " slots:\n";
    for (int i = 0; i < test_len; i++) {
        std::cout << "  Element " << (i+1) << " at position " << (i * vec_len) << ": "
                  << transposed_result[i * vec_len] << "\n";
    }

    return EXIT_SUCCESS;
}



// The core idea of our design is to manipulate the encrypted vector in such a way that only a single evaluation of the comparison function is needed to compare all values
// vector v = (v1,v2,v3), we produce vR = (v1,v2,v3,v1,v2,v3,v1,v2,v3), vC = (v1,v1,v1,v2,v2,v2,v3,v3,v3).

/**
 * @brief Replicates a row vector homomorphically using parallel rotations
 *
 * Takes a ciphertext containing a vector in the first vec_len slots and replicates
 * it vec_len times using homomorphic rotations in parallel with multiple Galois keys.
 *
 * @param row_initial Encrypted vector to replicate (first vec_len slots contain data)
 * @param vec_len Length of the vector to replicate
 * @param galois_key Galois key containing all rotation shifts
 * @param evaluator Arithmetic operator for homomorphic operations
 * @return Ciphertext containing the replicated row pattern
 */
heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator)
{
    // Create the result starting with the original
    heongpu::Ciphertext<Scheme> row_replicated = row_initial;

    std::cout << "Applying logarithmic rotations: ";

    // Perform logarithmic rotations and accumulate
    for (int i = vec_len/2; i > 0; i = i/2) {
        int shift = -(i * vec_len);
        std::cout << shift << " ";

        heongpu::Ciphertext<Scheme> rotated = row_replicated;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(row_replicated, rotated);
    }
    std::cout << "\n";

    return row_replicated;
}


/**
 * @brief Replicates a column vector homomorphically using parallel rotations
 *
 * Takes a ciphertext containing a vector in the first vec_len slots and replicates
 * it vec_len times using homomorphic rotations in parallel with multiple Galois keys.
 *
 * @param col_initial Encrypted column to replicate (first vec_len slots contain data)
 * @param vec_len Length of the column to replicate
 * @param galois_key Galois key containing all rotation shifts
 * @param evaluator Arithmetic operator for homomorphic operations
 * @return Ciphertext containing the replicated column pattern
 */
heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator)
{
    // Create all rotations in parallel
    std::vector<heongpu::Ciphertext<Scheme>> vector_ciphertexts(vec_len);
    vector_ciphertexts[0] = col_initial;  // First one is the original

    std::cout << "Applying rotations: ";

    // Perform all rotations sequentially - todo: check rotation direction
    for (int i = 1; i < vec_len; i++) {
        vector_ciphertexts[i] = col_initial;
        int shift = -(i * vec_len);
        std::cout << shift << " ";
        evaluator.rotate_rows_inplace(vector_ciphertexts[i], galois_key, shift);
    }
    std::cout << "\n";

    // Sum all rotated vectors
    std::cout << "Summing all rotated vectors...\n";
    heongpu::Ciphertext<Scheme> col_replicated = vector_ciphertexts[0];
    for (int i = 1; i < vec_len; i++) {
        evaluator.add_inplace(col_replicated, vector_ciphertexts[i]);
    }

    return col_replicated;
}

/**
 * @brief Transposes a row vector to a column vector homomorphically
 *
 * Takes a ciphertext containing a vector encoded as a row (first vec_len slots)
 * and transposes it to a column representation using logarithmic rotations.
 *
 * Algorithm from paper (Section 2.3, Algorithm 1):
 * for i = 1,...,⌈log N⌉:
 *     X ← X + (X ≫ N(N-1)/2^i)
 * X ← MaskC(X, 0)
 *
 * @param row_vector Encrypted row vector (first vec_len slots contain data)
 * @param vec_len Length of the vector (must be power of 2)
 * @param galois_key Galois key containing all rotation shifts
 * @param evaluator Arithmetic operator for homomorphic operations
 * @param encoder Encoder for creating mask plaintexts
 * @param context HE context for creating plaintexts
 * @return Ciphertext containing the transposed column vector
 */
heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector,
    int vec_len,
    heongpu::HEKeyGenerator<Scheme>& keygen,
    heongpu::Secretkey<Scheme>& secret_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale)
{
    heongpu::Ciphertext<Scheme> result = row_vector;

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    int N = vec_len;

    // Generate Galois keys for transpose operation (RIGHT rotations)
    // For transposeRowToColumn: need shifts N(N-1)/2^i for i=1 to log2(N)
    std::vector<int> transpose_galois_shifts;
    for (int i = 1; i <= log_n; i++) {
        int shift = -((N * (N - 1)) / (1 << i));  // NEGATIVE
        transpose_galois_shifts.push_back(shift);
    }

    std::cout << "Generating " << transpose_galois_shifts.size()
              << " Galois keys for transposeRowToColumn (shifts: ";
    for (int shift : transpose_galois_shifts) {
        std::cout << shift << " ";
    }
    std::cout << ")...\n";

    heongpu::Galoiskey<Scheme> galois_key(context, transpose_galois_shifts);
    keygen.generate_galois_key(galois_key, secret_key);

    std::cout << "Transposing row to column (log N = " << log_n << "):\n";

    // Perform logarithmic RIGHT rotations: X ← X + (X ≫ N(N-1)/2^i)
    // Negative shift = right rotation in HEonGPU
    for (int i = 1; i <= log_n; i++) {
        int shift = -((N * (N - 1)) / (1 << i));  // NEGATIVE for right rotation

        std::cout << "  Rotation step " << i << ": shift = " << shift << "\n";

        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }

    // MaskC(X, 0) - Keep only the first column (elements at positions 0, N, 2N, ...)
    std::cout << "  Applying column mask...\n";

    // Create mask: 1 at positions 0, N, 2N, 3N, ..., rest 0
    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> mask_values(total_slots, 0.0);

    for (int row = 0; row < vec_len; row++) {
        mask_values[row * vec_len] = 1.0;  // First column of each row
    }

    heongpu::Plaintext<Scheme> mask(context);
    encoder.encode(mask, mask_values, scale);
    evaluator.multiply_plain_inplace(result, mask);

    std::cout << "  Transposition complete!\n";

    return result;
}
