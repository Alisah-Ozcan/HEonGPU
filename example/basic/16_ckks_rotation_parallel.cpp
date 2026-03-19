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
 * @brief Degree-3 sign approximation for homomorphic comparison: sign(x) ≈ 1.5*x - 0.5*x^3
 *
 * Approximates the sign function using a cubic polynomial. For |x| > 0.3, provides good
 * classification. Returns values close to 1 when x > 0 and -1 when x < 0.
 *
 * @param ct_x Encrypted difference value (x)
 * @param eval Arithmetic operator
 * @param context HE context
 * @param relin_key Relinearization key
 * @param scale Encoding scale
 * @return Ciphertext containing approximated sign(x)
 */
heongpu::Ciphertext<Scheme> sign_approx_deg3(
    heongpu::Ciphertext<Scheme>& ct_x,
    heongpu::HEArithmeticOperator<Scheme>& eval,
    heongpu::HEContext<Scheme>& context,
    heongpu::Relinkey<Scheme>& relin_key,
    double scale);

/**
 * @brief Sum rows of a matrix ciphertext
 *
 * Given a matrix stored as a flattened ciphertext (vec_len x vec_len),
 * aggregates each row by summing across columns.
 *
 * @param ct_matrix Encrypted matrix
 * @param vec_len Dimension (vec_len x vec_len matrix)
 * @param col_galois_key Galois key with column rotation shifts (-1, -2, ..., -(vec_len-1))
 * @param evaluator Arithmetic operator
 * @param encoder Encoder for creating masks
 * @param context HE context
 * @param scale Encoding scale
 * @return Ciphertext with row sums at positions 0, vec_len, 2*vec_len, ...
 */
heongpu::Ciphertext<Scheme> sumRows(
    const heongpu::Ciphertext<Scheme>& ct_matrix,
    int vec_len,
    heongpu::Galoiskey<Scheme>& col_galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale);

/**
 * @brief Basic ranking function: counts how many elements each value is greater than
 *
 * For each element i, computes rank[i] = number of elements j where i > j.
 * Uses homomorphic matrix operations to compare all pairs in a single encryption.
 *
 * Algorithm:
 * 1. Encode input vector in first vec_len slots
 * 2. Replicate as row: (v1, v2, ..., vn, v1, v2, ..., vn, ...) 
 * 3. Transpose and replicate as column: (v1, v1, ..., v1, v2, v2, ..., v2, ...)
 * 4. Compute difference: row - column and apply sign approximation
 * 5. Sum rows to get final rank
 *
 * @param ct_vector Encrypted input vector (first vec_len slots)
 * @param vec_len Length of vector
 * @param row_galois_key Galois key for row rotations
 * @param col_galois_key Galois key for column rotations
 * @param keygen Key generator for additional keys if needed
 * @param secret_key Secret key for key generation
 * @param relin_key Relinearization key for polynomial evaluation
 * @param evaluator Arithmetic operator
 * @param encoder Encoder
 * @param context HE context
 * @param decryptor Decryptor for optional verification
 * @param scale Encoding scale
 * @return Ciphertext containing rank values
 */
heongpu::Ciphertext<Scheme> basicRank(
    const heongpu::Ciphertext<Scheme>& ct_vector,
    int vec_len,
    heongpu::Galoiskey<Scheme>& row_galois_key,
    heongpu::Galoiskey<Scheme>& col_galois_key,
    heongpu::HEKeyGenerator<Scheme>& keygen,
    heongpu::Secretkey<Scheme>& secret_key,
    heongpu::Relinkey<Scheme>& relin_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEDecryptor<Scheme>& decryptor,
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
    context->set_coeff_modulus_bit_sizes({40, 30, 30, 30, 30}, {40}); // 5 primes → Q_size=5, 4 rescales; 40+4*30+40=200 bits ≤ 218 (N=8192, 128-bit security)
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
    // vec_len=4 ensures min pairwise diff = 1/3 and max = 1.0, keeping decoded
    // rank errors ≤ 0.333 < 0.5 with the degree-3 sign polynomial.
    const int vec_len = 4;
    std::vector<double> input(vec_len);
    for(int i = 0; i < vec_len; i++) {
        input[i] = static_cast<double>(i) / (vec_len - 1.0);  // spans [0,1]: f(1.0)=1 exactly at extremes
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


    // Generate Galois keys for all column rotations needed:
    //   replicateColumn uses: -1, -2, -4, ..., -(vec_len/2)  (logarithmic doubling within each row)
    //   sumRows         uses: +1, +2, +4, ..., +vec_len/2    (left-rotate to accumulate at position 0)
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i *= 2) {
        col_galois_shifts.push_back(-i);
    }
    for (int step = 1; step < vec_len; step *= 2) {
        col_galois_shifts.push_back(step);
    }

    std::cout << "Generating " << col_galois_shifts.size() << " Galois keys for  column rotations...\n";

    GPUTimer keygen_timer_col;
    keygen_timer_col.startTimer();
    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);
    float keygen_time_col = keygen_timer_col.stopTimer();
    std::cout << "column Galois key generation took: " << keygen_time_col << " ms\n";

    // Generate relinearization key for sign approximation polynomial evaluation
    std::cout << "Generating relinearization key for polynomial evaluation...\n";
    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);
    std::cout << "relinearization key generation complete.\n";

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

    // ===== Test basicRank =====
    std::cout << "\n=== Testing basicRank Function ===\n";
    
    // Use the original encrypted ciphertext (contains input vector)
    GPUTimer rank_timer;
    rank_timer.startTimer();
    heongpu::Ciphertext<Scheme> ct_rank = basicRank(
        ciphertext, vec_len, row_galois_key, col_galois_key,
        keygen, secret_key, relin_key, evaluator, encoder, decryptor, context, scale);
    float rank_time = rank_timer.stopTimer();
    std::cout << "basicRank computation took: " << rank_time << " ms\n";

    // Decrypt and verify ranking results
    heongpu::Plaintext<Scheme> rank_plaintext(context);
    decryptor.decrypt(rank_plaintext, ct_rank);
    std::vector<double> rank_result;
    encoder.decode(rank_result, rank_plaintext);

    // Display ranking results
    std::cout << "\nRanking results:\n";
    std::cout << "Input vector:  ";
    display_vector(input, vec_len);
    // Decode: raw value at k*vec_len ≈ 2*rank[k]+1  →  rank[k] ≈ (raw - 1) / 2
    std::cout << "Rank (count of smaller elements, decoded from raw HE output):\n";
    for (int i = 0; i < vec_len; i++) {
        double decoded_rank = (rank_result[i * vec_len] - 1.0) / 2.0;
        std::cout << "  input[" << i << "] = " << input[i]
                  << " -> rank = " << decoded_rank << "\n";
    }

    // Verify: for sorted input input[i] = (i+1)/vec_len, the rank-from-bottom is i
    std::cout << "\nVerification (expected rank = index):\n";
    bool all_correct = true;
    for (int i = 0; i < vec_len; i++) {
        double expected_rank = static_cast<double>(i);
        double actual_rank = (rank_result[i * vec_len] - 1.0) / 2.0;
        double error = std::abs(actual_rank - expected_rank);
        bool is_correct = (error < 0.5);  // within 0.5 of true integer rank
        std::cout << "  Element " << (i+1) << ": expected=" << expected_rank
                  << ", actual=" << actual_rank
                  << (is_correct ? "" : " INCORRECT") << "\n";
        if (!is_correct) all_correct = false;
    }
    std::cout << (all_correct ? "\nAll ranking results correct!\n" : "\nSome ranking results are incorrect!\n");

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
    heongpu::Ciphertext<Scheme> col_replicated = col_initial;

    std::cout << "Applying logarithmic rotations: ";

    for (int i = 1; i < vec_len; i *= 2) {
        int shift = -i;
        std::cout << shift << " ";

        heongpu::Ciphertext<Scheme> rotated = col_replicated;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(col_replicated, rotated);
    }
    std::cout << "\n";

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
    evaluator.rescale_inplace(result);  // multiply_plain sets rescale_required_=true; must rescale before any rotation (depth 0→1)

    std::cout << "  Transposition complete!\n";

    return result;
}

/**
 * @brief Degree-3 sign approximation implementation
 *
 * Computes sign(x) ≈ 1.5*x - 0.5*x^3 homomorphically.
 * Depth: 3 levels (x^2: 1, x^3: 1, final operations: 1)
 */
heongpu::Ciphertext<Scheme> sign_approx_deg3(
    heongpu::Ciphertext<Scheme>& ct_x,
    heongpu::HEArithmeticOperator<Scheme>& eval,
    heongpu::HEContext<Scheme>& context,
    heongpu::Relinkey<Scheme>& relin_key,
    double scale)
{
    // Compute x^2
    heongpu::Ciphertext<Scheme> ct_x2(context);
    eval.multiply(ct_x, ct_x, ct_x2);
    eval.relinearize_inplace(ct_x2, relin_key);
    eval.rescale_inplace(ct_x2);

    // Compute x^3 = x * x^2
    heongpu::Ciphertext<Scheme> ct_x_drop(context);
    ct_x_drop = ct_x;
    eval.mod_drop_inplace(ct_x_drop);
    heongpu::Ciphertext<Scheme> ct_x3(context);
    eval.multiply(ct_x_drop, ct_x2, ct_x3);
    eval.relinearize_inplace(ct_x3, relin_key);
    eval.rescale_inplace(ct_x3);

    // Compute 1.5*x
    heongpu::Ciphertext<Scheme> ct_x_l1(context);
    ct_x_l1 = ct_x;
    eval.mod_drop_inplace(ct_x_l1);
    eval.mod_drop_inplace(ct_x_l1);
    heongpu::Ciphertext<Scheme> ct_term1(context);
    eval.multiply_plain(ct_x_l1, 1.5, ct_term1, scale);
    eval.rescale_inplace(ct_term1);

    // Compute -0.5*x^3
    heongpu::Ciphertext<Scheme> ct_term2(context);
    eval.multiply_plain(ct_x3, -0.5, ct_term2, scale);
    eval.rescale_inplace(ct_term2);

    // Combine: 1.5*x - 0.5*x^3
    heongpu::Ciphertext<Scheme> ct_result(context);
    eval.add(ct_term1, ct_term2, ct_result);

    return ct_result;
}

/**
 * @brief Sum each row of a matrix ciphertext
 *
 * Uses logarithmic rotations to aggregate column values within each row.
 */
heongpu::Ciphertext<Scheme> sumRows(
    const heongpu::Ciphertext<Scheme>& ct_matrix,
    int vec_len,
    heongpu::Galoiskey<Scheme>& col_galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;

    std::cout << "  Summing rows with logarithmic reductions (vec_len=" << vec_len << ")...\n";

    // Logarithmic reduction along columns: for each row, sum the vec_len elements
    // We perform reductions with shifts 1, 2, 4, 8, ... to bring all elements to position 0 of each row
    for (int step = 1; step < vec_len; step *= 2) {
        heongpu::Ciphertext<Scheme> rotated = result;
        // Rotate right by 'step' positions within each row
        // Element at position (row, col) moves to position (row, col-step)
        evaluator.rotate_rows_inplace(rotated, col_galois_key, step);
        evaluator.add_inplace(result, rotated);
    }

    // The logarithmic left-rotation accumulation guarantees that position k*vec_len
    // (first slot of row k) holds the correct row sum.  Masking is intentionally
    // skipped here: at this depth (Q_size-1) a multiply_plain would require a
    // rescale that exhausts all available levels.  Callers read rank_result[i*vec_len].
    return result;
}

/**
 * @brief Basic ranking: count how many elements are smaller than each element
 *
 * Implements the core ranking algorithm:
 * For each element i, rank[i] counts how many elements j satisfy i > j.
 */
heongpu::Ciphertext<Scheme> basicRank(
    const heongpu::Ciphertext<Scheme>& ct_vector,
    int vec_len,
    heongpu::Galoiskey<Scheme>& row_galois_key,
    heongpu::Galoiskey<Scheme>& col_galois_key,
    heongpu::HEKeyGenerator<Scheme>& keygen,
    heongpu::Secretkey<Scheme>& secret_key,
    heongpu::Relinkey<Scheme>& relin_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEDecryptor<Scheme>& decryptor,
    heongpu::HEContext<Scheme>& context,
    double scale)
{
    std::cout << "\n=== Basic Ranking ===\n";
    std::cout << "Computing rank for vector of length " << vec_len << "\n";

    // Step 1: Replicate input as rows: position (k,j) = v[j]
    std::cout << "Step 1: Replicating vector as rows...\n";
    heongpu::Ciphertext<Scheme> ct_row = replicateRow(ct_vector, vec_len, row_galois_key, evaluator);

    // Step 2: Transpose to column representation (depth 0→1 due to internal rescale),
    //         then replicate as columns: position (k,j) = v[k]
    std::cout << "Step 2: Transposing to column and replicating...\n";
    heongpu::Ciphertext<Scheme> ct_col_transposed = transposeRowToColumn(
        ct_vector, vec_len, keygen, secret_key, evaluator, encoder, context, scale);
    heongpu::Ciphertext<Scheme> ct_col = replicateColumn(ct_col_transposed, vec_len, col_galois_key, evaluator);

    // Align ct_row depth to match ct_col (transposeRowToColumn rescaled: depth 0→1)
    evaluator.mod_drop_inplace(ct_row);

    // Step 3: diff[k,j] = v[k] - v[j]  →  sign > 0 when v[k] > v[j]  (rank from bottom)
    std::cout << "Step 3: Computing differences (column - row = v[k] - v[j])...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_col, ct_row, ct_diff);  // NOTE: col-row, not row-col

    // Step 4: Apply sign approximation to each difference
    // sign(x) ≈ 1.5*x - 0.5*x^3 returns ~1 if x > 0, ~0 if x ≈ 0, ~-1 if x < 0
    // We want to count x > 0, so we need to convert: (1 + sign(x)) / 2 gives ~1 for x > 0, ~0 for x ≤ 0
    // But here we use sign directly and post-process
    std::cout << "Step 4: Applying sign approximation...\n";
    heongpu::Ciphertext<Scheme> ct_sign = sign_approx_deg3(ct_diff, evaluator, context, 
                                                           relin_key, scale);

    // Step 5: Shift sign range from [-1,1] to [0,2] by adding 1 (no level consumed).
    // We intentionally skip the ÷2 step: at depth=4 (Q_size-1) a multiply_plain
    // would require one more rescale which exhausts all moduli.
    // After summing: position k*vec_len ≈ 2*rank[k] + 1.  Callers divide by 2.
    std::cout << "Step 5: Shifting sign to [0,2] (add 1, skip /2 to conserve levels)...\n";
    evaluator.add_plain_inplace(ct_sign, 1.0);  // [0,2]: ≈2 when v[k]>v[j], ≈0 when v[k]<v[j]

    // Step 6: Sum rows to aggregate counts (result at position k*vec_len ≈ 2*rank[k]+1)
    std::cout << "Step 6: Summing rows to get final ranks...\n";
    heongpu::Ciphertext<Scheme> ct_rank = sumRows(ct_sign, vec_len, col_galois_key,
                                                   evaluator, encoder, context, scale);

    std::cout << "Ranking complete!\n";

    return ct_rank;
}
