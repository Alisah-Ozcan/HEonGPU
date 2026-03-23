#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

/**
 * @brief Thin wrapper around HEArithmeticOperator that promotes the protected
 *        evaluate_poly method to public, enabling BSGS Chebyshev polynomial
 *        evaluation from user code.
 *
 * HEOperator<Scheme::CKKS>::evaluate_poly uses baby-step/giant-step internally
 * and matches the algorithm described in the target paper. By deriving from
 * HEArithmeticOperator (which already inherits evaluate_poly as protected),
 * this class exposes it as a single public call without re-implementing BSGS.
 */
class CKKSPolyEvaluator : public heongpu::HEArithmeticOperator<Scheme>
{
  public:
    CKKSPolyEvaluator(heongpu::HEContext<Scheme> ctx,
                      heongpu::HEEncoder<Scheme>& enc)
        : heongpu::HEArithmeticOperator<Scheme>(ctx, enc)
    {
    }

    /**
     * @brief Evaluate a Chebyshev polynomial on a ciphertext using BSGS.
     *
     * Both Polynomial and evaluate_poly are protected in HEOperator, so they
     * can only be accessed from within a derived class. This method bridges
     * user-supplied coefficients (from approximate_function) into the internal
     * BSGS evaluation without exposing the protected types.
     *
     * @param ct           Input ciphertext (values in [a, b])
     * @param target_scale Desired output scale
     * @param coeffs       Chebyshev coefficients c[0..degree]
     * @param degree       Polynomial degree
     * @param relin_key    Relinearization key
     * @param a            Interval lower bound (default -1)
     * @param b            Interval upper bound (default  1)
     */
    heongpu::Ciphertext<Scheme>
    eval_chebyshev(heongpu::Ciphertext<Scheme>& ct, double target_scale,
                   const std::vector<Complex64>& coeffs, int degree,
                   heongpu::Relinkey<Scheme>& relin_key, double a = -1.0,
                   double b = 1.0)
    {
        // Polynomial is protected in HEOperator — constructible only here
        Polynomial poly(degree, coeffs, /*lead=*/false,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        std::cout << "  Chebyshev poly degree=" << degree
                  << " depth=" << poly.depth() << " levels\n";
        return evaluate_poly(ct, target_scale, poly, relin_key,
                             heongpu::ExecutionOptions());
    }
};

// Forward declarations
heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector, int vec_len,
    heongpu::HEKeyGenerator<Scheme>& keygen,
    heongpu::Secretkey<Scheme>& secret_key, CKKSPolyEvaluator& evaluator,
    heongpu::HEEncoder<Scheme>& encoder, heongpu::HEContext<Scheme>& context,
    double scale);

/**
 * @brief Chebyshev sign approximation using BSGS polynomial evaluation.
 *
 * Uses heongpu::approximate_function to compute degree-D Chebyshev
 * coefficients for sign(x) on [-1,1], then evaluates via the built-in
 * baby-step/giant-step evaluate_poly (promoted via CKKSPolyEvaluator).
 *
 * Depth consumed: ceil(log2(D)) levels ≈ 11 for D=2048.
 *
 * For the paper's regime (N=64 elements, min pairwise diff ≈ 1/63):
 *   - D=2048 ensures correct sign classification for |x| ≥ 0.016
 *   - Scale: input differences in [-1,1] require no extra normalization
 *
 * @param ct_diff  Encrypted difference ct_col - ct_row, at some depth d
 * @param poly_eval CKKSPolyEvaluator (exposes evaluate_poly)
 * @param relin_key Relinearization key
 * @param scale     Encoding scale (must match current ciphertext scale)
 * @param degree    Chebyshev degree; 2048 matches the paper for N≤256
 * @return Ciphertext containing sign approximation values ≈ ±1
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree = 2048);

/**
 * @brief Sum rows of a matrix ciphertext
 *
 * Given a matrix stored as a flattened ciphertext (vec_len x vec_len),
 * aggregates each row by summing across columns.
 *
 * @param ct_matrix Encrypted matrix
 * @param vec_len Dimension (vec_len x vec_len matrix)
 * @param col_galois_key Galois key with column rotation shifts
 * @param evaluator Arithmetic operator
 * @param encoder Encoder for creating masks
 * @param context HE context
 * @param scale Encoding scale
 * @return Ciphertext with row sums at positions 0, vec_len, 2*vec_len, ...
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& col_galois_key,
        CKKSPolyEvaluator& evaluator, heongpu::HEEncoder<Scheme>& encoder,
        heongpu::HEContext<Scheme>& context, double scale);

/**
 * @brief Normalize a plaintext vector to [0,1] before encryption.
 *
 * This is a client-side operation that MUST be applied before encode/encrypt.
 * The HE ranking protocol requires that all pairwise differences (v[k] - v[j])
 * lie within [-1,1], which is the domain of the Chebyshev sign approximation.
 * Normalizing input to [0,1] guarantees this: the maximum possible difference
 * is 1 - 0 = 1. The server only ever receives normalized, encrypted data.
 *
 * @param input Raw values (any finite range; all-equal input is undefined)
 * @return Values linearly rescaled to [0,1] via min-max normalization
 */
std::vector<double> normalizeForRanking(const std::vector<double>& input)
{
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    std::vector<double> normalized(input.size());
    for (size_t i = 0; i < input.size(); i++)
        normalized[i] = (input[i] - lo) / range;
    return normalized;
}

/**
 * @brief Basic ranking: counts how many elements each value is greater than.
 *
 * Precondition: ct_vector must encrypt values in [0,1]. The client is
 * responsible for calling normalizeForRanking() before encode/encrypt.
 * Passing un-normalized data silently breaks the Chebyshev sign step.
 *
 * Depth budget (14 levels available with poly_modulus_degree=32768, 15 primes):
 *   depth 0  → fresh ciphertext
 *   depth 1  → transposeRowToColumn (multiply_plain mask + rescale)
 *   depth 1  → mod_drop ct_row for alignment
 *   depth 13 → chebyshev_sign_approx degree-2048 (12 levels via BSGS)
 *   depth 13 → add_plain +1, sumRows (rotations+adds, no level change)
 *   Total: 13 ≤ 14 ✓
 */
heongpu::Ciphertext<Scheme>
basicRank(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
          heongpu::Galoiskey<Scheme>& row_galois_key,
          heongpu::Galoiskey<Scheme>& col_galois_key,
          heongpu::HEKeyGenerator<Scheme>& keygen,
          heongpu::Secretkey<Scheme>& secret_key,
          heongpu::Relinkey<Scheme>& relin_key, CKKSPolyEvaluator& evaluator,
          heongpu::HEEncoder<Scheme>& encoder,
          heongpu::HEDecryptor<Scheme>& decryptor,
          heongpu::HEContext<Scheme>& context, double scale);

/**
 * @brief GPU-aware timer using CUDA Events for accurate GPU timing
 */
class GPUTimer
{
    cudaEvent_t start, stop;

  public:
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() { cudaEventRecord(start); }

    float stopTimer()
    {
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

    // ===== HE Context =====
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // poly_modulus_degree=32768 → 16,384 available slots = 128×128
    // This matches the paper's single-ciphertext limit of N=128 for ranking.
    const size_t poly_modulus_degree = 32768;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Q = 60 + 14×40 = 620 bits; P = 60 bits → Q_tilde = 680 bits
    // 680 < 881 = heongpu_128bit_std_parms(32768) → 128-bit security ✓
    // 15 primes in Q → 14 usable computation levels (14 rescales before
    // exhausting Q)
    context->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {60});
    double scale = pow(2.0, 40); // matches 40-bit computation primes
    context->generate();

    // ===== Key generation =====
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // ===== Encoder / Encryptor / Decryptor / Evaluator =====
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    // CKKSPolyEvaluator IS an HEArithmeticOperator (public inheritance) and
    // can be passed wherever HEArithmeticOperator& is expected.
    CKKSPolyEvaluator evaluator(context, encoder);

    // vec_len=64: uses 64×64=4,096 of 16,384 available slots.
    // Paper's single-ciphertext limit is N=128 (128²=16,384 exactly fills
    // the slot space). Upgrade to 128 by changing this single line.
    // min pairwise diff = 1/(vec_len-1) = 1/63 ≈ 0.016; degree-2048
    // Chebyshev correctly classifies sign for |x| ≥ 0.016. ✓
    const int vec_len = 64;

    // Raw input: any finite values are valid here.
    std::vector<double> input(vec_len);
    for (int i = 0; i < vec_len; i++)
        input[i] = static_cast<double>(i); // 0, 1, 2, ..., 63

    // Client-side normalization: map to [0,1] before encryption so that all
    // pairwise differences lie in [-1,1], matching the Chebyshev sign domain.
    // The server never sees raw values — only normalized, encrypted data.
    std::vector<double> normalized_input = normalizeForRanking(input);

    int matrix_slots  = vec_len * vec_len;
    int available_slots = poly_modulus_degree / 2;
    if (matrix_slots > available_slots)
    {
        std::cerr << "Error: Matrix size " << vec_len << "x" << vec_len
                  << " (" << matrix_slots << " slots) exceeds available slots ("
                  << available_slots << ")" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Matrix size: " << vec_len << "x" << vec_len << " (using "
              << matrix_slots << "/" << available_slots << " slots)\n";

    // ===== Galois key generation =====
    // Row shifts for replicateRow: -(2^i * vec_len) for i = log2(vec_len/2)..0
    std::vector<int> row_galois_shifts;
    for (int i = vec_len / 2; i > 0; i = i / 2)
        row_galois_shifts.push_back(-(i * vec_len));

    std::cout << "Generating " << row_galois_shifts.size()
              << " Galois keys for row rotations...\n";
    GPUTimer keygen_timer;
    keygen_timer.startTimer();
    heongpu::Galoiskey<Scheme> row_galois_key(context, row_galois_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);
    std::cout << "row Galois key generation took: " << keygen_timer.stopTimer()
              << " ms\n";

    // Column shifts: -1,-2,-4,...,-(vec_len/2) for replicateColumn
    //                +1,+2,+4,...,+(vec_len/2) for sumRows
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i *= 2)
        col_galois_shifts.push_back(-i);
    for (int step = 1; step < vec_len; step *= 2)
        col_galois_shifts.push_back(step);

    std::cout << "Generating " << col_galois_shifts.size()
              << " Galois keys for column rotations...\n";
    GPUTimer keygen_timer_col;
    keygen_timer_col.startTimer();
    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);
    std::cout << "column Galois key generation took: "
              << keygen_timer_col.stopTimer() << " ms\n";

    // Relinearization key for polynomial evaluation inside chebyshev_sign_approx
    std::cout << "Generating relinearization key...\n";
    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);
    std::cout << "relinearization key generation complete.\n";

    // ===== Row Replication Test =====
    std::cout << "\n=== Homomorphic Row Replication ===\n";
    std::cout << "Original input vector: ";
    display_vector(input, vec_len);
    std::cout << "Normalized input (sent to server): ";
    display_vector(normalized_input, vec_len);

    std::vector<double> row_initial(poly_modulus_degree / 2, 0.0);
    for (int i = 0; i < vec_len; i++)
        row_initial[i] = normalized_input[i];

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    GPUTimer timer;
    timer.startTimer();
    heongpu::Ciphertext<Scheme> row_replicated =
        replicateRow(ciphertext, vec_len, row_galois_key, evaluator);
    float time_ms = timer.stopTimer();
    std::cout << "row replication took: " << time_ms << " ms\n";

    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, row_replicated);
    std::vector<double> row_result;
    encoder.decode(row_result, decrypted_ciphertext);

    std::cout << "Replicated row vector (first " << vec_len * vec_len
              << " slots):\n";
    display_vector(row_result, vec_len * vec_len);

    // ===== transposeRowToColumn Test =====
    std::cout << "\n=== Testing transposeRowToColumn ===\n";

    std::vector<double> test_row(poly_modulus_degree / 2, 0.0);
    int test_len = std::min(8, vec_len);
    for (int i = 0; i < test_len; i++)
        test_row[i] = i + 1;

    std::cout << "Test input (first " << test_len << " elements): ";
    display_vector(test_row, test_len);

    heongpu::Plaintext<Scheme> test_plaintext(context);
    encoder.encode(test_plaintext, test_row, scale);
    heongpu::Ciphertext<Scheme> test_ciphertext(context);
    encryptor.encrypt(test_ciphertext, test_plaintext);

    GPUTimer transpose_timer;
    transpose_timer.startTimer();
    heongpu::Ciphertext<Scheme> transposed = transposeRowToColumn(
        test_ciphertext, vec_len, keygen, secret_key, evaluator, encoder,
        context, scale);
    float transpose_time = transpose_timer.stopTimer();
    std::cout << "transposeRowToColumn took: " << transpose_time << " ms\n";

    heongpu::Plaintext<Scheme> transposed_plaintext(context);
    decryptor.decrypt(transposed_plaintext, transposed);
    std::vector<double> transposed_result;
    encoder.decode(transposed_result, transposed_plaintext);

    std::cout << "Transposed result (should show column pattern):\n";
    for (int i = 0; i < test_len; i++)
    {
        std::cout << "  Element " << (i + 1) << " at position "
                  << (i * vec_len) << ": "
                  << transposed_result[i * vec_len] << "\n";
    }

    // ===== basicRank Test =====
    std::cout << "\n=== Testing basicRank Function ===\n";

    GPUTimer rank_timer;
    rank_timer.startTimer();
    heongpu::Ciphertext<Scheme> ct_rank = basicRank(
        ciphertext, vec_len, row_galois_key, col_galois_key, keygen,
        secret_key, relin_key, evaluator, encoder, decryptor, context, scale);
    float rank_time = rank_timer.stopTimer();
    std::cout << "basicRank computation took: " << rank_time << " ms\n";

    heongpu::Plaintext<Scheme> rank_plaintext(context);
    decryptor.decrypt(rank_plaintext, ct_rank);
    std::vector<double> rank_result;
    encoder.decode(rank_result, rank_plaintext);

    std::cout << "\nRanking results:\n";
    std::cout << "Input vector:  ";
    display_vector(input, vec_len);
    // Raw value at k*vec_len ≈ 2*rank[k] + 1  →  rank[k] ≈ (raw - 1) / 2
    std::cout << "Rank (count of smaller elements, decoded from raw HE output):\n";
    for (int i = 0; i < vec_len; i++)
    {
        double decoded_rank = (rank_result[i * vec_len] - 1.0) / 2.0;
        std::cout << "  input[" << i << "] = " << input[i]
                  << " -> rank = " << decoded_rank << "\n";
    }

    // Verification: for sorted input input[i] = i/(vec_len-1), rank = i
    std::cout << "\nVerification (expected rank = index):\n";
    bool all_correct = true;
    for (int i = 0; i < vec_len; i++)
    {
        double expected_rank = static_cast<double>(i);
        double actual_rank   = (rank_result[i * vec_len] - 1.0) / 2.0;
        double error         = std::abs(actual_rank - expected_rank);
        bool is_correct      = (error < 0.5);
        std::cout << "  Element " << (i + 1) << ": expected=" << expected_rank
                  << ", actual=" << actual_rank
                  << (is_correct ? "" : " INCORRECT") << "\n";
        if (!is_correct)
            all_correct = false;
    }
    std::cout << (all_correct ? "\nAll ranking results correct!\n"
                              : "\nSome ranking results are incorrect!\n");

    return EXIT_SUCCESS;
}

// The core idea: manipulate the encrypted vector so that only a single
// evaluation of the comparison function is needed to compare all values.
// For vector v = (v1,v2,...,vN): produce
//   vR = (v1,v2,...,vN, v1,v2,...,vN, ...)  [row replication]
//   vC = (v1,v1,...,v1, v2,v2,...,v2, ...)  [column replication]

/**
 * @brief Replicates a row vector homomorphically using logarithmic rotations.
 */
heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
             heongpu::Galoiskey<Scheme>& galois_key,
             CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> row_replicated = row_initial;

    std::cout << "Applying logarithmic rotations: ";
    for (int i = vec_len / 2; i > 0; i = i / 2)
    {
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
 * @brief Replicates a column vector homomorphically using logarithmic
 * rotations.
 */
heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
                heongpu::Galoiskey<Scheme>& galois_key,
                CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> col_replicated = col_initial;

    std::cout << "Applying logarithmic rotations: ";
    for (int i = 1; i < vec_len; i *= 2)
    {
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
 * @brief Transposes a row vector to a column vector homomorphically.
 *
 * Algorithm (Section 2.3, Algorithm 1):
 * for i = 1,...,⌈log N⌉: X ← X + (X ≫ N(N-1)/2^i)
 * X ← MaskC(X, 0)
 *
 * Depth consumed: 1 level (multiply_plain mask + rescale).
 */
heongpu::Ciphertext<Scheme>
transposeRowToColumn(const heongpu::Ciphertext<Scheme>& row_vector,
                     int vec_len, heongpu::HEKeyGenerator<Scheme>& keygen,
                     heongpu::Secretkey<Scheme>& secret_key,
                     CKKSPolyEvaluator& evaluator,
                     heongpu::HEEncoder<Scheme>& encoder,
                     heongpu::HEContext<Scheme>& context, double scale)
{
    heongpu::Ciphertext<Scheme> result = row_vector;

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    int N     = vec_len;

    // Generate Galois keys for transpose right-rotations:
    //   shift = -N(N-1)/2^i for i=1..log2(N)
    std::vector<int> transpose_galois_shifts;
    for (int i = 1; i <= log_n; i++)
    {
        int shift = -((N * (N - 1)) / (1 << i));
        transpose_galois_shifts.push_back(shift);
    }

    std::cout << "Generating " << transpose_galois_shifts.size()
              << " Galois keys for transposeRowToColumn (shifts: ";
    for (int shift : transpose_galois_shifts)
        std::cout << shift << " ";
    std::cout << ")...\n";

    heongpu::Galoiskey<Scheme> galois_key(context, transpose_galois_shifts);
    keygen.generate_galois_key(galois_key, secret_key);

    std::cout << "Transposing row to column (log N = " << log_n << "):\n";
    for (int i = 1; i <= log_n; i++)
    {
        int shift = -((N * (N - 1)) / (1 << i));
        std::cout << "  Rotation step " << i << ": shift = " << shift << "\n";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }

    // MaskC(X, 0): keep only the first column (positions 0, N, 2N, ...)
    std::cout << "  Applying column mask...\n";
    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> mask_values(total_slots, 0.0);
    for (int row = 0; row < vec_len; row++)
        mask_values[row * vec_len] = 1.0;

    heongpu::Plaintext<Scheme> mask(context);
    encoder.encode(mask, mask_values, scale);
    evaluator.multiply_plain_inplace(result, mask);
    // rescale_required_=true after multiply_plain; rescale before any rotation
    evaluator.rescale_inplace(result); // depth 0 → 1
    std::cout << "  Transposition complete!\n";

    return result;
}

/**
 * @brief Chebyshev sign approximation using built-in BSGS evaluate_poly.
 *
 * Computes sign(x) ≈ Σ c_k T_k(x) for k=1,3,5,...,degree on [-1,1].
 * The Chebyshev series of the sign function contains only odd-degree terms.
 * heongpu::approximate_function computes minimax-quality coefficients via
 * Chebyshev interpolation.
 *
 * Depth: Polynomial::depth() = ceil(log2(degree)) levels consumed.
 *   degree=2048 → 11 levels; with scale management the total is ≤12 levels.
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree)
{
    std::cout << "  Computing degree-" << degree
              << " Chebyshev sign approximation...\n";

    // sign(x): +1 for x>0, -1 for x<0, 0 at x=0 (odd function)
    // approximate_function uses Chebyshev interpolation at degree+1 nodes
    auto sign_func = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0.0 ? 1.0 : (re < 0.0 ? -1.0 : 0.0), 0.0);
    };

    std::vector<Complex64> cheby_coeffs =
        heongpu::approximate_function(sign_func, -1.0, 1.0, degree);

    // Polynomial is protected in HEOperator — construction happens inside
    // CKKSPolyEvaluator::eval_chebyshev which has protected-member access
    return poly_eval.eval_chebyshev(ct_diff, scale, cheby_coeffs, degree,
                                    relin_key, /*a=*/-1.0, /*b=*/1.0);
}

/**
 * @brief Sum each row of a matrix ciphertext using logarithmic rotations.
 *
 * NOTE: Masking is intentionally skipped. At depth 13 (one prime remaining
 * before exhausting Q), a multiply_plain would require a rescale that uses
 * the last available level. Callers read rank_result[i*vec_len] directly.
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& col_galois_key,
        CKKSPolyEvaluator& evaluator, heongpu::HEEncoder<Scheme>& encoder,
        heongpu::HEContext<Scheme>& context, double scale)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;

    std::cout << "  Summing rows with logarithmic reductions (vec_len="
              << vec_len << ")...\n";

    for (int step = 1; step < vec_len; step *= 2)
    {
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, col_galois_key, step);
        evaluator.add_inplace(result, rotated);
    }

    return result;
}

/**
 * @brief Basic ranking: count how many elements are smaller than each element.
 */
heongpu::Ciphertext<Scheme>
basicRank(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
          heongpu::Galoiskey<Scheme>& row_galois_key,
          heongpu::Galoiskey<Scheme>& col_galois_key,
          heongpu::HEKeyGenerator<Scheme>& keygen,
          heongpu::Secretkey<Scheme>& secret_key,
          heongpu::Relinkey<Scheme>& relin_key, CKKSPolyEvaluator& evaluator,
          heongpu::HEEncoder<Scheme>& encoder,
          heongpu::HEDecryptor<Scheme>& decryptor,
          heongpu::HEContext<Scheme>& context, double scale)
{
    std::cout << "\n=== Basic Ranking ===\n";
    std::cout << "Computing rank for vector of length " << vec_len << "\n";

    // Step 1: row replication — position (k,j) = v[j]
    std::cout << "Step 1: Replicating vector as rows...\n";
    heongpu::Ciphertext<Scheme> ct_row =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);

    // Step 2: transpose to column then replicate — position (k,j) = v[k]
    // transposeRowToColumn rescales (depth 0→1)
    std::cout << "Step 2: Transposing to column and replicating...\n";
    heongpu::Ciphertext<Scheme> ct_col_transposed = transposeRowToColumn(
        ct_vector, vec_len, keygen, secret_key, evaluator, encoder, context,
        scale);
    heongpu::Ciphertext<Scheme> ct_col =
        replicateColumn(ct_col_transposed, vec_len, col_galois_key, evaluator);

    // Align ct_row depth to match ct_col (transposeRowToColumn rescaled: depth 0→1)
    evaluator.mod_drop_inplace(ct_row);

    // Step 3: diff[k,j] = v[k] - v[j] → sign > 0 when v[k] > v[j]
    std::cout << "Step 3: Computing differences (column - row = v[k] - v[j])...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_col, ct_row, ct_diff); // v[k] - v[j]: positive if v[k] > v[j]

    // Step 4: sign approximation using degree-2048 Chebyshev BSGS
    // ct_diff ∈ [-1,1] is guaranteed by the caller's normalizeForRanking()
    // precondition (input in [0,1] → diffs bounded to [-1,1]).
    // Depth consumed: 12 levels → total depth ≈ 13
    std::cout << "Step 4: Applying Chebyshev sign approximation (degree 2048)...\n";
    heongpu::Ciphertext<Scheme> ct_sign = chebyshev_sign_approx(
        ct_diff, evaluator, relin_key, scale, /*degree=*/2048);

    // Step 5: shift sign range [-1,1] to [0,2] by adding 1 (no level consumed)
    // Skip ÷2: at depth≈13, a multiply_plain would exhaust the last prime.
    // After sumRows: position k*vec_len ≈ 2*rank[k]+1; callers decode as (raw-1)/2.
    std::cout << "Step 5: Shifting sign to [0,2] (add 1, skip /2 to conserve levels)...\n";
    evaluator.add_plain_inplace(ct_sign, 1.0);

    // Step 6: sum rows to aggregate counts
    std::cout << "Step 6: Summing rows to get final ranks...\n";
    heongpu::Ciphertext<Scheme> ct_rank = sumRows(
        ct_sign, vec_len, col_galois_key, evaluator, encoder, context, scale);

    std::cout << "Ranking complete!\n";
    return ct_rank;
}
