#include "ranking.h"
#include "utils-basics.h"
#include "utils-eval.h"
#include "utils-matrices.h"
#include "utils-ptxt.h"
#include <cassert>
#include <omp.h>


std::vector<double> rank(
    const std::vector<double> &vec,
    const bool fractional,
    const double epsilon
)
{
    if (fractional)
    {
        std::vector<double> ranking(vec.size(), 0.5);
        for (size_t i = 0; i < vec.size(); i++)
            for (size_t j = 0; j < vec.size(); j++)
            { 
                if (std::abs(vec[i] - vec[j]) <= epsilon)
                    ranking[i] += 0.5;
                else if (vec[i] > vec[j])
                    ranking[i] += 1.0;
            }
        return ranking;
    }
    else
    {
        std::vector<double> ranking(vec.size(), 1.0);
        for (size_t i = 0; i < vec.size(); i++)
            for (size_t j = 0; j < vec.size(); j++)
            { 
                if (std::abs(vec[i] - vec[j]) <= epsilon && i > j)
                    ranking[i] += 1.0;
                else if (vec[i] > vec[j])
                    ranking[i] += 1.0;
            }
        return ranking;
    }
}


Ciphertext<DCRTPoly> rank(
    Ciphertext<DCRTPoly> c,
    const size_t vectorLength,
    const double leftBoundC,
    const double rightBoundC,
    const uint32_t degreeC,
    const bool cmpGt
)
{
    if (!cmpGt)
    {
        c = compare(
            replicateRow(c, vectorLength),
            replicateColumn(transposeRow(c, vectorLength, true), vectorLength),
            leftBoundC, rightBoundC, degreeC
        );
    }
    else
    {
        c = compareGt(
            replicateRow(c, vectorLength),
            replicateColumn(transposeRow(c, vectorLength, true), vectorLength),
            leftBoundC, rightBoundC, degreeC,
            0.005
        );
    }

    c = sumRows(c, vectorLength);
    c = c + (!cmpGt ? 0.5 : 1.0);

    return c;
}


std::vector<Ciphertext<DCRTPoly>> rank(
    const std::vector<Ciphertext<DCRTPoly>> &c,
    const size_t subVectorLength,
    const double leftBoundC,
    const double rightBoundC,
    const uint32_t degreeC,
    const bool cmpGt,
    const bool complOpt
)
{
    const size_t numCiphertext = c.size();

    static std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    std::cout << "===================================\n";
    std::cout << "Replicate\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> replR(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> replC(numCiphertext);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 2; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replR[j] = replicateRow(c[j], subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else
            {
                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replC[j] = replicateColumn(transposeRow(c[j], subVectorLength, true), subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
    
    if (!complOpt)
    {
        std::cout << "===================================\n";
        std::cout << "Compare\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> C(numCiphertext);
        std::vector<bool> Cinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < numCiphertext; j++)
        {
            for (size_t k = 0; k < numCiphertext; k++)
            {
                #pragma omp critical
                {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> Cjk;
                if (!cmpGt)
                {
                    Cjk = compare(
                        replR[j],
                        replC[k],
                        leftBoundC, rightBoundC, degreeC
                    );
                }
                else
                {
                    Cjk = compareGt(
                        replR[j],
                        replC[k],
                        leftBoundC, rightBoundC, degreeC,
                        0.005
                    );
                }

                #pragma omp critical
                {
                if (!Cinitialized[j]) { C[j] = Cjk; Cinitialized[j] = true; }
                else                  { C[j] = C[j] + Cjk;                  }
                }
                
                #pragma omp critical
                {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

        std::cout << "===================================\n";
        std::cout << "Sum\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (size_t j = 0; j < numCiphertext; j++)
        {
            #pragma omp critical
            {std::cout << "SumRows - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

            s[j] = sumRows(C[j], subVectorLength) + (!cmpGt ? 0.5 : 1.0);
            
            #pragma omp critical
            {std::cout << "SumRows - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
        
        return s;
    }
    else
    {
        std::cout << "===================================\n";
        std::cout << "Compare\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> Cv(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> Ch(numCiphertext);
        std::vector<bool> Cvinitialized(numCiphertext, false);
        std::vector<bool> Chinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        const size_t numReqThreads = numCiphertext * (numCiphertext + 1) / 2;
        std::cout << "Number of required threads: " << numReqThreads << std::endl;

        // for (size_t j = 0; j < numCiphertext; j++)
        // {
        //     for (size_t k = j; k < numCiphertext; k++)
        //     {
        // Collapse(2) with two nested for-loops creates issues here.
        #pragma omp parallel for
        for (size_t i = 0; i < numReqThreads; i++)
        {
            // Computing the indeces
            size_t j = 0, k = 0, counter = 0;
            bool loopCond = true;
            for (j = 0; j < numCiphertext && loopCond; j++)
                for (k = j; k < numCiphertext && loopCond; k++)
                    if (counter++ == i) loopCond = false;
            j--; k--;

            #pragma omp critical
            {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

            Ciphertext<DCRTPoly> Cjk;
            if (!cmpGt)
            {
                Cjk = compare(
                    replR[j],
                    replC[k],
                    leftBoundC, rightBoundC, degreeC
                );
            }
            else
            {
                Cjk = compareGt(
                    replR[j],
                    replC[k],
                    leftBoundC, rightBoundC, degreeC,
                    0.001
                );
            }

            #pragma omp critical
            {
            if (!Cvinitialized[j]) { Cv[j] = Cjk; Cvinitialized[j] = true; }
            else                   { Cv[j] = Cv[j] + Cjk;                  }
            }

            if (j != k)
            {
                Ciphertext<DCRTPoly> Ckj = 1.0 - Cjk;

                #pragma omp critical
                {
                if (!Chinitialized[k]) { Ch[k] = Ckj; Chinitialized[k] = true; }
                else                   { Ch[k] = Ch[k] + Ckj;                  }
                }
            }
            
            #pragma omp critical
            {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

        std::cout << "===================================\n";
        std::cout << "Sum\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> sv(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> sh(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);
        std::vector<bool> sinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2)
        for (size_t loopID = 0; loopID < 2; loopID++)
        {
            for (size_t j = 0; j < numCiphertext; j++)
            {
                if (loopID == 0)
                {
                    #pragma omp critical
                    {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                    sv[j] = sumRows(Cv[j], subVectorLength, true);
                    
                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = sv[j] + (!cmpGt ? 0.5 : 1.0); sinitialized[j] = true; }
                    else                  { s[j] = s[j] + sv[j];                                             }
                    }

                    #pragma omp critical
                    {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
                else
                {
                    #pragma omp critical
                    {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                    if (j > 0)
                    {
                        sh[j] = sumColumns(Ch[j], subVectorLength, true);
                        sh[j] = transposeColumn(sh[j], subVectorLength);

                        #pragma omp critical
                        {
                        if (!sinitialized[j]) { s[j] = sh[j] + (!cmpGt ? 0.5 : 1.0); sinitialized[j] = true; }
                        else                  { s[j] = s[j] + sh[j];                                             }
                        }
                    }

                    #pragma omp critical
                    {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
        
        return s;
    }
    
}


Ciphertext<DCRTPoly> rankWithCorrection(
    Ciphertext<DCRTPoly> c,
    const size_t vectorLength,
    const double leftBoundC,
    const double rightBoundC,
    const uint32_t degreeC,
    const bool parallel
)
{
    Ciphertext<DCRTPoly> rr = replicateRow(c, vectorLength);
    Ciphertext<DCRTPoly> rc = replicateColumn(transposeRow(c, vectorLength, true), vectorLength);

    std::vector<double> triangularMask(vectorLength * vectorLength, 0.0);
    for (size_t i = 0; i < vectorLength; i++)
        for (size_t j = 0; j < vectorLength; j++)
            if (j >= i) triangularMask[i * vectorLength + j] = 1.0;

    Ciphertext<DCRTPoly> correctionOffset;

    if (parallel)
    {
        omp_set_max_active_levels(2);
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                c = compare(rr, rc, leftBoundC, rightBoundC, degreeC);
                c = sumRows(c, vectorLength);
            }

            #pragma omp section
            {
                Ciphertext<DCRTPoly> e = equal(rr, rc, leftBoundC, rightBoundC, degreeC - 1);
                correctionOffset = sumRows(e * triangularMask, vectorLength) - 0.5 * sumRows(e, vectorLength);
            }
        }
    }
    else
    {
        c = compare(rr, rc, leftBoundC, rightBoundC, degreeC);
        Ciphertext<DCRTPoly> e = 4 * (1 - c) * c;
        correctionOffset = sumRows(e * triangularMask, vectorLength) - 0.5 * sumRows(e, vectorLength);
        c = sumRows(c, vectorLength);
    }

    return c + correctionOffset;
}


std::vector<Ciphertext<DCRTPoly>> rankWithCorrection(
    const std::vector<Ciphertext<DCRTPoly>> &c,
    const size_t subVectorLength,
    const double leftBoundC,
    const double rightBoundC,
    const uint32_t degreeC
)
{
    // omp_set_max_active_levels(2);

    const size_t numCiphertext = c.size();

    static std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    std::cout << "===================================\n";
    std::cout << "Replicate\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> replR(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> replC(numCiphertext);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 2; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replR[j] = replicateRow(c[j], subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else
            {
                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replC[j] = replicateColumn(transposeRow(c[j], subVectorLength, true), subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

    std::cout << "===================================\n";
    std::cout << "Compare\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> Cv(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> Ch(numCiphertext);
    std::vector<bool> Cvinitialized(numCiphertext, false);
    std::vector<bool> Chinitialized(numCiphertext, false);

    std::vector<Ciphertext<DCRTPoly>> Ev(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> Eh(numCiphertext);
    std::vector<bool> Evinitialized(numCiphertext, false);
    std::vector<bool> Ehinitialized(numCiphertext, false);

    std::vector<Ciphertext<DCRTPoly>> E(numCiphertext);
    std::vector<bool> Einitialized(numCiphertext, false);

    std::vector<double> triangularMask(subVectorLength * subVectorLength, 0.0);
    for (size_t i = 0; i < subVectorLength; i++)
        for (size_t j = 0; j < subVectorLength; j++)
            if (j <= i) triangularMask[i * subVectorLength + j] = 1.0;

    start = std::chrono::high_resolution_clock::now();

    const size_t numReqThreads = numCiphertext * (numCiphertext + 1) / 2;
    std::cout << "Number of required threads: " << numReqThreads << std::endl;

    // for (size_t j = 0; j < numCiphertext; j++)
    // {
    //     for (size_t k = j; k < numCiphertext; k++)
    //     {
    // Collapse(2) with two nested for-loops creates issues here.
    #pragma omp parallel for
    for (size_t i = 0; i < numReqThreads; i++)
    {
        // Computing the indeces
        size_t j = 0, k = 0, counter = 0;
        bool loopCond = true;
        for (j = 0; j < numCiphertext && loopCond; j++)
            for (k = j; k < numCiphertext && loopCond; k++)
                if (counter++ == i) loopCond = false;
        j--; k--;

        #pragma omp critical
        {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

        Ciphertext<DCRTPoly> Cjk = compare(
            replR[j],
            replC[k],
            leftBoundC, rightBoundC, degreeC
        );

        Ciphertext<DCRTPoly> Ejk = 4 * (1 - Cjk) * Cjk;

        #pragma omp critical
        {
        if (!Cvinitialized[j]) { Cv[j] = Cjk; Cvinitialized[j] = true; }
        else                   { Cv[j] = Cv[j] + Cjk;                  }
        }

        #pragma omp critical
        {
        if (!Evinitialized[j]) { Ev[j] = Ejk; Evinitialized[j] = true; }
        else                   { Ev[j] = Ev[j] + Ejk;                  }
        }

        if (j == k)
        {
            #pragma omp critical
            {
            if (!Einitialized[j]) { E[j] = Ejk * triangularMask; Einitialized[j] = true; }
            else                  { E[j] = E[j] + Ejk * triangularMask;                  }
            }
        }
        else
        {
            Ciphertext<DCRTPoly> Ckj = 1.0 - Cjk;

            #pragma omp critical
            {
            if (!Chinitialized[k]) { Ch[k] = Ckj; Chinitialized[k] = true; }
            else                   { Ch[k] = Ch[k] + Ckj;                  }
            }

            #pragma omp critical
            {
            if (!Ehinitialized[k]) { Eh[k] = Ejk; Ehinitialized[k] = true; }
            else                   { Eh[k] = Eh[k] + Ejk;                  }
            }

            #pragma omp critical
            {
            if (!Einitialized[k]) { E[k] = Ejk; Einitialized[k] = true; }
            else                  { E[k] = E[k] + Ejk;                  }
            }
        }
        
        #pragma omp critical
        {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

    std::cout << "===================================\n";
    std::cout << "Sum\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);
    std::vector<bool> sinitialized(numCiphertext, false);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 5; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> svj = sumRows(Cv[j], subVectorLength, true);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = svj; sinitialized[j] = true; }
                else                  { s[j] = s[j] + svj;                  }
                }

                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 1)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                if (j > 0)
                {
                    Ciphertext<DCRTPoly> shj = sumColumns(Ch[j], subVectorLength, true);
                    shj = transposeColumn(shj, subVectorLength);

                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = shj; sinitialized[j] = true; }
                    else                  { s[j] = s[j] + shj;                  }
                    }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 2)
            {
                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> evj = sumRows(Ev[j], subVectorLength, true);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = - 0.5 * evj; sinitialized[j] = true; }
                else                  { s[j] = s[j] - 0.5 * evj;                    }
                }

                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 3)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                if (j > 0)
                {
                    Ciphertext<DCRTPoly> ehj = sumColumns(Eh[j], subVectorLength, true);
                    ehj = transposeColumn(ehj, subVectorLength);

                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = - 0.5 * ehj; sinitialized[j] = true; }
                    else                  { s[j] = s[j] - 0.5 * ehj;                    }
                    }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 4)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}
                
                Ciphertext<DCRTPoly> ej = sumColumns(E[j], subVectorLength, true);
                ej = transposeColumn(ej, subVectorLength);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = ej; sinitialized[j] = true; }
                else                  { s[j] = s[j] + ej;                  }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
    
    return s;
    
}


std::vector<Ciphertext<DCRTPoly>> rankFG(
    const std::vector<Ciphertext<DCRTPoly>> &c,
    const size_t subVectorLength,
    const uint32_t dg,
    const uint32_t df,
    const bool cmpGt,
    const bool complOpt
)
{
    const size_t numCiphertext = c.size();

    static std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    std::cout << "===================================\n";
    std::cout << "Replicate\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> replR(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> replC(numCiphertext);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 2; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replR[j] = replicateRow(c[j], subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else
            {
                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replC[j] = replicateColumn(transposeRow(c[j], subVectorLength, true), subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
    
    if (!complOpt)
    {
        std::cout << "===================================\n";
        std::cout << "Compare\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> C(numCiphertext);
        std::vector<bool> Cinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < numCiphertext; j++)
        {
            for (size_t k = 0; k < numCiphertext; k++)
            {
                #pragma omp critical
                {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> Cjk;
                if (!cmpGt)
                {
                    Cjk = compareAdv(
                        replR[j],
                        replC[k],
                        dg, df
                    );
                }
                else
                {
                    Cjk = compareAdv(
                        replR[j],
                        replC[k] + 0.005,
                        dg, df
                    );
                }

                #pragma omp critical
                {
                if (!Cinitialized[j]) { C[j] = Cjk; Cinitialized[j] = true; }
                else                  { C[j] = C[j] + Cjk;                  }
                }
                
                #pragma omp critical
                {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

        std::cout << "===================================\n";
        std::cout << "Sum\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (size_t j = 0; j < numCiphertext; j++)
        {
            #pragma omp critical
            {std::cout << "SumRows - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

            s[j] = sumRows(C[j], subVectorLength) + (!cmpGt ? 0.5 : 1.0);
            
            #pragma omp critical
            {std::cout << "SumRows - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
        
        return s;
    }
    else
    {
        std::cout << "===================================\n";
        std::cout << "Compare\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> Cv(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> Ch(numCiphertext);
        std::vector<bool> Cvinitialized(numCiphertext, false);
        std::vector<bool> Chinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        const size_t numReqThreads = numCiphertext * (numCiphertext + 1) / 2;
        std::cout << "Number of required threads: " << numReqThreads << std::endl;

        // for (size_t j = 0; j < numCiphertext; j++)
        // {
        //     for (size_t k = j; k < numCiphertext; k++)
        //     {
        // Collapse(2) with two nested for-loops creates issues here.
        #pragma omp parallel for
        for (size_t i = 0; i < numReqThreads; i++)
        {
            // Computing the indeces
            size_t j = 0, k = 0, counter = 0;
            bool loopCond = true;
            for (j = 0; j < numCiphertext && loopCond; j++)
                for (k = j; k < numCiphertext && loopCond; k++)
                    if (counter++ == i) loopCond = false;
            j--; k--;

            #pragma omp critical
            {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

            Ciphertext<DCRTPoly> Cjk;
            if (!cmpGt)
            {
                Cjk = compareAdv(
                    replR[j],
                    replC[k],
                    dg, df
                );
            }
            else
            {
                Cjk = compareAdv(
                    replR[j],
                    replC[k] + 0.005,
                    dg, df
                );
            }

            #pragma omp critical
            {
            if (!Cvinitialized[j]) { Cv[j] = Cjk; Cvinitialized[j] = true; }
            else                   { Cv[j] = Cv[j] + Cjk;                  }
            }

            if (j != k)
            {
                Ciphertext<DCRTPoly> Ckj = 1.0 - Cjk;

                #pragma omp critical
                {
                if (!Chinitialized[k]) { Ch[k] = Ckj; Chinitialized[k] = true; }
                else                   { Ch[k] = Ch[k] + Ckj;                  }
                }
            }
            
            #pragma omp critical
            {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

        std::cout << "===================================\n";
        std::cout << "Sum\n";
        std::cout << "===================================\n";

        std::vector<Ciphertext<DCRTPoly>> sv(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> sh(numCiphertext);
        std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);
        std::vector<bool> sinitialized(numCiphertext, false);

        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2)
        for (size_t loopID = 0; loopID < 2; loopID++)
        {
            for (size_t j = 0; j < numCiphertext; j++)
            {
                if (loopID == 0)
                {
                    #pragma omp critical
                    {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                    sv[j] = sumRows(Cv[j], subVectorLength, true);
                    
                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = sv[j] + (!cmpGt ? 0.5 : 1.0); sinitialized[j] = true; }
                    else                  { s[j] = s[j] + sv[j];                                             }
                    }

                    #pragma omp critical
                    {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
                else
                {
                    #pragma omp critical
                    {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                    if (j > 0)
                    {
                        sh[j] = sumColumns(Ch[j], subVectorLength, true);
                        sh[j] = transposeColumn(sh[j], subVectorLength);

                        #pragma omp critical
                        {
                        if (!sinitialized[j]) { s[j] = sh[j] + (!cmpGt ? 0.5 : 1.0); sinitialized[j] = true; }
                        else                  { s[j] = s[j] + sh[j];                                             }
                        }
                    }

                    #pragma omp critical
                    {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
        
        return s;
    }
    
}


Ciphertext<DCRTPoly> rankWithCorrectionFG(
    Ciphertext<DCRTPoly> c,
    const size_t vectorLength,
    const uint32_t dg,
    const uint32_t df
)
{
    Ciphertext<DCRTPoly> rr = replicateRow(c, vectorLength);
    Ciphertext<DCRTPoly> rc = replicateColumn(transposeRow(c, vectorLength, true), vectorLength);

    std::vector<double> triangularMask(vectorLength * vectorLength, 0.0);
    for (size_t i = 0; i < vectorLength; i++)
        for (size_t j = 0; j < vectorLength; j++)
            if (j >= i) triangularMask[i * vectorLength + j] = 1.0;

    Ciphertext<DCRTPoly> correctionOffset;

    c = compareAdv(rr, rc, dg, df);
    Ciphertext<DCRTPoly> e = 4 * (1 - c) * c;
    correctionOffset = sumRows(e * triangularMask, vectorLength) - 0.5 * sumRows(e, vectorLength);
    c = sumRows(c, vectorLength);

    return c + correctionOffset;
}


std::vector<Ciphertext<DCRTPoly>> rankWithCorrectionFG(
    const std::vector<Ciphertext<DCRTPoly>> &c,
    const size_t subVectorLength,
    const uint32_t dg,
    const uint32_t df
)
{
    // omp_set_max_active_levels(2);

    const size_t numCiphertext = c.size();

    static std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    std::cout << "===================================\n";
    std::cout << "Replicate\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> replR(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> replC(numCiphertext);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 2; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replR[j] = replicateRow(c[j], subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else
            {
                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                replC[j] = replicateColumn(transposeRow(c[j], subVectorLength, true), subVectorLength);

                #pragma omp critical
                {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

    std::cout << "===================================\n";
    std::cout << "Compare\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> Cv(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> Ch(numCiphertext);
    std::vector<bool> Cvinitialized(numCiphertext, false);
    std::vector<bool> Chinitialized(numCiphertext, false);

    std::vector<Ciphertext<DCRTPoly>> Ev(numCiphertext);
    std::vector<Ciphertext<DCRTPoly>> Eh(numCiphertext);
    std::vector<bool> Evinitialized(numCiphertext, false);
    std::vector<bool> Ehinitialized(numCiphertext, false);

    std::vector<Ciphertext<DCRTPoly>> E(numCiphertext);
    std::vector<bool> Einitialized(numCiphertext, false);

    std::vector<double> triangularMask(subVectorLength * subVectorLength, 0.0);
    for (size_t i = 0; i < subVectorLength; i++)
        for (size_t j = 0; j < subVectorLength; j++)
            if (j <= i) triangularMask[i * subVectorLength + j] = 1.0;

    start = std::chrono::high_resolution_clock::now();

    const size_t numReqThreads = numCiphertext * (numCiphertext + 1) / 2;
    std::cout << "Number of required threads: " << numReqThreads << std::endl;

    // for (size_t j = 0; j < numCiphertext; j++)
    // {
    //     for (size_t k = j; k < numCiphertext; k++)
    //     {
    // Collapse(2) with two nested for-loops creates issues here.
    #pragma omp parallel for
    for (size_t i = 0; i < numReqThreads; i++)
    {
        // Computing the indeces
        size_t j = 0, k = 0, counter = 0;
        bool loopCond = true;
        for (j = 0; j < numCiphertext && loopCond; j++)
            for (k = j; k < numCiphertext && loopCond; k++)
                if (counter++ == i) loopCond = false;
        j--; k--;

        #pragma omp critical
        {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

        Ciphertext<DCRTPoly> Cjk = compareAdv(
            replR[j],
            replC[k],
            dg, df
        );

        Ciphertext<DCRTPoly> Ejk = 4 * (1 - Cjk) * Cjk;

        #pragma omp critical
        {
        if (!Cvinitialized[j]) { Cv[j] = Cjk; Cvinitialized[j] = true; }
        else                   { Cv[j] = Cv[j] + Cjk;                  }
        }

        #pragma omp critical
        {
        if (!Evinitialized[j]) { Ev[j] = Ejk; Evinitialized[j] = true; }
        else                   { Ev[j] = Ev[j] + Ejk;                  }
        }

        if (j == k)
        {
            #pragma omp critical
            {
            if (!Einitialized[j]) { E[j] = Ejk * triangularMask; Einitialized[j] = true; }
            else                  { E[j] = E[j] + Ejk * triangularMask;                  }
            }
        }
        else
        {
            Ciphertext<DCRTPoly> Ckj = 1.0 - Cjk;

            #pragma omp critical
            {
            if (!Chinitialized[k]) { Ch[k] = Ckj; Chinitialized[k] = true; }
            else                   { Ch[k] = Ch[k] + Ckj;                  }
            }

            #pragma omp critical
            {
            if (!Ehinitialized[k]) { Eh[k] = Ejk; Ehinitialized[k] = true; }
            else                   { Eh[k] = Eh[k] + Ejk;                  }
            }

            #pragma omp critical
            {
            if (!Einitialized[k]) { E[k] = Ejk; Einitialized[k] = true; }
            else                  { E[k] = E[k] + Ejk;                  }
            }
        }
        
        #pragma omp critical
        {std::cout << "(j, k) = (" << j << ", " << k << ") - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;

    std::cout << "===================================\n";
    std::cout << "Sum\n";
    std::cout << "===================================\n";

    std::vector<Ciphertext<DCRTPoly>> s(numCiphertext);
    std::vector<bool> sinitialized(numCiphertext, false);

    start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t loopID = 0; loopID < 5; loopID++)
    {
        for (size_t j = 0; j < numCiphertext; j++)
        {
            if (loopID == 0)
            {
                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> svj = sumRows(Cv[j], subVectorLength, true);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = svj; sinitialized[j] = true; }
                else                  { s[j] = s[j] + svj;                  }
                }

                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 1)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                if (j > 0)
                {
                    Ciphertext<DCRTPoly> shj = sumColumns(Ch[j], subVectorLength, true);
                    shj = transposeColumn(shj, subVectorLength);

                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = shj; sinitialized[j] = true; }
                    else                  { s[j] = s[j] + shj;                  }
                    }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 2)
            {
                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                Ciphertext<DCRTPoly> evj = sumRows(Ev[j], subVectorLength, true);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = - 0.5 * evj; sinitialized[j] = true; }
                else                  { s[j] = s[j] - 0.5 * evj;                    }
                }

                #pragma omp critical
                {std::cout << "SumV - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 3)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}

                if (j > 0)
                {
                    Ciphertext<DCRTPoly> ehj = sumColumns(Eh[j], subVectorLength, true);
                    ehj = transposeColumn(ehj, subVectorLength);

                    #pragma omp critical
                    {
                    if (!sinitialized[j]) { s[j] = - 0.5 * ehj; sinitialized[j] = true; }
                    else                  { s[j] = s[j] - 0.5 * ehj;                    }
                    }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
            else if (loopID == 4)
            {
                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}
                
                Ciphertext<DCRTPoly> ej = sumColumns(E[j], subVectorLength, true);
                ej = transposeColumn(ej, subVectorLength);
                
                #pragma omp critical
                {
                if (!sinitialized[j]) { s[j] = ej; sinitialized[j] = true; }
                else                  { s[j] = s[j] + ej;                  }
                }

                #pragma omp critical
                {std::cout << "SumH - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "COMPLETED (" << elapsed_seconds.count() << "s)" << std::endl << std::endl;
    
    return s;
    
}
