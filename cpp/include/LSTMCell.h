#pragma once
#include "Matrix.h"

/**
 * Cache of intermediate values for one BPTT time-step.
 */
struct LSTMStepCache {
    Matrix h_prev;   // [H x B]
    Matrix c_prev;   // [H x B]
    Matrix xh;       // [(H+E) x B]  concatenated [h ; x]
    Matrix ft;       // forget gate post-activation
    Matrix it;       // input  gate post-activation
    Matrix ct;       // cell   gate post-activation (g in some texts)
    Matrix ot;       // output gate post-activation
    Matrix c_new;    // new cell state
};

/**
 * Single LSTM step: pure functions, no state.
 * Used by LSTMLayer to process one time-step across a whole batch.
 */
class LSTMCell {
public:
    /**
     * One forward step.
     * @param xh        [(H+E) x B]
     * @param h_prev    [H x B]
     * @param c_prev    [H x B]
     * @param Wf,Wi,Wc,Wo   [H x (H+E)]
     * @param bf,bi,bc,bo   [H x 1]
     * @param cache_out filled with intermediates for BPTT
     * @returns {h_new [H x B], c_new [H x B]}
     */
    static std::pair<Matrix, Matrix> forward(
        const Matrix& xh,
        const Matrix& h_prev,
        const Matrix& c_prev,
        const Matrix& Wf, const Matrix& Wi,
        const Matrix& Wc, const Matrix& Wo,
        const Matrix& bf, const Matrix& bi,
        const Matrix& bc, const Matrix& bo,
        LSTMStepCache& cache_out);

    /**
     * One backward step (BPTT through a single time step).
     * @param dh        gradient from future h, [H x B]
     * @param dc        gradient from future c, [H x B]  (modified in-place)
     * @param cache     intermediates from forward
     * @param Wf,Wi,Wc,Wo   weight matrices (read-only)
     * Accumulates into dWf,dWi,... and returns d_xh for the previous step.
     */
    static Matrix backward(
        const Matrix& dh,
        Matrix& dc,            // in/out
        const LSTMStepCache& cache,
        const Matrix& Wf, const Matrix& Wi,
        const Matrix& Wc, const Matrix& Wo,
        Matrix& dWf, Matrix& dWi, Matrix& dWc, Matrix& dWo,
        Matrix& dbf, Matrix& dbi, Matrix& dbc, Matrix& dbo,
        size_t B);
};