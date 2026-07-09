#include "LSTMCell.h"
#include "Activations.h"
#include <cassert>

std::pair<Matrix, Matrix> LSTMCell::forward(
    const Matrix& xh,
    const Matrix& h_prev,
    const Matrix& c_prev,
    const Matrix& Wf, const Matrix& Wi,
    const Matrix& Wc, const Matrix& Wo,
    const Matrix& bf, const Matrix& bi,
    const Matrix& bc, const Matrix& bo,
    LSTMStepCache& cache)
{
    size_t H = h_prev.rows;
    size_t B = h_prev.cols;

    auto gate = [&](const Matrix& W, const Matrix& b) {
        Matrix z = W.matmul(xh);
        for (size_t i = 0; i < H; ++i)
            for (size_t bb = 0; bb < B; ++bb)
                z(i, bb) += b(i, 0);
        return z;
    };

    Matrix z_f = gate(Wf, bf);
    Matrix z_i = gate(Wi, bi);
    Matrix z_c = gate(Wc, bc);
    Matrix z_o = gate(Wo, bo);

    Matrix ft(H, B), it(H, B), ct(H, B), ot(H, B), c_new(H, B);
    for (size_t k = 0; k < H * B; ++k) {
        ft.data[k]    = Activations::sigmoid(z_f.data[k]);
        it.data[k]    = Activations::sigmoid(z_i.data[k]);
        ct.data[k]    = Activations::tanh_act(z_c.data[k]);
        ot.data[k]    = Activations::sigmoid(z_o.data[k]);
        c_new.data[k] = ft.data[k] * c_prev.data[k] + it.data[k] * ct.data[k];
    }

    Matrix h_new(H, B);
    for (size_t k = 0; k < H * B; ++k)
        h_new.data[k] = ot.data[k] * Activations::tanh_act(c_new.data[k]);

    cache.h_prev = h_prev.copy();
    cache.c_prev = c_prev.copy();
    cache.xh     = xh.copy();
    cache.ft     = ft;
    cache.it     = it;
    cache.ct     = ct;
    cache.ot     = ot;
    cache.c_new  = c_new;

    return {h_new, c_new};
}

Matrix LSTMCell::backward(
    const Matrix& dh,
    Matrix& dc,
    const LSTMStepCache& cache,
    const Matrix& Wf, const Matrix& Wi,
    const Matrix& Wc, const Matrix& Wo,
    Matrix& dWf, Matrix& dWi, Matrix& dWc, Matrix& dWo,
    Matrix& dbf, Matrix& dbi, Matrix& dbc, Matrix& dbo,
    size_t B)
{
    size_t H  = dh.rows;
    size_t HE = Wf.cols;

    Matrix do_t(H, B);
    for (size_t k = 0; k < H * B; ++k) {
        float tanh_c = Activations::tanh_act(cache.c_new.data[k]);
        do_t.data[k] = dh.data[k] * tanh_c
                     * Activations::sigmoid_deriv(cache.ot.data[k]);
    }

    for (size_t k = 0; k < H * B; ++k) {
        float tanh_c = Activations::tanh_act(cache.c_new.data[k]);
        dc.data[k] += dh.data[k] * cache.ot.data[k] * (1.f - tanh_c * tanh_c);
    }

    Matrix df_t(H, B), di_t(H, B), dct(H, B);
    for (size_t k = 0; k < H * B; ++k) {
        df_t.data[k] = dc.data[k] * cache.c_prev.data[k]
                     * Activations::sigmoid_deriv(cache.ft.data[k]);
        di_t.data[k] = dc.data[k] * cache.ct.data[k]
                     * Activations::sigmoid_deriv(cache.it.data[k]);
        dct.data[k]  = dc.data[k] * cache.it.data[k]
                     * Activations::tanh_deriv(cache.ct.data[k]);
    }

    for (size_t k = 0; k < H * B; ++k)
        dc.data[k] *= cache.ft.data[k];

    auto accum = [&](Matrix& dW, const Matrix& dgate) {
        for (size_t i = 0; i < H; ++i)
            for (size_t j = 0; j < HE; ++j)
                for (size_t b = 0; b < B; ++b)
                    dW(i, j) += dgate(i, b) * cache.xh(j, b);
    };
    accum(dWf, df_t);
    accum(dWi, di_t);
    accum(dWc, dct);
    accum(dWo, do_t);

    auto accum_b = [&](Matrix& db, const Matrix& dgate) {
        for (size_t i = 0; i < H; ++i)
            for (size_t b = 0; b < B; ++b)
                db(i, 0) += dgate(i, b);
    };
    accum_b(dbf, df_t);
    accum_b(dbi, di_t);
    accum_b(dbc, dct);
    accum_b(dbo, do_t);

    Matrix d_xh(HE, B, 0.f);

    auto add_contrib = [&](const Matrix& W, const Matrix& dg) {
        for (size_t i = 0; i < HE; ++i)
            for (size_t b = 0; b < B; ++b)
                for (size_t k = 0; k < H; ++k)
                    d_xh(i, b) += W(k, i) * dg(k, b);
    };
    add_contrib(Wf, df_t);
    add_contrib(Wi, di_t);
    add_contrib(Wc, dct);
    add_contrib(Wo, do_t);

    return d_xh;
}