template<typename T>
inline constexpr T SQ(T const x) { return x * x; };

template<bool plus_direction, bool FD, typename T>
struct WENO5_Branch {
public:

    static constexpr int sz = 2;

    template<typename Ts>
    static inline __attribute__((always_inline)) T
    branch(std::array<Ts, 5> const &WENOVAR) {

        int constexpr minus2 = (plus_direction) ? 0 : 4;
        int constexpr minus1 = (plus_direction) ? 1 : 3;
        int constexpr plus0 = 2;
        int constexpr plus1 = (plus_direction) ? 3 : 1;
        int constexpr plus2 = (plus_direction) ? 4 : 0;


        const T dFD[3]{1. / 16., 10. / 16., 5. / 16.};

        const T dFV[3]{1. / 10., 3. / 5., 3. / 10.};

        T f[3];
        if constexpr (FD) {
            // Finite difference reconstruction

            f[0] = 3. / 8. * WENOVAR[minus2] - 10. / 8. * WENOVAR[minus1] +
                   15. / 8. * WENOVAR[plus0];
            f[1] = -1. / 8. * WENOVAR[minus1] + 6. / 8. * WENOVAR[plus0] +
                   3. / 8. * WENOVAR[plus1];
            f[2] = 3. / 8. * WENOVAR[plus0] + 6. / 8. * WENOVAR[plus1] -
                   1. / 8. * WENOVAR[plus2];
        } else {
            f[0] = 1. / 3. * WENOVAR[minus2] - 7. / 6. * WENOVAR[minus1] +
                   11. / 6. * WENOVAR[plus0];
            f[1] = -1. / 6. * WENOVAR[minus1] + 5. / 6. * WENOVAR[plus0] +
                   1. / 3. * WENOVAR[plus1];
            f[2] = 1. / 3. * WENOVAR[plus0] + 5. / 6. * WENOVAR[plus1] -
                   1. / 6. * WENOVAR[plus2];
        }

        // Smooth WENO weights: Note that these are from Del Zanna et al. 2007
        // (A.18)
	//
	        T beta[3];

        const T beta_coeff[2]{13. / 12., 0.25};

        beta[0] = beta_coeff[0] *
                  SQ(WENOVAR[minus2] + WENOVAR[plus0] - 2.0 * WENOVAR[minus1]) +
                  beta_coeff[1] * SQ(WENOVAR[minus2] - 4. * WENOVAR[minus1] +
                                     3. * WENOVAR[plus0]);

        beta[1] = beta_coeff[0] *
                  SQ(WENOVAR[minus1] + WENOVAR[plus1] - 2.0 * WENOVAR[plus0]) +
                  beta_coeff[1] * SQ(WENOVAR[minus1] - WENOVAR[plus1]);

        beta[2] = beta_coeff[0] *
                  SQ(WENOVAR[plus0] + WENOVAR[plus2] - 2.0 * WENOVAR[plus1]) +
                  beta_coeff[1] * SQ(3. * WENOVAR[plus0] - 4. * WENOVAR[plus1] +
                                     WENOVAR[plus2]);

        // Rescale epsilon
        //    constexpr double epsL = 1.e-42;
        const T epsL = 1.e-42;

        // WENO-Z+: Acker et al. 2016

        const T tau_5 = GReX::abs(beta[0] - beta[2]);

        const T indicator[3]{(tau_5) / (beta[0] + epsL), (tau_5) / (beta[1] + epsL),
                             (tau_5) / (beta[2] + epsL)};

        T alpha[3]{1. + SQ(indicator[0]), 1. + SQ(indicator[1]),
                   1. + SQ(indicator[2])};

        T alpha_sum = 0.;
        if constexpr (FD) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                alpha[i] *= dFD[i];
                alpha_sum += alpha[i];
            };
        } else { // FV

#pragma unroll
            for (int i = 0; i < 3; ++i) {
                alpha[i] *= dFV[i];
                alpha_sum += alpha[i];
            };
        }

        T flux = 0.;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            flux += f[i] * alpha[i]; // / alpha_sum;
        };

        return flux / alpha_sum;
    };
};
