#ifndef CNS_HYDRO_K_H_
#define CNS_HYDRO_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
cns_ctoprim (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& u,
             amrex::Array4<amrex::Real> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;
    using T = amrex::Real;

    Real rho = u(i,j,k,URHO);
    Real ux = u(i,j,k,UMX);
    Real uy = u(i,j,k,UMY);
    Real uz = u(i,j,k,UMZ);
    Real ei = u(i,j,k,UEDEN);
    ei *= rhoinv;


    if (u(i,j,k,URHO) < parm.smallr) {
      rho = parm.smallr;
      ei = 0;
      ux = 0;
      uy = 0;
      uz = 0;
    }

    T press = 0.;
    T epst;
    T press_prev = 1.e99;
    T press_pp = 1.e109;

    T rhob =parm.smallr;
    T eps;

    auto const rhostari = 1./rho;

    auto const Snorm = std::sqrt(ux * ux + uy * uy+ uz * uz)*rhostari;

    int nn = 0;
    T lorentz = 1.;
    T hWi =  1.;

    bool print = false;
//    while ((amrex::Math::abs(press - press_prev) > c2p_tol * press) && (nn < 1000)) {
    for(nn=0; nn< 20; ++nn){

      auto const hW = (ei + press) * rhostari + 1.;
      hWi = 1. / hW;


      auto const v = amrex::min( amrex::abs(Snorm*hWi), 0.9999499987499375);
      auto const v2 = v*v;

      auto const lorentz2i = amrex::Math::abs(1.-(v2));
      auto const lorentzi = amrex::Math::sqrt(lorentz2i);
      lorentz = 1./lorentzi;


      rhob = rho * lorentzi;

//      auto eps = Uout[RHOB]*(hW * lorentzi - 1.) - press;
      eps = ei * lorentz2i - (rho /(lorentz+1.) + press)*v2;
      eps =  amrex::max(eps, rho*eps_atmo);
      Uout[EPS] = eps;


      press_pp = press_prev;
      press_prev = press;

      // Need EOS call here!!
      press = eps*(parm.gamma-1.); //EOS_t::press_eps__rho_eps_th(eps, Uout[RHOB], Uout[EPS]);
      press = amrex::Math::abs(press); //std::max(-(U[RHOSTAR]+U[TAUENERGY]), press);

      //Aitken accelerator
      auto const R = (press - press_prev)/(press_prev - press_pp);
      auto const Paitken = std::abs(press_prev + (press-press_prev)/(1.-R));

     if(std::abs(R) < 1. && nn > 2){
	      press_pp = press_prev;
	      press_prev = press;
	      press=Paitken;
     };


      nn++;
    };

     auto const conv =  hWi * rhostari * lorentz;

    q(i,j,k,QRHO) = rhob;
    q(i,j,k,QU) = ux*conv;
    q(i,j,k,QV) = uy*conv;
    q(i,j,k,QW) = uz*conv;
    q(i,j,k,QEINT) = eps;
    q(i,j,k,QPRES) = press;
    auto rhoh  = eps*parm.gamma+Uout[QRHO];
    q(i,j,k,QCS) = parm.gamma*press/rhoh;
    q(i,j,k,QTEMP) = Real(0.0);
}

AMREX_GPU_DEVICE
inline
void
cns_flux_to_dudt (int i, int j, int k, int n,
                  amrex::Array4<amrex::Real> const& dudt,
                  amrex::Array4<amrex::Real const> const& fx,
                  amrex::Array4<amrex::Real const> const& fy,
                  amrex::Array4<amrex::Real const> const& fz,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    dudt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
        +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n))
        +           dxinv[2] * (fz(i,j,k,n) - fz(i,j,k+1,n));
}

namespace {


AMREX_GPU_DEVICE
void WENOZ(amrex::Real const w0, 
           amrex::Real const w1, 
	   amrex::Real const w2, 
	   amrex::Real const w3,
	   amrex::Real const w4, 
	   amrex::Real& result) noexcept {
	
	using T =amrex::Real;

//        int const minus2 = 0;
//        int const minus1 = 1;
//        int const plus0 = 2;
//        int const plus1 = 3;
//        int const plus2 = 4;

	auto const SQ = [] (auto const x) {return x*x;};


//        const T dFD[3]{1. / 16., 10. / 16., 5. / 16.};

	amrex::Real FV0 = (amrex::Real(1.) / amrex::Real(10.));
	amrex::Real FV1 = amrex::Real(amrex::Real(3.) / amrex::Real(5.));
	amrex::Real FV2 = amrex::Real(amrex::Real(3.) / amrex::Real(10.));

//       if constexpr (FD) {
//           // Finite difference reconstruction
//
//           f[0] = 3. / 8. * w0 - 10. / 8. * w1 +
//                  15. / 8. * w2;
//           f[1] = -1. / 8. * w1 + 6. / 8. * w2 +
//                  3. / 8. * w3;
//           f[2] = 3. / 8. * w2 + 6. / 8. * w3 -
//                  1. / 8. * w4;
//       } else {
	amrex::Real f0 =amrex::Real(amrex::Real( 1.)/amrex::Real(3.))* w0 - amrex::Real(amrex::Real(7.)/amrex::Real(6.))* w1 +
                  amrex::Real(amrex::Real(11.)/amrex::Real(6.))* w2;
	amrex::Real f1 =amrex::Real(amrex::Real(-1.)/amrex::Real(6.))* w1 + amrex::Real(amrex::Real(5.)/amrex::Real(6.))* w2 +
                  amrex::Real(amrex::Real( 1.)/amrex::Real(3.))* w3;
	amrex::Real f2 =amrex::Real(amrex::Real( 1.)/amrex::Real(3.))* w2  + amrex::Real(amrex::Real(5.)/amrex::Real(6.))* w3 -
                  amrex::Real(amrex::Real( 1.)/amrex::Real(6.))* w4;
//        }

        // Smooth WENO weights: Note that these are from Del Zanna et al. 2007
        // (A.18)
	//
	    
	amrex::Real beta_coeff0 = amrex::Real(13.) / amrex::Real(12.);
        amrex::Real beta_coeff1 =	amrex::Real(0.25);

	amrex::Real beta0 = beta_coeff0 *
                SQ(w0 + w2 - amrex::Real(2.0) * w1) +
                beta_coeff1 * SQ(w0 - amrex::Real(4.) * w1 +
                                   amrex::Real(3.) * w2);

	amrex::Real beta1 = beta_coeff0 *
                SQ(w1 + w3 - amrex::Real(2.0) * w2) +
                beta_coeff1 * SQ(w1 - w3);

	amrex::Real beta2 = beta_coeff0 *
                  SQ(w2 + w4 - amrex::Real(2.0) * w3) +
                  beta_coeff1 * SQ(amrex::Real(3.) * w2 - amrex::Real(4.) * w3 +
                                     w4);

        // Rescale epsilon
        //    constexpr amrex::Real epsL = 1.e-42;
        const amrex::Real epsL = amrex::Real(1.e-6);

        // WENO-Z+: Acker et al. 2016

        const T tau_5 = amrex::Math::abs(beta0 - beta2);

	T indicator0 =(tau_5) / (beta0 + epsL);
       	T indicator1 =(tau_5) / (beta1 + epsL);
        T indicator2 =(tau_5) / (beta2 + epsL);

	T alpha0 = amrex::Real(1.) + SQ(indicator0);
        T alpha1 = amrex::Real(1.) + SQ(indicator1);
        T alpha2 = amrex::Real(1.) + SQ(indicator2);

//        if constexpr (FD) {
//#pragma unroll
//            for (int i = 0; i < 3; ++i) {
//                alpha[i] *= dFD[i];
//                alpha_sum += alpha[i];
//            };
//        } else { // FV

	 alpha0 *= FV0;
	 alpha1 *= FV1;
	 alpha2 *= FV2;


	 T alpha_sum =  alpha0+ alpha1+ alpha2;

//        }



        result= (alpha0*f0 + alpha1*f1 + alpha2*f2) / alpha_sum;

	if(result != result)
	printf("%e %e %e %e %e \n", w0, w1,w2,w3,w4);

    };


}

AMREX_GPU_DEVICE
inline
void
cns_wenoz_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dqL,
             amrex::Array4<amrex::Real> const& dqR,
             amrex::Array4<amrex::Real const> const& q) noexcept
{
    using amrex::Real;

#pragma unroll
    for(int nn=0; nn< 5; ++nn){

	      Real wm1= q(i-3,j,k,nn); 
	      Real w0 = q(i-2,j,k,nn); 
	      Real w1 = q(i-1,j,k,nn); 
	      Real w2 = q(i-0,j,k,nn); 
	      Real w3 = q(i+1,j,k,nn); 
	      Real w4 = q(i+2,j,k,nn);
       

       WENOZ( w4, w3, w2, w1, w0,
	        dqR(i,j,k,nn));

       WENOZ (wm1, w0, w1, w2, w3,
		       dqL(i,j,k,nn));

    };


}

AMREX_GPU_DEVICE
inline
void
cns_wenoz_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dqL,
             amrex::Array4<amrex::Real> const& dqR,
             amrex::Array4<amrex::Real const> const& q) noexcept
{
    using amrex::Real;

#pragma unroll
    for(int nn=0; nn< 5; ++nn){

	      Real wm1= q(i,j-3,k,nn); 
	      Real w0 = q(i,j-2,k,nn); 
	      Real w1 = q(i,j-1,k,nn); 
	      Real w2 = q(i,j-0,k,nn); 
	      Real w3 = q(i,j+1,k,nn); 
	      Real w4 = q(i,j+2,k,nn);
       

       WENOZ( w4, w3, w2, w1, w0, dqR(i,j,k,nn));

       WENOZ( wm1, w0, w1, w2, w3, dqL(i,j,k,nn));

    };

}

AMREX_GPU_DEVICE
inline
void
cns_wenoz_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dqL,
             amrex::Array4<amrex::Real> const& dqR,
             amrex::Array4<amrex::Real const> const& q) noexcept
{
    using amrex::Real;

#pragma unroll
    for(int nn=0; nn< 5; ++nn){

	      Real wm1= q(i,j,k-3,nn); 
	      Real w0 = q(i,j,k-2,nn); 
	      Real w1 = q(i,j,k-1,nn); 
	      Real w2 = q(i,j,k-0,nn); 
	      Real w3 = q(i,j,k+1,nn); 
	      Real w4 = q(i,j,k+2,nn);
       

       WENOZ( w4, w3, w2, w1, w0, dqR(i,j,k,nn));

       WENOZ( wm1, w0, w1, w2, w3, dqL(i,j,k,nn));

    };

}







namespace {

AMREX_GPU_DEVICE
inline
void
riemann (const amrex::Real gamma, const amrex::Real smallp, const amrex::Real smallr,
         const amrex::Real rl, const amrex::Real ul, const amrex::Real pl,
         const amrex::Real ut1l, const amrex::Real ut2l,
         const amrex::Real rr, const amrex::Real ur, const amrex::Real pr,
         const amrex::Real ut1r, const amrex::Real ut2r,
         amrex::Real& flxrho, amrex::Real& flxu, amrex::Real& flxut,
         amrex::Real& flxutt, amrex::Real& flxe) noexcept
{
    using amrex::Real;

    constexpr Real weakwv = Real(1.e-3);
    constexpr Real small = Real(1.e-6);

    Real clsql = gamma*pl*rl;
    Real clsqr = gamma*pr*rr;
    Real wl = std::sqrt(clsql);
    Real wr = std::sqrt(clsqr);
    Real cleft = wl/rl;
    Real cright = wr/rr;
    Real ccsmall = small*(cleft+cright);

    Real pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar,smallp);
    Real pstnm1 = pstar;

    Real wlsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pl)+pstar)*rl;
    Real wrsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pr)+pstar)*rr;

    wl = std::sqrt(wlsq);
    wr = std::sqrt(wrsq);
    Real ustarp = ul - (pstar-pl)/wl;
    Real ustarm = ur + (pstar-pr)/wr;

    pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar,smallp);

    Real ustar;
    for (int iter = 0; iter < 3; ++iter)
    {
        wlsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pl)+pstar)*rl;
        wrsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pr)+pstar)*rr;

        wl = Real(1.)/std::sqrt(wlsq);
        wr = Real(1.)/std::sqrt(wrsq);

        Real ustnm1 = ustarm;
        Real ustnp1 = ustarp;

        ustarm = ur - (pr - pstar)*wr;
        ustarp = ul + (pl - pstar)*wl;

        Real dpditer = amrex::Math::abs(pstnm1-pstar);
        Real zp = amrex::Math::abs(ustarp-ustnp1);
        if (zp-weakwv*cleft < Real(0.0) ) {
            zp = dpditer*wl;
        }
        Real zm = amrex::Math::abs(ustarm-ustnm1);
        if (zm-weakwv*cright < Real(0.0) ) {
            zm = dpditer*wr;
        }

        Real zz = zp+zm;
        Real denom = dpditer/ amrex::max(zz,ccsmall);
        pstnm1 = pstar;
        pstar = pstar - denom*(ustarm-ustarp);
        pstar = amrex::max(pstar,smallp);
        ustar = Real(0.5)*(ustarm+ustarp);
    }

    Real ro, uo, po, sgnm, utrans1, utrans2;
    if (ustar > Real(0.)) {
        ro = rl;
        uo = ul;
        po = pl;
        sgnm = Real(1.);
        utrans1 = ut1l;
        utrans2 = ut2l;
    } else if (ustar < Real(0.)) {
        ro = rr;
        uo = ur;
        po = pr;
        sgnm = Real(-1.);
        utrans1 = ut1r;
        utrans2 = ut2r;
    } else {
        uo = Real(0.5)*(ur+ul);
        po = Real(0.5)*(pr+pl);
        ro = Real(2.)*(rl*rr)/(rl+rr);
        sgnm = Real(1.);
        utrans1 = Real(0.5)*(ut1l+ut1r);
        utrans2 = Real(0.5)*(ut2l+ut2r);
    }
    Real wosq = (Real(0.5)*(gamma-Real(1.))*(pstar+po)+pstar)*ro;
    Real co = std::sqrt(gamma * po / ro);
    Real wo = std::sqrt(wosq);
    Real dpjmp = pstar-po;
    Real rstar = ro/(Real(1.)-ro*dpjmp/wosq);
    Real cstar = std::sqrt(gamma * pstar / rstar);
    Real spout = co-sgnm*uo;
    Real spin = cstar - sgnm*uo;
    if(pstar >= po) {
        spin = wo/ro-sgnm*uo;
        spout = spin;
    }
    Real ss = amrex::max(spout-spin, spout+spin);
    Real frac = Real(0.5)*(Real(1.)+(spin+spout)/amrex::max(ss,ccsmall));

    Real rgdnv, ugdnv, pgdnv;
    if (spout < Real(0.)) {
        rgdnv = ro;
        ugdnv = uo;
        pgdnv = po;
    } else if(spin >= Real(0.)) {
        rgdnv = rstar;
        ugdnv = ustar;
        pgdnv = pstar;
    } else {
        rgdnv = frac*rstar + (Real(1.) - frac)* ro;
        ugdnv = frac*ustar + (Real(1.) - frac)* uo;
        pgdnv = frac*pstar + (Real(1.) - frac)* po;
    }
    
    flxrho = rgdnv*ugdnv;
    flxu = rgdnv*ugdnv*ugdnv+pgdnv;
    flxut = rgdnv*ugdnv*utrans1;
    flxutt = rgdnv*ugdnv*utrans2;
    flxe = ugdnv*(Real(0.5)*rgdnv*(ugdnv*ugdnv+utrans1*utrans1+utrans2*utrans2) + pgdnv/(gamma -Real(1.)) + pgdnv);
}
}

AMREX_GPU_DEVICE
inline
void
cns_riemann_x (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fx,
               amrex::Array4<amrex::Real const> const& dqL,
               amrex::Array4<amrex::Real const> const& dqR,
               amrex::Array4<amrex::Real const> const& q,
               Parm const& parm) noexcept
{
    using amrex::Real;

    Real rl = dqL(i,j,k,QRHO);
    rl = amrex::max(rl, parm.smallr);
    Real ul = dqL(i,j,k,QU);
    Real pl = dqL(i,j,k,QPRES);
    pl = amrex::max(pl, parm.smallp);
    Real ut1l = dqL(i,j,k,QV);
    Real ut2l = dqL(i,j,k,QW);

    Real rr = dqR(i,j,k,QRHO);
    rr = amrex::max(rr, parm.smallr);
    Real ur = dqR(i,j,k,QU);
    Real pr = dqR(i,j,k,QPRES);
    pr = amrex::max(pr, parm.smallp);
    Real ut1r = dqR(i,j,k,QV);
    Real ut2r = dqR(i,j,k,QW);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fx(i,j,k,URHO), fx(i,j,k,UMX), fx(i,j,k,UMY), fx(i,j,k,UMZ), fx(i,j,k,UEDEN));
}

AMREX_GPU_DEVICE
inline
void
cns_riemann_y (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fy,
               amrex::Array4<amrex::Real const> const& dqL,
               amrex::Array4<amrex::Real const> const& dqR,
               amrex::Array4<amrex::Real const> const& q,
               Parm const& parm) noexcept
{
    using amrex::Real;

    Real rl = dqL(i,j,k,QRHO);
    rl = amrex::max(rl, parm.smallr);
    Real ul = dqL(i,j,k,QV);
    Real pl = dqL(i,j,k,QPRES);
    pl = amrex::max(pl, parm.smallp);
    Real ut1l = dqL(i,j,k,QU);
    Real ut2l = dqL(i,j,k,QW);

    Real rr = dqR(i,j,k,QRHO);
    rr = amrex::max(rr, parm.smallr);
    Real ur = dqR(i,j,k,QV);
    Real pr = dqR(i,j,k,QPRES);
    pr = amrex::max(pr, parm.smallp);
    Real ut1r = dqR(i,j,k,QU);
    Real ut2r = dqR(i,j,k,QW);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fy(i,j,k,URHO), fy(i,j,k,UMY), fy(i,j,k,UMX), fy(i,j,k,UMZ), fy(i,j,k,UEDEN));
}

AMREX_GPU_DEVICE
inline
void
cns_riemann_z (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fz,
               amrex::Array4<amrex::Real const> const& dqL,
               amrex::Array4<amrex::Real const> const& dqR,
               amrex::Array4<amrex::Real const> const& q,
               Parm const& parm) noexcept
{
    using amrex::Real;

    Real rl = dqL(i,j,k,QRHO);
    rl = amrex::max(rl, parm.smallr);
    Real ul = dqL(i,j,k,QW);
    Real pl = dqL(i,j,k,QPRES);
    pl = amrex::max(pl, parm.smallp);
    Real ut1l = dqL(i,j,k,QU);
    Real ut2l = dqL(i,j,k,QV);

    Real rr = dqR(i,j,k,QRHO);
    rr = amrex::max(rr, parm.smallr);
    Real ur = dqR(i,j,k,QW);
    Real pr = dqR(i,j,k,QPRES);
    pr = amrex::max(pr, parm.smallp);
    Real ut1r = dqR(i,j,k,QU);
    Real ut2r = dqR(i,j,k,QV);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fz(i,j,k,URHO), fz(i,j,k,UMZ), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,UEDEN));
}

#endif
