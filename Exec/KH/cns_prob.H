#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

AMREX_GPU_DEVICE
inline
void
switch_to_cons_id (const amrex::Real gamma, 
                amrex::Real &Urho,
                amrex::Real &Ue, // initialise with eps
                amrex::Real &Ux,
                amrex::Real &Uy,
                amrex::Real &Uz){

    auto const z2 = Ux * Ux + Uy * Uy + Uz * Uz;
    auto const lorentz2 = 1. + z2;
    auto const lorentz = sqrt(lorentz2);

    // Need EOS call here!!
    auto press =  Ue; //eps*(gamma-1.);

    auto eps = press/(gamma -1.);

    Urho *= lorentz;
    auto const rhohW = Urho + (press + eps) * lorentz;

    Ux *= rhohW;
    Uy *= rhohW;
    Uz *= rhohW;

//    U[TAUENERGY] = rhohW * lorentz - press -  U[RHOSTAR];
//    This should also work without double precision for small velocities
    Ue  = press*z2 + eps * lorentz2 + Urho*z2/(lorentz+1.);
};

AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;


      const Real A = 0.1;
      const Real a = 0.01;
      const Real sigma = 0.1;
      const Real Vshear = 0.5;
      const Real rho0 = 0.505;
      const Real rho1 = 0.495;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

//    constexpr Real pi = Real(3.14159265358979323846264338327950288);

    const Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    const Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    const Real x = prob_lo[0] + (i+Real(0.5))*dx[0];


    if(y <= 0.){
        state(i,j,k,URHO ) = rho0 - rho1*tanh((y+0.5)/a);
   	state(i,j,k,UEDEN) = 1.;
   	state(i,j,k,UMX)   = -Vshear*tanh((y+0.5)/a);
   	state(i,j,k,UMY)   = -A*Vshear*sin(2.*M_PI*x)*exp(-(y+0.5)*(y+0.5)/sigma/sigma);
   	state(i,j,k,UMZ)   = 0.;
    } 
    else{
   	state(i,j,k,URHO )  = rho0 + rho1*tanh((y-0.5)/a);
   	state(i,j,k,UEDEN)  = 1.;
   	state(i,j,k,UMX)    = Vshear*tanh((y-0.5)/a);
   	state(i,j,k,UMY)    = A*Vshear*sin(2.*M_PI*x)*exp(-(y-0.5)*(y-0.5)/sigma/sigma);
   	state(i,j,k,UMZ)    = 0.;
    }

      auto const v2 =      state(i,j,k,UMX)*state(i,j,k,UMX)
			  +state(i,j,k,UMY)*state(i,j,k,UMY)
			  +state(i,j,k,UMZ)*state(i,j,k,UMZ);

      auto const lorentz = sqrt(1./(1.-v2));

      state(i,j,k,UMX) *= lorentz;
      state(i,j,k,UMY) *= lorentz;
      state(i,j,k,UMZ) *= lorentz;
//      state(i,j,k,UEDEN) /= parm.eos_gamma - 1.;

      switch_to_cons_id(parm.eos_gamma, state(i,j,k,URHO ), state(i,j,k,UEDEN), state(i,j,k,UMX), state(i,j,k,UMY), state(i,j,k,UMZ));


      state(i,j,k,UEINT) = state(i,j,k,UEDEN);
}

#endif
