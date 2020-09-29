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
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    constexpr Real pi = Real(3.14159265358979323846264338327950288);

    const Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    const Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    const Real x = prob_lo[0] + (i+Real(0.5))*dx[0];

    auto const x = U.grid.get_coords(0,i);
    auto const y = U.grid.get_coords(1,j);

    if(y <= 0.){
        state(i,j,k,URHO ) = rho0 - rho1*amrex::Math::tanh((y+0.5)/a);
   	state(i,j,k,UEDEN) = 1.;
   	state(i,j,k,UMX)   = -Vshear*amrex::Math::tanh((y+0.5)/a);
   	state(i,j,k,UMY)   = -A*Vshear*amrex::Math::sin(2.*M_PI*x)*amrex::Math::exp(-(y+0.5)*(y+0.5)/sigma/sigma);
   	state(i,j,k,UMZ)   = 0.;
    } 
    else{
   	state(i,j,k,URHO )  = rho0 + rho1*amrex::Math::tanh((y-0.5)/a);
   	state(i,j,k,UEDEN)  = 1.;
   	state(i,j,k,UMX)    = Vshear*amrex::Math::tanh((y-0.5)/a);
   	state(i,j,k,UMY)    = A*Vshear*amrex::Math::sin(2.*M_PI*x)*amrex::Math::exp(-(y-0.5)*(y-0.5)/sigma/sigma);
   	state(i,j,k,UMZ)    = 0.;
    }

      auto const v2 =      state(i,j,k,UMX)*state(i,j,k,UMX)
			  +state(i,j,k,UMY)*state(i,j,k,UMY)
			  +state(i,j,k,UMZ)*state(i,j,k,UMZ);

      auto const lorentz = amrex::Math::sqrt(1./(1.-v2));

      state(i,j,k,UMX) *= lorentz;
      state(i,j,k,UMY) *= lorentz;
      state(i,j,k,UMZ) *= lorentz;
      state(i,j,k,UEDEN) /= parm.eos_gamma - 1.;

      switch_to_cons(parm.eos_gamma, state(i,j,k,URHO ), state(i,j,k,UEDEN), state(i,j,k,UMX), state(i,j,k,UMY), state(i,j,k,UMZ));


      state(i,j,k,UEINT) = state(i,j,k,UEDEN)
}

#endif
