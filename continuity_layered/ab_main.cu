// A/B harness: original 11-kernel launcher vs the optimized one.
// Same device inputs; checks max|diff| on fh and times both with CUDA events.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "gpu_rt.h"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); std::exit(1);} }while(0)

extern "C" void continuity_layered_cuda_launch(
    const double*,const double*,const double*,const double*,const double*,const double*,const double*,
    double*,double*,double*,double*,double*,double*,double*,
    int,int,int,int,int,int,int,double,int);
extern "C" void continuity_opt_launch(
    const double*,const double*,const double*,const double*,const double*,const double*,const double*,
    double*,double*,double*,int,int,int,int,int,int,int,double);

static int iarg(int c,char**v,int k,int d){ return k<c?atoi(v[k]):d; }

int main(int argc,char**argv){
    const int NGHOST=3; const double DX=10.0,DY=10.0,H_MIN=1e-6; const int DO_POS=0;
    int nxp=iarg(argc,argv,1,473), nyp=iarg(argc,argv,2,297), nz=iarg(argc,argv,3,30);
    int reps=iarg(argc,argv,4,300);
    int nx=nxp+2*NGHOST, ny=nyp+2*NGHOST;
    size_t nT=(size_t)nx*ny*nz, nU=(size_t)(nx+1)*ny*nz, nV=(size_t)nx*(ny+1)*nz;
    size_t mT=(size_t)nx*ny, mU=(size_t)(nx+1)*ny, mV=(size_t)nx*(ny+1);

    std::vector<double> h(nT),u(nU),v(nV),wet(mT),iar(mT),dy(mU),dx(mV);
    for(int k=1;k<=nz;++k)for(int j=1;j<=ny;++j)for(int i=1;i<=nx;++i){
        double gi=(double)(i-nx/2)/(double)std::max(nx/8,1), gj=(double)(j-ny/2)/(double)std::max(ny/8,1);
        h[(size_t)(i-1)+(size_t)nx*((size_t)(j-1)+(size_t)ny*(k-1))]=
            20.0+(double)k*0.5+5.0*std::exp(-(gi*gi+gj*gj))+0.4*std::sin(0.01*i)*std::cos(0.013*j);
    }
    for(int j=1;j<=ny;++j)for(int i=1;i<=nx;++i){
        wet[(size_t)(i-1)+(size_t)nx*(j-1)]=1.0; iar[(size_t)(i-1)+(size_t)nx*(j-1)]=1.0/(DX*DY);}
    for(int k=1;k<=nz;++k)for(int j=1;j<=ny;++j)for(int i=1;i<=nx+1;++i)
        u[(size_t)(i-1)+(size_t)(nx+1)*((size_t)(j-1)+(size_t)ny*(k-1))]=0.5*std::sin(0.02*i)*std::cos(0.03*k);
    for(int k=1;k<=nz;++k)for(int j=1;j<=ny+1;++j)for(int i=1;i<=nx;++i)
        v[(size_t)(i-1)+(size_t)nx*((size_t)(j-1)+(size_t)(ny+1)*(k-1))]=0.3*std::cos(0.017*j);
    for(int j=1;j<=ny;++j)for(int i=1;i<=nx+1;++i) dy[(size_t)(i-1)+(size_t)(nx+1)*(j-1)]=DY;
    for(int j=1;j<=ny+1;++j)for(int i=1;i<=nx;++i) dx[(size_t)(i-1)+(size_t)nx*(j-1)]=DX;

    double *d_h,*d_u,*d_v,*d_wet,*d_dy,*d_dx,*d_iar;
    double *d_hflx,*d_hfrx,*d_hfly,*d_hfry,*d_mfx,*d_mfy,*d_fh;   // original scratch+out
    double *d_mfx2,*d_mfy2,*d_fh2;                                // opt scratch+out
    CUDA_OK(cudaMalloc(&d_h,nT*8));CUDA_OK(cudaMalloc(&d_u,nU*8));CUDA_OK(cudaMalloc(&d_v,nV*8));
    CUDA_OK(cudaMalloc(&d_wet,mT*8));CUDA_OK(cudaMalloc(&d_iar,mT*8));
    CUDA_OK(cudaMalloc(&d_dy,mU*8));CUDA_OK(cudaMalloc(&d_dx,mV*8));
    CUDA_OK(cudaMalloc(&d_hflx,nU*8));CUDA_OK(cudaMalloc(&d_hfrx,nU*8));
    CUDA_OK(cudaMalloc(&d_hfly,nV*8));CUDA_OK(cudaMalloc(&d_hfry,nV*8));
    CUDA_OK(cudaMalloc(&d_mfx,nU*8));CUDA_OK(cudaMalloc(&d_mfy,nV*8));CUDA_OK(cudaMalloc(&d_fh,nT*8));
    CUDA_OK(cudaMalloc(&d_mfx2,nU*8));CUDA_OK(cudaMalloc(&d_mfy2,nV*8));CUDA_OK(cudaMalloc(&d_fh2,nT*8));
    CUDA_OK(cudaMemcpy(d_h,h.data(),nT*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_u,u.data(),nU*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_v,v.data(),nV*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wet,wet.data(),mT*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_iar,iar.data(),mT*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dy,dy.data(),mU*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dx,dx.data(),mV*8,cudaMemcpyHostToDevice));
    for(double*p:{d_hflx,d_hfrx,d_hfly,d_hfry,d_mfx,d_mfy,d_fh,d_mfx2,d_mfy2,d_fh2}){}
    CUDA_OK(cudaMemset(d_hflx,0,nU*8));CUDA_OK(cudaMemset(d_hfrx,0,nU*8));
    CUDA_OK(cudaMemset(d_hfly,0,nV*8));CUDA_OK(cudaMemset(d_hfry,0,nV*8));
    CUDA_OK(cudaMemset(d_mfx,0,nU*8));CUDA_OK(cudaMemset(d_mfy,0,nV*8));CUDA_OK(cudaMemset(d_fh,0,nT*8));
    CUDA_OK(cudaMemset(d_mfx2,0,nU*8));CUDA_OK(cudaMemset(d_mfy2,0,nV*8));CUDA_OK(cudaMemset(d_fh2,0,nT*8));

#define RUN_ORIG(S) continuity_layered_cuda_launch(d_h,d_u,d_v,d_wet,d_dy,d_dx,d_iar,\
        d_hflx,d_hfrx,d_hfly,d_hfry,d_mfx,d_mfy,d_fh,nx,ny,nz,NGHOST,nxp,nyp,DO_POS,H_MIN,(S))
#define RUN_OPT() continuity_opt_launch(d_h,d_u,d_v,d_wet,d_dy,d_dx,d_iar,\
        d_mfx2,d_mfy2,d_fh2,nx,ny,nz,NGHOST,nxp,nyp,DO_POS,H_MIN)

    // correctness: one call each, compare fh
    RUN_ORIG(1); RUN_OPT(); CUDA_OK(cudaDeviceSynchronize());
    std::vector<double> f0(nT),f1(nT);
    CUDA_OK(cudaMemcpy(f0.data(),d_fh,nT*8,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(f1.data(),d_fh2,nT*8,cudaMemcpyDeviceToHost));
    double dmax=0,rmax=0; size_t nbad=0;
    for(size_t i=0;i<nT;++i){ double d=std::fabs(f0[i]-f1[i]); dmax=std::max(dmax,d);
        double s=std::max(std::fabs(f0[i]),std::fabs(f1[i])); if(s>1e-30){double r=d/s; rmax=std::max(rmax,r); if(r>1e-12)++nbad;} }

    // timing: async pipeline, event pair around `reps` launches, min over windows
    auto bench=[&](int which)->double{
        cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b));
        for(int w=0;w<5;++w){ if(which==0) RUN_ORIG(0); else RUN_OPT(); } CUDA_OK(cudaDeviceSynchronize());
        double best=1e30;
        for(int win=0;win<6;++win){
            CUDA_OK(cudaEventRecord(a));
            for(int r=0;r<reps;++r){ if(which==0) RUN_ORIG(0); else RUN_OPT(); }
            CUDA_OK(cudaEventRecord(b)); CUDA_OK(cudaEventSynchronize(b));
            float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,a,b)); best=std::min(best,(double)ms/reps);
        }
        cudaEventDestroy(a); cudaEventDestroy(b); return best;
    };
    double t_orig=bench(0), t_opt=bench(1);

    std::printf("grid %dx%dx%d (%zu cells)\n", nxp,nyp,nz,nT);
    std::printf("  correctness fh: max|diff|=%.3e  max rel=%.3e  (%zu cells > 1e-12)%s\n",
                dmax,rmax,nbad, (rmax<1e-12?"  OK":"  *** DIFF ***"));
    std::printf("  original 11-kernel : %.4f ms/rep\n", t_orig);
    std::printf("  optimized          : %.4f ms/rep   -> %.3fx\n", t_opt, t_orig/t_opt);
    return 0;
}
