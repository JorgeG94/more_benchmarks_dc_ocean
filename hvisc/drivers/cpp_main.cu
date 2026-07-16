// NATIVE C++/CUDA driver for the ocean horizontal-viscosity Smagorinsky
// closure: cudaMalloc + main(), no Fortran, no OpenACC. Recomputes the driver
// state in C++, runs the same hvisc_kernel.cu launchers, and (with DC_REF /
// --ref) cross-checks du_visc against the do-concurrent reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "gpu_rt.h"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); std::exit(1);} }while(0)

extern "C" void hvisc_compute_smag_cuda_launch(const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,const double*,const double*,const double*,
    double*,double*,double,double,double,double,int,int,int,int);
extern "C" void hvisc_compute_face_cuda_launch(const double*,const double*,const double*,const double*,
    double*,double*,const double*,const double*,const double*,const double*,const double*,const double*,
    int,int,int,int);

static int iarg(int c,char**v,int k,int d){ return k<c && v[k][0]!='-' ? atoi(v[k]) : d; }
static double wall(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

int main(int argc,char**argv){
    const int NGHOST=3; const double DX=1000.0,DY=1000.0,C_SMAG=0.2,AH_BG=1.0,AH_MAX=1e5,NS=0.0;
    int nxp=iarg(argc,argv,1,473), nyp=iarg(argc,argv,2,297), nz=iarg(argc,argv,3,30);
    int reps=iarg(argc,argv,4,200), warm=iarg(argc,argv,5,10), sync=iarg(argc,argv,6,1);
    int nx=nxp+2*NGHOST, ny=nyp+2*NGHOST;
    std::string ref;
    for(int a=1;a<argc;++a){ std::string s=argv[a]; if(s=="--ref"&&a+1<argc) ref=argv[a+1]; }
    if(const char*e=getenv("DC_REF")) if(ref.empty()) ref=e;

    size_t nU=(size_t)(nx+1)*ny*nz, nV=(size_t)nx*(ny+1)*nz;
    size_t mT=(size_t)nx*ny, mBu=(size_t)(nx+1)*(ny+1), mCu=(size_t)(nx+1)*ny, mCv=(size_t)nx*(ny+1);
    std::vector<double> u(nU),v(nV),dxT(mT,DX),dyT(mT,DY),idxT(mT,1.0/DX),idyT(mT,1.0/DX);
    std::vector<double> dy_dxBu(mBu,1.0),dx_dyBu(mBu,1.0),idyCv(mCv,1.0/DX),idxCu(mCu,1.0/DX),wet_q(mBu,1.0);
    std::vector<double> dy_dxT(mT,1.0),iareaCu(mCu,1.0/(DX*DY)),dx_dyT(mT,1.0),iareaCv(mCv,1.0/(DX*DY));
    for(int k=1;k<=nz;++k)for(int j=1;j<=ny;++j)for(int i=1;i<=nx+1;++i)
        u[(size_t)(i-1)+(size_t)(nx+1)*((size_t)(j-1)+(size_t)ny*(k-1))]=(1.0+0.1*k)*std::sin(0.02*i)*std::cos(0.03*j);
    for(int k=1;k<=nz;++k)for(int j=1;j<=ny+1;++j)for(int i=1;i<=nx;++i)
        v[(size_t)(i-1)+(size_t)nx*((size_t)(j-1)+(size_t)(ny+1)*(k-1))]=(1.0+0.1*k)*std::cos(0.017*i)*std::sin(0.023*j);

    // ADOPT the DC reference's inputs (du_ref, then u, v) so the cross-check
    // measures the kernel, not libm sin/cos differences. See LOGBOOK note.
    std::vector<double> du_ref; bool have_ref=false;
    if(!ref.empty()){
        FILE*f=std::fopen(ref.c_str(),"rb");
        if(!f) std::printf("  cross-check : ref %s not found\n", ref.c_str());
        else { int rnx,rny,rnz;
            if(std::fread(&rnx,4,1,f)&&std::fread(&rny,4,1,f)&&std::fread(&rnz,4,1,f)&&rnx==nx&&rny==ny&&rnz==nz){
                du_ref.resize(nU);
                if(std::fread(du_ref.data(),8,nU,f)==nU && std::fread(u.data(),8,nU,f)==nU && std::fread(v.data(),8,nV,f)==nV)
                    have_ref=true;
            } else std::printf("  cross-check : ref shape mismatch\n");
            std::fclose(f); }
    }

    double *d_u,*d_v,*d_dxT,*d_dyT,*d_idxT,*d_idyT,*d_dydxBu,*d_dxdyBu,*d_idyCv,*d_idxCu,*d_wetq;
    double *d_dydxT,*d_iareaCu,*d_dxdyT,*d_iareaCv,*d_ahx,*d_ahy,*d_du,*d_dv;
    auto MK=[&](double**p,size_t n){ CUDA_OK(cudaMalloc(p,n*8)); };
    MK(&d_u,nU);MK(&d_v,nV);MK(&d_dxT,mT);MK(&d_dyT,mT);MK(&d_idxT,mT);MK(&d_idyT,mT);
    MK(&d_dydxBu,mBu);MK(&d_dxdyBu,mBu);MK(&d_idyCv,mCv);MK(&d_idxCu,mCu);MK(&d_wetq,mBu);
    MK(&d_dydxT,mT);MK(&d_iareaCu,mCu);MK(&d_dxdyT,mT);MK(&d_iareaCv,mCv);
    MK(&d_ahx,nU);MK(&d_ahy,nV);MK(&d_du,nU);MK(&d_dv,nV);
    auto CP=[&](double*d,std::vector<double>&h){ CUDA_OK(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice)); };
    CP(d_u,u);CP(d_v,v);CP(d_dxT,dxT);CP(d_dyT,dyT);CP(d_idxT,idxT);CP(d_idyT,idyT);
    CP(d_dydxBu,dy_dxBu);CP(d_dxdyBu,dx_dyBu);CP(d_idyCv,idyCv);CP(d_idxCu,idxCu);CP(d_wetq,wet_q);
    CP(d_dydxT,dy_dxT);CP(d_iareaCu,iareaCu);CP(d_dxdyT,dx_dyT);CP(d_iareaCv,iareaCv);
    CUDA_OK(cudaMemset(d_ahx,0,nU*8));CUDA_OK(cudaMemset(d_ahy,0,nV*8));
    CUDA_OK(cudaMemset(d_du,0,nU*8));CUDA_OK(cudaMemset(d_dv,0,nV*8));

    auto closure=[&](int s){
        hvisc_compute_smag_cuda_launch(d_u,d_v,d_dxT,d_dyT,d_idxT,d_idyT,d_dydxBu,d_dxdyBu,d_idyCv,d_idxCu,d_wetq,
                                       d_ahx,d_ahy,C_SMAG,AH_BG,AH_MAX,NS,nx,ny,nz,s);
        hvisc_compute_face_cuda_launch(d_u,d_v,d_ahx,d_ahy,d_du,d_dv,d_dydxT,d_dxdyBu,d_iareaCu,d_dxdyT,d_dydxBu,d_iareaCv,nx,ny,nz,s);
    };
    for(int r=0;r<warm;++r) closure(1); CUDA_OK(cudaDeviceSynchronize());
    double best=1e30;
    for(int w=0;w<5;++w){ double t=wall(); for(int r=0;r<reps;++r) closure(sync); CUDA_OK(cudaDeviceSynchronize());
        best=std::min(best,(wall()-t)*1000.0/reps); }

    std::vector<double> du(nU); CUDA_OK(cudaMemcpy(du.data(),d_du,nU*8,cudaMemcpyDeviceToHost));
    double smin=du[0],smax=du[0],ssum=0; for(double x:du){smin=std::min(smin,x);smax=std::max(smax,x);ssum+=x;}
    std::printf("======================================================================\n");
    std::printf("  driver: NATIVE C++/CUDA (cudaMalloc + main), no Fortran, no OpenACC\n");
    std::printf("  domain: %dx%dx%d  Smag closure : %.4f ms/rep\n", nxp,nyp,nz, best);
    std::printf("  sanity: sum du %.6e  (min %.3e max %.3e)\n", ssum, smin, smax);

    if(have_ref){
        double dmax=0,rmax=0,fmax=0; for(size_t i=0;i<nU;++i){ double d=std::fabs(du[i]-du_ref[i]); dmax=std::max(dmax,d);
            fmax=std::max(fmax,std::fabs(du_ref[i])); double s=std::max(std::fabs(du[i]),std::fabs(du_ref[i])); if(s>1e-30) rmax=std::max(rmax,d/s); }
        double frel=fmax>0?dmax/fmax:0;   // field-relative: robust to near-zero cells
        std::printf("  cross-check vs DC ref: max|diff| %.5e  field-rel %.5e  (ptwise-rel %.2e)  %s\n",
                    dmax,frel,rmax, frel<1e-12?"OK":"*** DIFF ***");
    }
    std::printf("======================================================================\n");
    return 0;
}
