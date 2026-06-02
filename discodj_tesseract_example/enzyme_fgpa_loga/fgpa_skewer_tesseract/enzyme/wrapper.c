// C wrapper for Enzyme AD of the FGPA skewer transform.
#include <stddef.h>

int enzyme_dup;
int enzyme_const;

extern void fgpa_skewer_transform(int* res,
                                  int* n_skewers,
                                  int* n_los,
                                  int* skewer_x,
                                  int* skewer_y,
                                  float* delta_field,
                                  float* flux,
                                  float* log_a,
                                  float* beta,
                                  float* rho_floor);

extern void __enzyme_autodiff(void*, ...);
extern void __enzyme_fwddiff(void*, ...);

void fgpa_skewer_forward(int res,
                         int n_skewers,
                         int n_los,
                         const int* skewer_x,
                         const int* skewer_y,
                         const float* delta_field,
                         float* flux,
                         float log_a,
                         float beta,
                         float rho_floor)
{
    int res_ = res;
    int n_skewers_ = n_skewers;
    int n_los_ = n_los;
    float log_a_ = log_a;
    float beta_ = beta;
    float rho_floor_ = rho_floor;
    fgpa_skewer_transform(&res_, &n_skewers_, &n_los_,
        (int*)skewer_x, (int*)skewer_y,
        (float*)delta_field, flux, &log_a_, &beta_, &rho_floor_);
}

void fgpa_skewer_vjp(int res,
                     int n_skewers,
                     int n_los,
                     const int* skewer_x,
                     const int* skewer_y,
                     const float* delta_field,
                     float* ddelta_field,
                     const float* flux,
                     float* dflux,
                     float log_a,
                     float* dlog_a,
                     float beta,
                     float* dbeta,
                     float rho_floor)
{
    int res_ = res;
    int n_skewers_ = n_skewers;
    int n_los_ = n_los;
    float log_a_ = log_a;
    float beta_ = beta;
    float rho_floor_ = rho_floor;

    __enzyme_autodiff((void*)fgpa_skewer_transform,
        enzyme_const, &res_,
        enzyme_const, &n_skewers_,
        enzyme_const, &n_los_,
        enzyme_const, (int*)skewer_x,
        enzyme_const, (int*)skewer_y,
        enzyme_dup, (float*)delta_field, ddelta_field,
        enzyme_dup, (float*)flux, dflux,
        enzyme_dup, &log_a_, dlog_a,
        enzyme_dup, &beta_, dbeta,
        enzyme_const, &rho_floor_);
}

void fgpa_skewer_jvp(int res,
                     int n_skewers,
                     int n_los,
                     const int* skewer_x,
                     const int* skewer_y,
                     const float* delta_field,
                     float* ddelta_field,
                     float* flux,
                     float* dflux,
                     float log_a,
                     float dlog_a,
                     float beta,
                     float dbeta,
                     float rho_floor)
{
    int res_ = res;
    int n_skewers_ = n_skewers;
    int n_los_ = n_los;
    float log_a_ = log_a;
    float dlog_a_ = dlog_a;
    float beta_ = beta;
    float dbeta_ = dbeta;
    float rho_floor_ = rho_floor;

    __enzyme_fwddiff((void*)fgpa_skewer_transform,
        enzyme_const, &res_,
        enzyme_const, &n_skewers_,
        enzyme_const, &n_los_,
        enzyme_const, (int*)skewer_x,
        enzyme_const, (int*)skewer_y,
        enzyme_dup, (float*)delta_field, ddelta_field,
        enzyme_dup, flux, dflux,
        enzyme_dup, &log_a_, &dlog_a_,
        enzyme_dup, &beta_, &dbeta_,
        enzyme_const, &rho_floor_);
}
