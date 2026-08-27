/**
 * @file rri_opencl.c
 * @brief OpenCL host-side backend implementation: device selection,
 * program build (kernels.h + cl/rri_kernels.cl concatenated at runtime),
 * and the "upload, run, download" dispatch functions declared in
 * rri/opencl.h. See that header's file-level comment for the overall
 * design (correctness-first, not yet a persistent-buffer/throughput
 * design) and cl/rri_kernels.cl's file-level comment for the kernel
 * source itself and the per-direction flattened-buffer layout used for
 * the slope/groundwater kernels.
 *
 * Deliberately uses the OpenCL 1.x API surface only (`clCreateCommandQueue`,
 * not the OpenCL 2.0 `clCreateCommandQueueWithProperties`) so the exact
 * same source builds and runs unchanged against PoCL (OpenCL 3.0,
 * local validation) and the Mesa Clover platform on the AMD Polaris10
 * GPU this was validated against (OpenCL 1.1 -- see README.md's OpenCL
 * section for that validation run).
 */
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>

#include "rri/opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef RRI_KERNELS_H_PATH
#error "RRI_KERNELS_H_PATH must be defined by the build (path to include/rri/kernels.h)"
#endif
#ifndef RRI_CL_SRC_PATH
#error "RRI_CL_SRC_PATH must be defined by the build (path to cl/rri_kernels.cl)"
#endif

struct rri_cl_backend {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_qr, k_qs, k_qg, k_infilt;
    char device_name[256];
};

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "rri_opencl: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

static int device_has_fp64(cl_device_id dev)
{
    char ext[4096] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL);
    return strstr(ext, "cl_khr_fp64") != NULL;
}

/* Pick the first fp64-capable device, preferring a GPU when prefer_gpu is
 * set and one is actually available -- otherwise fall through to
 * whatever fp64-capable device exists (CPU, e.g. PoCL). Scans every
 * platform: multiple OpenCL platforms can be installed side by side
 * (e.g. this project's remote GPU host has both Clover and a rusticl
 * platform that reports 0 devices -- skipped naturally here since it
 * has nothing to enumerate). */
static int select_device(cl_platform_id *out_plat, cl_device_id *out_dev, int prefer_gpu)
{
    cl_uint nplat = 0;
    clGetPlatformIDs(0, NULL, &nplat);
    if (nplat == 0) { fprintf(stderr, "rri_opencl: no OpenCL platforms found\n"); return -1; }
    cl_platform_id *plats = malloc(sizeof(cl_platform_id) * nplat);
    clGetPlatformIDs(nplat, plats, NULL);

    cl_platform_id fallback_plat = 0; cl_device_id fallback_dev = 0;
    cl_platform_id gpu_plat = 0; cl_device_id gpu_dev = 0;

    for (cl_uint p = 0; p < nplat; p++) {
        cl_uint ndev = 0;
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
        if (ndev == 0) continue;
        cl_device_id *devs = malloc(sizeof(cl_device_id) * ndev);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs, NULL);
        for (cl_uint d = 0; d < ndev; d++) {
            if (!device_has_fp64(devs[d])) continue;
            cl_device_type t;
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
            if (t == CL_DEVICE_TYPE_GPU && !gpu_dev) { gpu_plat = plats[p]; gpu_dev = devs[d]; }
            if (!fallback_dev) { fallback_plat = plats[p]; fallback_dev = devs[d]; }
        }
        free(devs);
    }
    free(plats);

    if (prefer_gpu && gpu_dev) { *out_plat = gpu_plat; *out_dev = gpu_dev; return 0; }
    if (fallback_dev) { *out_plat = fallback_plat; *out_dev = fallback_dev; return 0; }
    fprintf(stderr, "rri_opencl: no cl_khr_fp64-capable OpenCL device found\n");
    return -1;
}

rri_cl_backend *rri_cl_backend_init(int prefer_gpu)
{
    rri_cl_backend *b = calloc(1, sizeof(*b));
    if (select_device(&b->platform, &b->device, prefer_gpu) != 0) { free(b); return NULL; }

    char vendor[128] = {0}, name[128] = {0}, ver[128] = {0};
    clGetDeviceInfo(b->device, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, NULL);
    clGetDeviceInfo(b->device, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
    clGetDeviceInfo(b->device, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, NULL);
    snprintf(b->device_name, sizeof(b->device_name), "%s / %s / %s", vendor, name, ver);

    cl_int err;
    b->context = clCreateContext(NULL, 1, &b->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateContext failed (%d)\n", err); free(b); return NULL; }
    b->queue = clCreateCommandQueue(b->context, b->device, 0, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateCommandQueue failed (%d)\n", err); clReleaseContext(b->context); free(b); return NULL; }

    char *kernels_h = read_file(RRI_KERNELS_H_PATH);
    char *cl_src = read_file(RRI_CL_SRC_PATH);
    if (!kernels_h || !cl_src) { free(kernels_h); free(cl_src); rri_cl_backend_free(b); return NULL; }

    /* Concatenated at the OpenCL API level (three separate source
     * strings, joined by the driver's own compiler) rather than by
     * manual string concatenation in this file -- kernels.h's content
     * is handed to clCreateProgramWithSource completely unmodified,
     * per that file's hard requirement. */
    const char *pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    const char *srcs[3] = { pragma, kernels_h, cl_src };
    b->program = clCreateProgramWithSource(b->context, 3, srcs, NULL, &err);
    free(kernels_h); free(cl_src);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateProgramWithSource failed (%d)\n", err); rri_cl_backend_free(b); return NULL; }

    err = clBuildProgram(b->program, 1, &b->device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(b->program, b->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        char *log = malloc(logsz + 1);
        clGetProgramBuildInfo(b->program, b->device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL);
        log[logsz] = '\0';
        fprintf(stderr, "rri_opencl: kernel build failed on %s:\n%s\n", b->device_name, log);
        free(log);
        rri_cl_backend_free(b);
        return NULL;
    }

    cl_int e1, e2, e3, e4;
    b->k_qr = clCreateKernel(b->program, "rri_cl_k_qr_calc", &e1);
    b->k_qs = clCreateKernel(b->program, "rri_cl_k_qs_calc", &e2);
    b->k_qg = clCreateKernel(b->program, "rri_cl_k_qg_calc", &e3);
    b->k_infilt = clCreateKernel(b->program, "rri_cl_k_infilt", &e4);
    if (e1 != CL_SUCCESS || e2 != CL_SUCCESS || e3 != CL_SUCCESS || e4 != CL_SUCCESS) {
        fprintf(stderr, "rri_opencl: clCreateKernel failed (%d,%d,%d,%d)\n", e1, e2, e3, e4);
        rri_cl_backend_free(b);
        return NULL;
    }
    return b;
}

void rri_cl_backend_free(rri_cl_backend *b)
{
    if (!b) return;
    if (b->k_qr) clReleaseKernel(b->k_qr);
    if (b->k_qs) clReleaseKernel(b->k_qs);
    if (b->k_qg) clReleaseKernel(b->k_qg);
    if (b->k_infilt) clReleaseKernel(b->k_infilt);
    if (b->program) clReleaseProgram(b->program);
    if (b->queue) clReleaseCommandQueue(b->queue);
    if (b->context) clReleaseContext(b->context);
    free(b);
}

const char *rri_cl_backend_device_name(const rri_cl_backend *b) { return b->device_name; }

/* ---- small helpers for the "upload, run, download, release" pattern
 * described in opencl.h's file-level comment -- each rri_cl_* dispatch
 * function below creates one buffer per kernel argument with
 * CL_MEM_COPY_HOST_PTR (uploads synchronously at creation time) and
 * releases them all again before returning, rather than keeping any
 * device-resident state across calls. `err` is intentionally unchecked
 * here (a failed clCreateBuffer surfaces as a NULL cl_mem, which the
 * subsequent clSetKernelArg/clEnqueueNDRangeKernel call will itself
 * report as CL_INVALID_MEM_OBJECT) -- fine for this correctness-focused
 * milestone; a throughput-oriented rewrite should check every OpenCL
 * return code properly. ------------------------------------------- */

static cl_mem buf_ro(rri_cl_backend *b, const void *data, size_t bytes)
{
    cl_int err;
    cl_mem m = clCreateBuffer(b->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void *)data, &err);
    return m;
}
/* Read-write: used only by rri_cl_infilt, which both reads AND updates
 * hs_idx/gampt_ff_idx in place (matching rri_infilt's CPU signature). */
static cl_mem buf_rw(rri_cl_backend *b, void *data, size_t bytes)
{
    cl_int err;
    cl_mem m = clCreateBuffer(b->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data, &err);
    return m;
}
/* Write-only, no host data to upload -- the kernel's output buffer. */
static cl_mem buf_out(rri_cl_backend *b, size_t bytes)
{
    cl_int err;
    cl_mem m = clCreateBuffer(b->context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    return m;
}

void rri_cl_qr_calc(rri_cl_backend *b, const rri_riv_cellset *rc, const double *hr_idx,
                     double ns_river, double *qr_idx)
{
    size_t n = (size_t)rc->count;
    cl_mem m_domain = buf_ro(b, rc->domain, n * sizeof(int));
    cl_mem m_zb = buf_ro(b, rc->zb, n * sizeof(double));
    cl_mem m_dis = buf_ro(b, rc->dis, n * sizeof(double));
    cl_mem m_down = buf_ro(b, rc->down, n * sizeof(int));
    cl_mem m_width = buf_ro(b, rc->width, n * sizeof(double));
    cl_mem m_hr = buf_ro(b, hr_idx, n * sizeof(double));
    cl_mem m_qr = buf_out(b, n * sizeof(double));

    int i = 0;
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_domain);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_zb);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_dis);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_down);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_width);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_hr);
    clSetKernelArg(b->k_qr, i++, sizeof(double), &ns_river);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &m_qr);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qr, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, m_qr, CL_TRUE, 0, n * sizeof(double), qr_idx, 0, NULL, NULL);

    clReleaseMemObject(m_domain); clReleaseMemObject(m_zb); clReleaseMemObject(m_dis);
    clReleaseMemObject(m_down); clReleaseMemObject(m_width); clReleaseMemObject(m_hr); clReleaseMemObject(m_qr);
}

/* Pack rri_slo_cellset's RRI_LMAX8 array-of-pointers fields into one
 * flat [l * count + k] host buffer for upload -- see cl/rri_kernels.cl's
 * file-level comment for why this repacking exists. */
static double *pack_lmax8_double(double *const src[RRI_LMAX8], int count)
{
    double *flat = malloc(sizeof(double) * 4 * (size_t)count);
    for (int l = 0; l < 4; l++) memcpy(flat + (size_t)l * count, src[l], sizeof(double) * (size_t)count);
    return flat;
}
static int *pack_lmax8_int(int *const src[RRI_LMAX8], int count)
{
    int *flat = malloc(sizeof(int) * 4 * (size_t)count);
    for (int l = 0; l < 4; l++) memcpy(flat + (size_t)l * count, src[l], sizeof(int) * (size_t)count);
    return flat;
}
static void unpack_lmax8_double(const double *flat, double *dst[RRI_LMAX8], int count)
{
    for (int l = 0; l < 4; l++) memcpy(dst[l], flat + (size_t)l * count, sizeof(double) * (size_t)count);
}

void rri_cl_qs_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                     double area, double *qs_idx[RRI_LMAX8])
{
    size_t n = (size_t)sc->count;
    int *down_flat = pack_lmax8_int(sc->down, sc->count);
    double *dis_flat = pack_lmax8_double(sc->dis, sc->count);
    double *len_flat = pack_lmax8_double(sc->len, sc->count);

    cl_mem m_zb = buf_ro(b, sc->zb, n * sizeof(double));
    cl_mem m_ns = buf_ro(b, sc->ns_slope, n * sizeof(double));
    cl_mem m_ka = buf_ro(b, sc->ka, n * sizeof(double));
    cl_mem m_da = buf_ro(b, sc->da, n * sizeof(double));
    cl_mem m_dm = buf_ro(b, sc->dm, n * sizeof(double));
    cl_mem m_beta = buf_ro(b, sc->beta, n * sizeof(double));
    cl_mem m_soildepth = buf_ro(b, sc->soildepth, n * sizeof(double));
    cl_mem m_gammaa = buf_ro(b, sc->gammaa, n * sizeof(double));
    cl_mem m_dif = buf_ro(b, sc->dif, n * sizeof(int));
    cl_mem m_down = buf_ro(b, down_flat, 4 * n * sizeof(int));
    cl_mem m_dis = buf_ro(b, dis_flat, 4 * n * sizeof(double));
    cl_mem m_len = buf_ro(b, len_flat, 4 * n * sizeof(double));
    cl_mem m_down1d = buf_ro(b, sc->down_1d, n * sizeof(int));
    cl_mem m_dis1d = buf_ro(b, sc->dis_1d, n * sizeof(double));
    cl_mem m_len1d = buf_ro(b, sc->len_1d, n * sizeof(double));
    cl_mem m_hs = buf_ro(b, hs_idx, n * sizeof(double));
    cl_mem m_qs = buf_out(b, 4 * n * sizeof(double));

    int lmax = sc->lmax, count = sc->count;
    int i = 0;
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_zb);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_ns);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_ka);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_da);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_dm);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_beta);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_soildepth);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_gammaa);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_dif);
    clSetKernelArg(b->k_qs, i++, sizeof(int), &lmax);
    clSetKernelArg(b->k_qs, i++, sizeof(int), &count);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_down);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_dis);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_len);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_down1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_dis1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_len1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_hs);
    clSetKernelArg(b->k_qs, i++, sizeof(double), &area);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &m_qs);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qs, 1, NULL, &gws, NULL, 0, NULL, NULL);
    double *qs_flat = malloc(4 * n * sizeof(double));
    clEnqueueReadBuffer(b->queue, m_qs, CL_TRUE, 0, 4 * n * sizeof(double), qs_flat, 0, NULL, NULL);
    unpack_lmax8_double(qs_flat, qs_idx, sc->count);
    free(qs_flat);

    clReleaseMemObject(m_zb); clReleaseMemObject(m_ns); clReleaseMemObject(m_ka); clReleaseMemObject(m_da);
    clReleaseMemObject(m_dm); clReleaseMemObject(m_beta); clReleaseMemObject(m_soildepth); clReleaseMemObject(m_gammaa);
    clReleaseMemObject(m_dif); clReleaseMemObject(m_down); clReleaseMemObject(m_dis); clReleaseMemObject(m_len);
    clReleaseMemObject(m_down1d); clReleaseMemObject(m_dis1d); clReleaseMemObject(m_len1d);
    clReleaseMemObject(m_hs); clReleaseMemObject(m_qs);
    free(down_flat); free(dis_flat); free(len_flat);
}

void rri_cl_qg_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                     double area, double *qg_idx[RRI_LMAX8])
{
    size_t n = (size_t)sc->count;
    int *down_flat = pack_lmax8_int(sc->down, sc->count);
    double *dis_flat = pack_lmax8_double(sc->dis, sc->count);
    double *len_flat = pack_lmax8_double(sc->len, sc->count);

    cl_mem m_zb = buf_ro(b, sc->zb, n * sizeof(double));
    cl_mem m_gammag = buf_ro(b, sc->gammag, n * sizeof(double));
    cl_mem m_kg0 = buf_ro(b, sc->kg0, n * sizeof(double));
    cl_mem m_fpg = buf_ro(b, sc->fpg, n * sizeof(double));
    cl_mem m_ksg = buf_ro(b, sc->ksg, n * sizeof(double));
    cl_mem m_dif = buf_ro(b, sc->dif, n * sizeof(int));
    cl_mem m_down = buf_ro(b, down_flat, 4 * n * sizeof(int));
    cl_mem m_dis = buf_ro(b, dis_flat, 4 * n * sizeof(double));
    cl_mem m_len = buf_ro(b, len_flat, 4 * n * sizeof(double));
    cl_mem m_down1d = buf_ro(b, sc->down_1d, n * sizeof(int));
    cl_mem m_dis1d = buf_ro(b, sc->dis_1d, n * sizeof(double));
    cl_mem m_len1d = buf_ro(b, sc->len_1d, n * sizeof(double));
    cl_mem m_hg = buf_ro(b, hg_idx, n * sizeof(double));
    cl_mem m_qg = buf_out(b, 4 * n * sizeof(double));

    int lmax = sc->lmax, count = sc->count;
    int i = 0;
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_zb);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_gammag);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_kg0);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_fpg);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_ksg);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_dif);
    clSetKernelArg(b->k_qg, i++, sizeof(int), &lmax);
    clSetKernelArg(b->k_qg, i++, sizeof(int), &count);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_down);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_dis);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_len);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_down1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_dis1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_len1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_hg);
    clSetKernelArg(b->k_qg, i++, sizeof(double), &area);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &m_qg);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qg, 1, NULL, &gws, NULL, 0, NULL, NULL);
    double *qg_flat = malloc(4 * n * sizeof(double));
    clEnqueueReadBuffer(b->queue, m_qg, CL_TRUE, 0, 4 * n * sizeof(double), qg_flat, 0, NULL, NULL);
    unpack_lmax8_double(qg_flat, qg_idx, sc->count);
    free(qg_flat);

    clReleaseMemObject(m_zb); clReleaseMemObject(m_gammag); clReleaseMemObject(m_kg0); clReleaseMemObject(m_fpg);
    clReleaseMemObject(m_ksg); clReleaseMemObject(m_dif); clReleaseMemObject(m_down); clReleaseMemObject(m_dis);
    clReleaseMemObject(m_len); clReleaseMemObject(m_down1d); clReleaseMemObject(m_dis1d); clReleaseMemObject(m_len1d);
    clReleaseMemObject(m_hg); clReleaseMemObject(m_qg);
    free(down_flat); free(dis_flat); free(len_flat);
}

void rri_cl_infilt(rri_cl_backend *b, const rri_slo_cellset *sc, double dt,
                    double *hs_idx, double *gampt_ff_idx, double *gampt_f_idx)
{
    size_t n = (size_t)sc->count;
    cl_mem m_ksv = buf_ro(b, sc->ksv, n * sizeof(double));
    cl_mem m_faif = buf_ro(b, sc->faif, n * sizeof(double));
    cl_mem m_gammaa = buf_ro(b, sc->gammaa, n * sizeof(double));
    cl_mem m_limit = buf_ro(b, sc->infilt_limit, n * sizeof(double));
    cl_mem m_hs = buf_rw(b, hs_idx, n * sizeof(double));
    cl_mem m_ff = buf_rw(b, gampt_ff_idx, n * sizeof(double));
    cl_mem m_f = buf_out(b, n * sizeof(double));

    int i = 0;
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_ksv);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_faif);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_gammaa);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_limit);
    clSetKernelArg(b->k_infilt, i++, sizeof(double), &dt);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_hs);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_ff);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &m_f);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_infilt, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, m_hs, CL_TRUE, 0, n * sizeof(double), hs_idx, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, m_ff, CL_TRUE, 0, n * sizeof(double), gampt_ff_idx, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, m_f, CL_TRUE, 0, n * sizeof(double), gampt_f_idx, 0, NULL, NULL);

    clReleaseMemObject(m_ksv); clReleaseMemObject(m_faif); clReleaseMemObject(m_gammaa);
    clReleaseMemObject(m_limit); clReleaseMemObject(m_hs); clReleaseMemObject(m_ff); clReleaseMemObject(m_f);
}

/* ---- RK45-stage drivers: identical control flow to rri_funcr/rri_funcs/
 * rri_funcg (src/rri_riv.c, rri_slope.c, rri_gw.c) -- only the discharge
 * kernel call is swapped for the OpenCL dispatch above; the host-side
 * flux-scatter step is copied verbatim (kept as plain C, not itself an
 * OpenCL kernel, per rri.h's file-level comment on why that part of the
 * solver stays host-side regardless of backend). Any change to the CPU
 * versions' scatter logic must be mirrored here. ------------------- */

void rri_cl_funcr(rri_cl_backend *b, const rri_riv_cellset *rc, const double *vr_idx,
                   double ns_river, double area, double *hr_idx, double *fr_idx,
                   double *qr_idx, double *qr_sum_scratch)
{
    for (int k = 0; k < rc->count; k++) hr_idx[k] = rri_vr2hr(vr_idx[k], area, rc->area_ratio[k]);

    rri_cl_qr_calc(b, rc, hr_idx, ns_river, qr_idx);

    for (int k = 0; k < rc->count; k++) qr_sum_scratch[k] = 0.0;
    for (int k = 0; k < rc->count; k++) {
        qr_sum_scratch[k] += qr_idx[k];
        int kk = rc->down[k];
        if (rc->domain[kk] != 0) qr_sum_scratch[kk] -= qr_idx[k];
    }
    for (int k = 0; k < rc->count; k++) fr_idx[k] = -qr_sum_scratch[k];
}

void rri_cl_funcs(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                   const double *qp_t_idx, double area, double *fs_idx, double *qs_idx[RRI_LMAX8])
{
    rri_cl_qs_calc(b, sc, hs_idx, area, qs_idx);

    for (int k = 0; k < sc->count; k++) {
        double outflow = 0.0;
        for (int l = 0; l < RRI_LMAX8; l++) outflow += qs_idx[l][k];
        fs_idx[k] = qp_t_idx[k] - outflow;
    }
    for (int k = 0; k < sc->count; k++) {
        int lmax = sc->lmax;
        int dif_p = sc->dif[k];
        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break;
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            fs_idx[kk] += qs_idx[l][k];
        }
    }
}

void rri_cl_funcg(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                   double area, double *fg_idx, double *qg_idx[RRI_LMAX8])
{
    rri_cl_qg_calc(b, sc, hg_idx, area, qg_idx);

    for (int k = 0; k < sc->count; k++) {
        fg_idx[k] = 0.0;
        if (sc->gammag[k] > 0.0) {
            double s = 0.0;
            for (int l = 0; l < RRI_LMAX8; l++) s += qg_idx[l][k];
            fg_idx[k] = s / sc->gammag[k];
        }
    }
    for (int k = 0; k < sc->count; k++) {
        if (sc->gammag[k] <= 0.0) continue;
        int lmax = sc->lmax;
        int dif_p = sc->dif[k];
        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break;
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            fg_idx[kk] -= qg_idx[l][k] / sc->gammag[k];
        }
    }
}
