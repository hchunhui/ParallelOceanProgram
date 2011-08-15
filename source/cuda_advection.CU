#include "util.h"

static int nx_block, ny_block;
static int km;
static int max_blocks_clinic;

typedef double myfloat;
#define v2(name,a,b) \
	name[((b)-1)*nx_block+((a)-1)]
#define v3(name,a,b,c) \
	name[nx_block*( ny_block*((c)-1)+((b)-1) )+((a)-1)]
#define v4(name,a,b,c,d) \
	name[nx_block*( ny_block*(km*((d)-1)+((c)-1))+((b)-1) )+((a)-1)]

myfloat *dDXU;
static int DXUSIZE;
myfloat *dDYU;
static int DYUSIZE;
myfloat *dDZU;
static int DZUSIZE;
myfloat *dWUK;
static int WUKSIZE;
myfloat *dUUU;
static int UUUSIZE;
myfloat *dVVV;
static int VVVSIZE;
myfloat *dUUW;
static int UUWSIZE;
myfloat *dUUE;
static int UUESIZE;
myfloat *dVUS;
static int VUSSIZE;
myfloat *dVUN;
static int VUNSIZE;

//exec dP[..] = val;
__global__ void
kernel_assign(
	myfloat *dP,
	myfloat val)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	dP[i] = val;
}

//partial bottom cells = .true.
__global__ void
kernel1_pbc(
	myfloat *dDXU, myfloat *dDYU, myfloat *dDZU,
	int nx_block, int ny_block, int km, int max_blocks_clinic,
	int ibeg, int jbeg, int k, int bid,
	myfloat *dWUK, myfloat *dUUU, myfloat *dVVV,
	myfloat *dUUW, myfloat *dUUE, myfloat *dVUS, myfloat *dVUN)
{
	int i = threadIdx.x+ibeg-1;
	int j = blockIdx.x+jbeg-1;
	
	v2(dUUW, i, j) =
		0.25 * (
			v3(dUUU, i, j, k) *
			v3(dDYU, i, j, bid) *
			v4(dDZU, i, j, k, bid)
			+
			v3(dUUU, i-1, j, k) *
			v3(dDYU, i-1, j, bid) *
			v4(dDZU, i-1, j, k, bid))
		+
		0.125 * (
			v3(dUUU, i, j-1, k) *
			v3(dDYU, i, j-1, bid) *
			v4(dDZU, i, j-1, k, bid)
			+
			v3(dUUU, i-1, j-1, k) *
			v3(dDYU, i-1, j-1, bid) *
			v4(dDZU, i-1, j-1, k, bid)
			+
			v3(dUUU, i, j+1, k) *
			v3(dDYU, i, j+1, bid) *
			v4(dDZU, i, j+1, k, bid)
			+
			v3(dUUU, i-1, j+1, k) *
			v3(dDYU, i-1, j+1, bid) *
			v4(dDZU, i-1, j+1, k,bid));

	v2(dUUE, i, j) =
		0.25 * (
			v3(dUUU, i+1, j, k) *
			v3(dDYU, i+1, j, bid) *
			v4(dDZU, i+1, j, k, bid)
			+
			v3(dUUU, i, j, k) *
			v3(dDYU, i, j, bid) *
			v4(dDZU, i, j, k, bid))
		+
		0.125 * (
			v3(dUUU, i+1, j-1, k) *
			v3(dDYU, i+1, j-1, bid) *
			v4(dDZU, i+1, j-1, k, bid)
			+
			v3(dUUU, i, j-1, k) *
			v3(dDYU, i, j-1, bid) *
			v4(dDZU, i, j-1, k, bid)
			+
			v3(dUUU, i+1, j+1, k) *
			v3(dDYU, i+1, j+1, bid) *
			v4(dDZU, i+1, j+1, k, bid)
			+
			v3(dUUU, i, j+1, k) *
			v3(dDYU, i, j+1, bid) *
			v4(dDZU, i, j+1, k,bid));

	v2(dVUS, i, j) =
		0.25 * (
			v3(dVVV, i, j, k) *
			v3(dDXU, i, j, bid) *
			v4(dDZU, i, j, k, bid)
			+
			v3(dVVV, i, j-1, k) *
			v3(dDXU, i, j-1, bid) *
			v4(dDZU, i, j-1, k, bid))
		+
		0.125 * (
			v3(dVVV, i-1, j, k) *
			v3(dDXU, i-1, j, bid) *
			v4(dDZU, i-1, j, k, bid)
			+
			v3(dVVV, i-1, j-1, k) *
			v3(dDXU, i-1, j-1, bid) *
			v4(dDZU, i-1, j-1, k, bid)
			+
			v3(dVVV, i+1, j, k) *
			v3(dDXU, i+1, j, bid) *
			v4(dDZU, i+1, j, k, bid)
			+
			v3(dVVV, i+1, j-1, k) *
			v3(dDXU, i+1, j-1, bid) *
			v4(dDZU, i+1, j-1, k,bid));

	v2(dVUN, i, j) =
		0.25 * (
			v3(dVVV, i, j+1, k) *
			v3(dDXU, i, j+1, bid) *
			v4(dDZU, i, j+1, k, bid)
			+
			v3(dVVV, i, j, k) *
			v3(dDXU, i, j, bid) *
			v4(dDZU, i, j, k, bid))
		+
		0.125 * (
			v3(dVVV, i-1, j+1, k) *
			v3(dDXU, i-1, j+1, bid) *
			v4(dDZU, i-1, j+1, k, bid)
			+
			v3(dVVV, i-1, j, k) *
			v3(dDXU, i-1, j, bid) *
			v4(dDZU, i-1, j, k, bid)
			+
			v3(dVVV, i+1, j+1, k) *
			v3(dDXU, i+1, j+1, bid) *
			v4(dDZU, i+1, j+1, k, bid)
			+
			v3(dVVV, i+1, j, k) *
			v3(dDXU, i+1, j, bid) *
			v4(dDZU, i+1, j, k,bid));

}

//partial bottom cells = .false.
__global__ void
kernel1_npbc(
	myfloat *dDXU, myfloat *dDYU, myfloat *dDZU,
	int nx_block, int ny_block, int km, int max_blocks_clinic,
	int ibeg, int jbeg, int k, int bid,
	myfloat *dWUK, myfloat *dUUU, myfloat *dVVV,
	myfloat *dUUW, myfloat *dUUE, myfloat *dVUS, myfloat *dVUN)
{
	int i = threadIdx.x+ibeg-1;
	int j = blockIdx.x+jbeg-1;
	
	v2(dUUW, i, j) =
		0.25 * (
			v3(dUUU, i, j, k) *
			v3(dDYU, i, j, bid)
			+
			v3(dUUU, i-1, j, k) *
			v3(dDYU, i-1, j, bid))
		+
		0.125 * (
			v3(dUUU, i, j-1, k) *
			v3(dDYU, i, j-1, bid)
			+
			v3(dUUU, i-1, j-1, k) *
			v3(dDYU, i-1, j-1, bid)
			+
			v3(dUUU, i, j+1, k) *
			v3(dDYU, i, j+1, bid)
			+
			v3(dUUU, i-1, j+1, k) *
			v3(dDYU, i-1, j+1, bid));

	v2(dUUE, i, j) =
		0.25 * (
			v3(dUUU, i+1, j, k) *
			v3(dDYU, i+1, j, bid)
			+
			v3(dUUU, i, j, k) *
			v3(dDYU, i, j, bid))
		+
		0.125 * (
			v3(dUUU, i+1, j-1, k) *
			v3(dDYU, i+1, j-1, bid)
			+
			v3(dUUU, i, j-1, k) *
			v3(dDYU, i, j-1, bid)
			+
			v3(dUUU, i+1, j+1, k) *
			v3(dDYU, i+1, j+1, bid)
			+
			v3(dUUU, i, j+1, k) *
			v3(dDYU, i, j+1, bid));

	v2(dVUS, i, j) =
		0.25 * (
			v3(dVVV, i, j, k) *
			v3(dDXU, i, j, bid)
			+
			v3(dVVV, i, j-1, k) *
			v3(dDXU, i, j-1, bid))
		+
		0.125 * (
			v3(dVVV, i-1, j, k) *
			v3(dDXU, i-1, j, bid)
			+
			v3(dVVV, i-1, j-1, k) *
			v3(dDXU, i-1, j-1, bid)
			+
			v3(dVVV, i+1, j, k) *
			v3(dDXU, i+1, j, bid)
			+
			v3(dVVV, i+1, j-1, k) *
			v3(dDXU, i+1, j-1, bid));

	v2(dVUN, i, j) =
		0.25 * (
			v3(dVVV, i, j+1, k) *
			v3(dDXU, i, j+1, bid)
			+
			v3(dVVV, i, j, k) *
			v3(dDXU, i, j, bid))
		+
		0.125 * (
			v3(dVVV, i-1, j+1, k) *
			v3(dDXU, i-1, j+1, bid)
			+
			v3(dVVV, i-1, j, k) *
			v3(dDXU, i-1, j, bid)
			+
			v3(dVVV, i+1, j+1, k) *
			v3(dDXU, i+1, j+1, bid)
			+
			v3(dVVV, i+1, j, k) *
			v3(dDXU, i+1, j, bid));

}

extern "C"
void cuda_advection_alloc_
(int *pnx, int *pny, int *pkm, int *pmax_clinic)
{
	int bsize;
	
       	//calculate size
	nx_block = *pnx;
	ny_block = *pny;
	km = *pkm;
	max_blocks_clinic = *pmax_clinic;
	bsize = sizeof(myfloat)*nx_block*ny_block;
	DXUSIZE = km * bsize;
	DYUSIZE = km * bsize;
	DZUSIZE = km * max_blocks_clinic * bsize;
	WUKSIZE = bsize;
	UUUSIZE = km * bsize;
	VVVSIZE = km * bsize;
	UUWSIZE = bsize;
	UUESIZE = bsize;
	VUSSIZE = bsize;
	VUNSIZE = bsize;

	//allocate device memory
	clock_t c1, c2, c15;
	float t1, t15;

	c1 = clock();
	//const parameters
	_C(cudaMalloc(&dDXU, DXUSIZE));
	_C(cudaMalloc(&dDYU, DYUSIZE));
	c15 = clock();
	_C(cudaMalloc(&dDZU, DZUSIZE));

	//input parameters
	_C(cudaMalloc(&dWUK, WUKSIZE));
	_C(cudaMalloc(&dUUU, UUUSIZE));
	_C(cudaMalloc(&dVVV, VVVSIZE));

	//output parameters
	_C(cudaMalloc(&dUUW, UUWSIZE));
	_C(cudaMalloc(&dUUE, UUESIZE));
	
	_C(cudaMalloc(&dVUS, VUSSIZE));
	_C(cudaMalloc(&dVUN, VUNSIZE));
	c2 = clock();
	t1 = (c2 - c1) * 1.0 / CLOCKS_PER_SEC;
	t15 = (c15 - c1) * 1.0 / CLOCKS_PER_SEC;

	printf("cudaMalloc: %f s\n", t1);
	printf("start time: %f s\n", t15);
	printf("nx_block: %d ny_block: %d km: %d\n",
	       nx_block, ny_block, km);
}

extern "C"
void cuda_advection_copyconst_
(myfloat *DXU, myfloat *DYU, myfloat *DZU)
{
	cudaMemcpy(dDXU, DXU, DXUSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dDYU, DYU, DYUSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dDZU, DZU, DZUSIZE, cudaMemcpyHostToDevice);
}

extern "C"
void cuda_advection_copyin_
(myfloat *WUK, myfloat *UUU, myfloat *VVV)
{
	cudaMemcpy(dWUK, WUK, WUKSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dUUU, UUU, UUUSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dVVV, VVV, VVVSIZE, cudaMemcpyHostToDevice);
}

extern "C"
void cuda_advection_copyout_
(myfloat *UUW, myfloat *UUE, myfloat *VUS, myfloat *VUN)
{
	cudaMemcpy(UUW, dUUW, UUWSIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(UUE, dUUE, UUESIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(VUS, dVUS, VUSSIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(VUN, dVUN, VUNSIZE, cudaMemcpyDeviceToHost);
}

extern "C"
void cuda_advection_free_()
{
	//const parameters
	_C(cudaFree(dDXU));
	_C(cudaFree(dDYU));
	_C(cudaFree(dDZU));

	//input parameters
	_C(cudaFree(dWUK));
	_C(cudaFree(dUUU));
	_C(cudaFree(dVVV));

	//output parameters
	_C(cudaFree(dUUW));
	_C(cudaFree(dUUE));
	
	_C(cudaFree(dVUS));
	_C(cudaFree(dVUN));
}

extern "C"
void cuda_advection_run_pbc_
(int *ibeg, int *iend, int *jbeg, int *jend, int *k,int *bid)
{
     	dim3 dimGrid(*jend - *jbeg + 3);
	dim3 dimBlock(*iend - *ibeg + 3);

	kernel1_npbc<<<dimGrid, dimBlock>>>
		(dDXU, dDYU, dDZU,
		 nx_block, ny_block, km, max_blocks_clinic,
		 *ibeg, *jbeg, *k, *bid,
		 dWUK, dUUU, dVVV,
		 dUUW, dUUE, dVUS, dVUN);
}

extern "C"
void cuda_advection_run_npbc_
(int *ibeg, int *iend, int *jbeg, int *jend, int *k, int *bid)
{
     	dim3 dimGrid(*jend - *jbeg + 3);
	dim3 dimBlock(*iend - *ibeg + 3);

	kernel1_npbc<<<dimGrid, dimBlock>>>
		(dDXU, dDYU, dDZU,
		 nx_block, ny_block, km, max_blocks_clinic,
		 *ibeg, *jbeg, *k, *bid,
		 dWUK, dUUU, dVVV,
		 dUUW, dUUE, dVUS, dVUN);
}

extern "C"
void cuda_advection_run_clearoutput_
()
{
     	dim3 dimGrid(ny_block);
	dim3 dimBlock(nx_block);

	kernel_assign<<<dimGrid, dimBlock>>>(dUUW, 0.0);
	kernel_assign<<<dimGrid, dimBlock>>>(dUUE, 0.0);
	kernel_assign<<<dimGrid, dimBlock>>>(dVUN, 0.0);
	kernel_assign<<<dimGrid, dimBlock>>>(dVUS, 0.0);
}