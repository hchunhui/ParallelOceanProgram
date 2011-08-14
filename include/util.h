#ifndef _UTIL_H_
#define _UTIL_H_
#include <stdio.h>
static void HandleError( cudaError_t err, const char *file, int line)
{
	if(err != cudaSuccess)
	{
		printf( "%s in %s at line %d\n",
			cudaGetErrorString(err),
			file,
			line);
		exit(2);
	}
}
#define _C(err) (HandleError(err, __FILE__, __LINE__))

#endif
