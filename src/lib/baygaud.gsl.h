#ifndef __2DBAT_GSL_H__
#define __2DBAT_GSL_H__

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "baygaud.cfitsio.h"
#include "baygaud.multinest.h"
#include "baygaud.trfit.h"
#include "baygaud.sort.h"
#include "baygaud.global_params.h"
#include "baygaud.2dmaps.h"
#include "baygaud.etc.h"
#include "baygaud.gfit.h"
#include "baygaud.memory.h"
#include "baygaud.mpi.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>
#include <float.h>
#include <time.h>

// 2DBAT user defined functions
// GSL related

// GSL histogram
void robust_mean_std(double *input, int n, double *robust_mean, double *robust_std);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// --- End of line

#endif


