#ifndef TFINTEG2D_H
#define TFINTEG2D_H

#define TF_INTEGRATION_USE_GL

#if defined TF_INTEGRATION_USE_CUDA
	#include "tfinteg2dcuda.h"
#elif defined TF_INTEGRATION_USE_GL
	#include "tfinteg2dgl.h"
#endif

namespace yy {
namespace volren {

#if defined TF_INTEGRATION_USE_CUDA
    typedef TFInteg2DCUDA TFInteg2D;
#elif defined TF_INTEGRATION_USE_GL
    typedef TFInteg2DGL TFInteg2D;
#endif

} // namespace volren
} // namespace yy

#endif // TFINTEG2D_H
