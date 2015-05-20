#include "tfintegrater.h"
#include "tfinteg1d.h"
#include "tfinteg2d.h"

//
//
//
//
// Factory create method
//
//
//
//

std::unique_ptr<TFIntegrater> TFIntegrater::create(bool preintegrate)
{
    if (preintegrate)
        return std::unique_ptr<TFIntegrater>(new TFInteg2D());
    return std::unique_ptr<TFIntegrater>(new TFInteg1D());
}

TFIntegrater::TFIntegrater()
{

}

TFIntegrater::~TFIntegrater()
{

}
