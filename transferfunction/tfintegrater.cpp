#include "tfintegrater.h"
#include "tfinteg1d.h"
#include "tfinteg2d.h"

namespace yy {
namespace volren {

TFIntegrater::TFIntegrater()
 : integ(new TFInteg2D())
 , preinteg(true)
{

}

TFIntegrater::TFIntegrater(bool preinteg)
 : preinteg(preinteg)
{
    if (!preinteg)
        integ.reset(new TFInteg1D());
    else
        integ.reset(new TFInteg2D());
}

void TFIntegrater::convertTo(bool preinteg)
{
    if (this->preinteg == preinteg)
        return;
    this->preinteg = preinteg;
    if (!this->preinteg)
        integ.reset(new TFInteg1D());
    else
        integ.reset(new TFInteg2D());
}

} // namespace volren
} // namespace yy
