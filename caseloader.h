#ifndef CASELOADER_H
#define CASELOADER_H

#include <string>
#include <memory>
#include <volumegl.h>

class CaseLoader
{
public:
    CaseLoader(const std::string& filename);
    ~CaseLoader() {}

    void open();
    std::shared_ptr<yy::volren::VolumeGL> scalarField() const { return _scalarField; }
    std::shared_ptr<yy::volren::VolumeGL> velocityField() const { return _velocityField; }

private:
    std::string filename;
    std::shared_ptr<yy::volren::VolumeGL> _scalarField;
    std::shared_ptr<yy::volren::VolumeGL> _velocityField;
};

#endif // CASELOADER_H
