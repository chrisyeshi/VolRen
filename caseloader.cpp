#include "caseloader.h"
#include <volumegl.h>
#include <volloadraw.h>
#include <fstream>
#include <json/json.h>

CaseLoader::CaseLoader(const std::string &filename)
  : filename(filename)
{

}

void CaseLoader::open()
{
    const static int nVelComp = 3;
    std::ifstream fin(filename.c_str(), std::ifstream::binary);
    Json::Value json;
    Json::Reader reader;
    reader.parse(fin, json);
    // json reading
    std::string sFile = json["scalar"].asString();
    int w = json["dimension"][0].asInt();
    int h = json["dimension"][1].asInt();
    int d = json["dimension"][2].asInt();
    static std::map<std::string, yy::IVolume::ScalarType> str2type
            = { { "unsigned char", yy::IVolume::ST_Unsigned_Char },
                { "char", yy::IVolume::ST_Char },
                { "float", yy::IVolume::ST_Float },
                { "double", yy::IVolume::ST_Double } };
    yy::IVolume::ScalarType type = str2type[json["scalartype"].asString()];
    // load scalar volume
    _scalarField.reset(new yy::volren::VolumeGL(yy::VolLoadRAW(sFile).open(type, w, h, d)));
    // load velocity volumes
    if (json["velocity"].isArray())
    {
        std::vector<std::string> vFiles(nVelComp);
        for (int iVelComp = 0; iVelComp < nVelComp; ++iVelComp)
            vFiles[iVelComp] = json["velocity"][iVelComp].asString();
        std::unique_ptr<unsigned char[]> vData(new unsigned char [_scalarField->nBytes() * nVelComp]);
        for (int iVelComp = 0; iVelComp < nVelComp; ++iVelComp)
        {
            auto vVol = yy::VolLoadRAW(vFiles[iVelComp]).open(type, w, h, d);
            const auto& data = vVol->getData();
            int nbps = vVol->nBytesPerScalar();
            assert(1 == vVol->nScalarsPerVoxel());
            assert(vVol->nBytesPerScalar() == vVol->nBytesPerVoxel());
            for (int iScalar = 0; iScalar < vVol->nScalars(); ++iScalar)
                memcpy(&vData[nbps * (3 * iScalar + iVelComp)], &data[nbps * iScalar], nbps);
        }
        std::shared_ptr<yy::Volume> vVolume = std::make_shared<yy::Volume>(vData, type, nVelComp, w, h, d);
        _velocityField.reset(new yy::volren::VolumeGL(vVolume));

    } else
    {
        std::string vFile = json["velocity"].asString();
        std::unique_ptr<unsigned char[]> vData(new unsigned char [_scalarField->nBytes() * nVelComp]);
        size_t nBytes = w * h * d * sizeof(float) * nVelComp;
        std::ifstream fin(vFile.c_str(), std::ifstream::binary);
        fin.read(reinterpret_cast<char*>(vData.get()), nBytes);
        std::shared_ptr<yy::Volume> vVolume = std::make_shared<yy::Volume>(vData, type, nVelComp, w, h, d);
        _velocityField.reset(new yy::volren::VolumeGL(vVolume));
    }
}

