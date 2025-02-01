import urllib.request

class ParticleDataDownloader:
    def __init__(self):
        self.photon_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/AtBT8y4MiQYFcgc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
        self.electron_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/FbXw3V4XNyYB3oA/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'
        self.download_file(self.photon_url, "./data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5")
        self.download_file(self.electron_url, "./data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5")

    @classmethod
    def download_file(self, url, filename):
        urllib.request.urlretrieve(url, filename)
