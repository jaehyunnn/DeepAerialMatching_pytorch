r"""Functions to download semantic correspondence datasets"""

from os.path import exists, join, basename
from os import makedirs, remove
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_train(datapath):
    if not exists(datapath):
        makedirs(datapath)

    dest_current = datapath+'/current'
    dest_past = datapath+'/past'

    if not exists(dest_current):
        makedirs(dest_current)
        gdd.download_file_from_google_drive(file_id='1h6opbXMpdxL5KhsLMO3flKYtLOoSs0oI',
                                            dest_path=dest_current+'/current.zip',
                                            unzip=True)
    if not exists(dest_past):
        gdd.download_file_from_google_drive(file_id='1aJi58H3EscOYqF6C_KLsKMDrb1XSgor2',
                                            dest_path=dest_past+'/past.zip',
                                            unzip=True)

def download_eval(datapath):
    if not exists(datapath):
        makedirs(datapath)

    dest = datapath + '/GoogleEarth_pck'

    if not exists(dest):
        makedirs(dest)
        gdd.download_file_from_google_drive(file_id='1DCYxMKPVrpx13vErOoeApVExPdZnMNLp',
                                            dest_path=dest + '/GoogleEarth_pck.zip',
                                            unzip=True)