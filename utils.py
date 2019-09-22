from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from urllib.error import HTTPError, URLError
import pathlib

import queue
from six.moves.urllib.request import urlretrieve

def make_folder_if_not_exists(target_dir):
    if not os.path.exists(target_dir):
        try:
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
            print("Success.")
        except Exception as e:
            print("Fail. excpetion:{}".format(e))
            raise e
    else:
        print("Already exist.:{}".format(target_dir))

def get_file(fullpath, origin):
    fpath = fullpath
    print("Download path:", fpath)
    fname = os.path.basename(fpath)
    dir_fullpath = os.path.dirname(fullpath)
    if not os.path.isdir(dir_fullpath):
        make_folder_if_not_exists(dir_fullpath)

    if os.path.isfile(fpath):
        return fpath
    print('Downloading data from', origin)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath)
            return fpath
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if os.path.exists(fpath):
            os.remove(fpath)
        raise

from keras.utils import get_file