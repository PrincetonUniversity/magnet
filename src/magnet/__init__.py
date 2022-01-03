from ._version import version
import os.path
from importlib.resources import path
import magnet
from magnet.utils.config import Config


__version__ = version


with path(magnet, 'magnet.ini') as ini_file:
    filenames = [ini_file]
    # If a magnet.ini file exists where we were imported from, use it after the pre-packaged .ini file
    # This allows overriding of values without having to alter the pre-packaged .ini file, which may
    # be buried deep inside the site-packages folder.
    if os.path.exists('magnet.ini'):
        filenames.append('magnet.ini')
    config = Config('magnet', filenames)
