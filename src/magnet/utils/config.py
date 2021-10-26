import os
import configparser


class ConfigSection(object):
    """
    A thin wrapper over a ConfigParser's SectionProxy object,
    that tries to infer the types of values, and makes them available as attributes
    Currently int/float/str are supported.
    """
    def __init__(self, config, section_proxy):
        self.config = config
        self.name = section_proxy.name
        self.d = {}  # key value dict where the value is typecast to int/float/str

        for k, v in section_proxy.items():

            if v in ('True', 'False'):
                self.d[k] = eval(v)
                continue

            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    # We interpret a missing value as None, and a "" as the empty string
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    elif v == '':
                        v = None
                    self.d[k] = v
                else:
                    self.d[k] = v
            else:
                self.d[k] = v

    def __setattr__(self, key, value):
        if key in ('config', 'name', 'd'):
            return super(ConfigSection, self).__setattr__(key, value)
        else:
            self.d[key] = value

    def __getattr__(self, item):
        if item not in ('config', 'name', 'd'):
            # If an environment variable exists with name <CONFIG_NAME>_<SECTION>_<ITEM>, use it
            env_varname = '_'.join([str(x).upper() for x in [self.config.name, self.name, item]])
            env_var = os.getenv(env_varname)
            return env_var or self.d[item]

    def items(self):
        return self.d.items()


class Config(object):
    def __init__(self, name, filenames):
        self.name = name
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.init_from_files(filenames)

    def init_from_files(self, filenames):
        self.config.read(filenames)
        self._read_sections()

    def read(self, filename):
        self.config.read(filename)
        self._read_sections()

    def _read_sections(self):
        for section in self.config.sections():
            setattr(self, section, ConfigSection(self, self.config[section]))

    def sections(self):
        return self.config.sections()