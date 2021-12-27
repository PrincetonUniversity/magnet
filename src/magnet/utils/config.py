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
            self.d[k] = self.parse(v)

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

    def parse(self, s):
        s = s.strip()
        if s in ('True', 'False'):
            return eval(s)

        try:
            v = int(s)
        except ValueError:
            try:
                v = float(s)
            except ValueError:
                # We interpret a missing value as None, and a "" as the empty string
                if s.startswith('"') and s.endswith('"'):
                    v = s[1:-1]
                elif s == '':
                    v = None
                elif s.startswith('[') and s.endswith(']'):
                    return [self.parse(t) for t in s[1:-1].split(',')]
                return v
            else:
                return v
        else:
            return v

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
