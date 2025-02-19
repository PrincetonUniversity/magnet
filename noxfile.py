import nox

nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(python=['3.9'])
def tests(session):
    session.install('pytest')
    session.install('-e', '.[dev]')
    session.run('pytest', 'tests')
