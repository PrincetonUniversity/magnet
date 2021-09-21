import nox


@nox.session(python=['3.7', '3.8', '3.9'])
def tests(session):
    session.install('pytest')
    session.install('-e', '.[dev]')
    session.run('pytest', 'tests')
