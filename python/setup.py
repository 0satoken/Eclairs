from setuptools import setup

setup(
    name="pyecl",
    version="1.0",
    description="Eclairs: Efficient Codes for the LArge scales of the unIveRSe",
    url="https://github.com/0satoken/Eclairs",
    author="Ken Osato",
    author_email="ken.osato@chiba-u.jp",
    zip_safe=False,
    packages=['pyecl'],
    package_dir={'pyecl': 'pyecl'},
    package_data={'pyecl': ['pyeclairs.so']},
)

