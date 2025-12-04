import setuptools

if __name__ == "__main__":
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        author_email="clement.grisi@radboudumc.nl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/clemsgrs/hs2p",
        project_urls={"Bug Tracker": "https://github.com/clemsgrs/hs2p/issues"},
        packages=setuptools.find_packages(exclude=["tests"]),
        exclude_package_data={"": ["tests"]},
    )
