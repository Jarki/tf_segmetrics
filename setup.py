import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tf_segmetrics',
    version='0.0.1',
    author='Jarki',
    description='Metrics that can be used for semantic segmentation implemented using tensorflow.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=['metrics'],
    install_requires=['tensorflow', 'numpy'],
)