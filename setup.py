import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Draw-and-Understand",
    version="0.0.1",
    author="Afeng-x",
    description="Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AFeng-x/Draw-and-Understand",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
