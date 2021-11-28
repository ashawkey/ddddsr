from setuptools import setup

if __name__ == '__main__':
    setup(
        name="ddddsr",
        version='0.1.4',
        description="end-to-end super resolution toolkit",
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ashawkey/ddddsr',
        author='kiui',
        author_email='ashawkey1999@gmail.com',
        packages=['ddddsr'],
        include_package_data=True,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3 ',
        ],
        keywords='deep learning, super resolution',
        install_requires=[
            'numpy',
            'onnxruntime',
            'opencv-python',
            'shapely',
            'pyclipper',
            'pillow',
        ],
    )