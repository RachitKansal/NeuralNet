#!/usr/bin/env python

from setuptools import setup

ENTRY_POINTS = {
	# Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
	'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        #'exampletutorials = orangecontrib.example.tutorials',
    ),

	'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'NeuralNet = orangecontrib.neuralnet.widgets',
    ),

	# 'orange.widgets': (
 #        'NeuralNet = orangecontrib.neuralnet.widgets',
 #    ),
}

KEYWORDS = (
    'orange3 add-on',
)

if __name__ == '__main__':
	setup(
		name="Orange3 NeuralNet Add-on",
		packages=['orangecontrib',
                  'orangecontrib.neuralnet',
                  'orangecontrib.neuralnet.widgets'],
        package_data={
        	#'orangecontrib.neuralnet': ['tutorials/*.ows'],
            'orangecontrib.neuralnet.widgets': ['icons/*'],
        },
        install_requires=[
            'Orange',
            'numpy',
            'scipy',
            'theano',
            'cython',
            'h5py',
            'keras',
        ],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
	)