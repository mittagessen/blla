baseline neural layout analysis
===============================

This is a fully neural layout analysis tool capable of extracting arbitrarily
shaped lines from documents. It operates by first labelling all
baselines in the document using a FCN-style deep network and then extracting a
fixed environment around these baselines.

Everything is highly experimental, subject to changes without notice, and will
break frequently.

Installation
------------

Run:

::
        $ pip3 install .

to install the dependencies and the command line tool. For development purposes
use:

::
        $ pip3 install --editable .

Training
--------

Training requires a directory with triplets of input images
$prefix.{plain,seeds}.png. `plain` are RGB inputs, `seeds` are 8bpp baselines
annotations with each non-zero value representing a single baseline.

::

   $ blla train --validation val train
