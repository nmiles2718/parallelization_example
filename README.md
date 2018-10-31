# parallelization_example
This repo contains two examples on how to parallelize your workflow using `dask`. 

[dask_vs_multiprocessing.py](https://github.com/nmiles2718/parallelization_example/blob/master/dask_vs_multiprocessing.py):
 - A trivialized example of how to parallelize a function of more than argument using `dask`. It also includes a solution using the `multiprocessing` module, however this solution only works in python 3.3+. 
 - The main point of the example is to demonstrate the ability to pass the arguments directly to the function when using `dask`. This is in contrast with the approach in `multiprocessing` which requires the use of the `starmap` method on the [`Pool`](https://docs.python.org/3/library/multiprocessing.html?highlight=pool#module-multiprocessing.pool) class, which will unpack a tuple of arguments for you. 

[find_sources.py]():
- A slightly more complicated example of how to parallelize the process of finding sources in a list of astronomical images. In order to keep the example self-contained, it leverages one of the example datasets installed with the [`photutils`](https://photutils.readthedocs.io/en/stable/) package to generate a fake dataset.
- It can be run from the command line using the command line arguments to set the number of images in the fake dataset, to set the number of worker processes to use, to specify which parallelization framework to use (`dask` or `multiprocessing`),to plot all the sources found for one of the fake datasets (after processing has completed), or to just run things serially so you can establish a baseline for comparison. 

```console 
(astroconda3) [nmiles@:nathan parallelization_example]$ python find_sources.py -h
usage: find_sources.py [-h] [-dask] [-n N] [-nworkers NWORKERS] [-s] [-plot]

optional arguments:
  -h, --help          show this help message and exit
  -dask               Use dask instead of multiprocessing (True)
  -n N                Number of datasets to process (20)
  -nworkers NWORKERS  set the number of workers (os.cpu_count())
  -s                  Run the analysis serially (False)
  -plot               Show an example of the sources found (False)
  ```
