# Eclairs

Author: Ken Osato  
Contributors: Takahiro Nishimichi, Francis Bernardeau, and Atsushi Taruya  

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

## Prerequisite

To compile `eclairs`, the required libraries are

* C++ compiler (`g++`, `icpc`, or `clang++`)  
* [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/)  

To compile the python wrapper ``, additional requirements are  

* Python3
* pip
* [Boost](https://www.boost.org/)

Specify the paths for these libraries in `Makefile`.
Then, an executable `eclairs` is created.
The basic usage is

```C++
> ./eclairs [initial parameter file]
```

You can find example parameter files at [inifiles](inifiles).
If you pass nothing, the code runs with default parameters.

## Jupyter notebooks for `pyeclairs`

We provide tutorials to run `pyeclairs` as Jupyter notebooks at [notebooks](notebooks).
Currently, only basic tutorial notebook is available. We will add more notebooks later.

## Notes on the fast mode with the response function approach

In [Osato et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021PhRvD.104j3501O),
the response function approach is implemented in `eclairs`.
This mode requires the precomputed kernels to compute the response function.
Indeed, the module to generate the precomputed tables is found in this repository.
Since the calculations take long (a few days with MPI parallelizations),
we provide the precomputed tables with 10 cosmological parameter sets assuming $\Lambda$CDM cosmology.
We will put the download link later on this repository.

## License

This code can be distributed under MIT License.
For details, please see the LICENSE file.  
If you use this code in your work, please cite the following papers.

* [Osato, Nishimichi, Bernardeau, and Taruya (2019)](https://ui.adsabs.harvard.edu/abs/2019PhRvD..99f3530O)
* [Osato, Nishimichi, Taruya, and Bernardeau (2021)](https://ui.adsabs.harvard.edu/abs/2021PhRvD.104j3501O)
* [Osato, Nishimichi, Taruya, and Bernardeau (2023)](https://ui.adsabs.harvard.edu/abs/2023PhRvD.108l3541O/abstract)
