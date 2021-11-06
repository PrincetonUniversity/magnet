[![Build](https://github.com/PrincetonUniversity/magnet/actions/workflows/main.yml/badge.svg)](https://github.com/PrincetonUniversity/magnet/actions/workflows/main.yml)
![MagNet Logo](app/img/magnetlogo.jpg)

Princeton MagNet is a large-scale dataset designed to enable researchers modeling magnetic core loss using machine learning to accelerate the design process of power electronics. The dataset contains a large amount of voltage and current data of different magnetic components with different shapes of waveforms and different properties measured in the real world. Researchers may use these data as pairs of excitations and responses to build up analytical magnetic models or calculate the core loss to derive static models.

## Website

Princeton MagNet is currently deployed at https://mag-net.princeton.edu/

## Documentation

The web application for Princeton MagNet uses the `magnet` package, a python package under development where most of
the functionality is exposed. Before `magnet` is released on PyPI, it can be installed using
`pip install git+https://github.com/PrincetonUniversity/magnet`.

API Documentation for `magnet` can be viewed online at [https://princetonuniversity.github.io/magnet/](https://princetonuniversity.github.io/magnet/)

## How to Cite

If you used MagNet, please cite us with the following.


H. Li, S. Lee, M. Luo, C. R. Sullivan, Y. Chen and M. Chen, "MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling,” IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Aalborg, Denmark, 2020.

E. Dogariu, H. Li, D. Serrano López, S. Wang, M. Luo and M. Chen, "Transfer Learning Methods for Magnetic Core Loss Modeling,” IEEE Workshop on Control and Modeling of Power Electronics (COMPEL), Cartagena de Indias, Colombia, 2021.

H. Li, T. Guillod, E. Dogariu, A. Nadler, S. Wang, M. Luo, Y. Chen, C. R. Sullivan, and M. Chen, "MagNet: an Open-Source Database for Data-Driven Magnetic Core Loss Modeling," IEEE Applied Power Electronics Conference (APEC), Houston, 2022.


## Team Members

Princeton MagNet is currently maintained by the Power Electronics Research Lab as Princeton University. We also collaborate with Dartmouth College, and Plexim.

![MagNet Team](app/img/magnetteam.jpg)

## Sponsors

This work is sponsored by the ARPA-E DIFFERENTIATE Program, Princeton CSML DataX program, Princeton Andlinger Center for Energy and the Environment, and National Science Foundation under the NSF CAREER Award. 

![MagNet Sponsor](app/img/sponsor.jpg)
