# tfShearlab
Tensorflow implementation of Shearlab, including a python wrapper of the [Julia Shearlab APi](https://github.com/arsenal9971/Shearlab.jl)

## Installation and dependencies 

**tfShearlab** can be easily installed by running

```bash
$ pip install https://github.com/arsenal9971/tfshearlab/archive/master.zip
```

This package has the next dependencies

- **Julia language**: One can either [precompiled packages](https://julialang.org/downloads/) or build from [source](https://github.com/JuliaLang/julia). This package requires the Julia version 0.6 or higher.

- **Shearlab.jl**: To install the library in Julia 0.6.x one needs to run the command `julia -e 'Pkg.add("Shearlab")`.

- **Pyjulia**: One can install the Python API of Julia with the command `pip install julia`, for more details on installation check the [documentation](https://odlgroup.github.io/odl/getting_started/installing.html).
   - One also needs to make the Julia and Python enviroment to coincide running the command `julia -e 'ENV["PYTHON"]="<your-python-executable>"; Pkg.add("PyCall"); Pkg.build("PyCall")'`. One can find its python executable path by running on the terminal `$(which python)`.

- **SSL certificates**: Sometimes you need to give (and add to bashrc.) the SSL certificates path using `export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem`.

## Description 

[Shearlab.jl](https://github.com/arsenal9971/Shearlab.jl) is a Julia Library with toolbox for two- and threedimensional data processing using the Shearlet system as basis functions which generates an sparse representation of cartoon-like functions.  

**tfShearlab** is a tensorflow implementation of the Shearlet transform using the Julia API as backend. The reason for this implementation lies in mainly in the GPU-functionalities of **tensorflow** that accelerates the fft-based convolutions; in comparison with the version without tensorflow, the Shearlet decomposition and recosntruction are about `30x` faster in a GTX 1080 graphic card. 

This package also contains a python wrapper of the Julia API, so one can perform the Shearlet transform without tensorflow.

For the 2D version one has three important functions:

- Generate the Shearlet System.
```python
getshearletsystem2D(rows,cols,nScales,shearLevels,full,directionalFilter,scalingFilter);
```

- Decoding of a signal X.
```python
tfsheardec2D(Xtf, tfshearlets)  
```

- Reconstruction of a signal Xtf.
```python
tfshearrec2D(coeffstf,tfshearlets,tfdualFrameWeights )
```

For more detailed usage functionalities check the original [Shearlab manual](http://shearlab.org/files/documents/ShearLab3Dv10_Manual.pdf), or [examples](https://github.com/arsenal9971/tfshearlab/tree/master/examples) for scientific reference one can also read ["ShearLab 3D: Faithful Digital Shearlet Transforms Based on Compactly Supported Shearlets"](http://www.math.tu-berlin.de/fileadmin/i26_fg-kutyniok/Kutyniok/Papers/ShearLab3D.pdf).
