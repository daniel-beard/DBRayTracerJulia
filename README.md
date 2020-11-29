# DBRayTracerJulia

[![Build Status](https://dev.azure.com/danielbeard0/danielbeard0/_apis/build/status/daniel-beard.DBRayTracerJulia)](https://dev.azure.com/danielbeard0/danielbeard0/_build/latest?definitionId=2) ![julia](https://img.shields.io/badge/julia-1.5-brightgreen.svg)

![Image](output.png)

- This is a Julia port of the Swift raytracer here: [DBRayTracer](https://github.com/daniel-beard/DBRaytracer)

## Running

You can run this either single threaded by not passing a `--threads` argument, or with multiple threads.

```bash
julia --threads 12 raytracer.jl
```

