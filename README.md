# Disco-DJ Tesseract Examples

This repository contains two Disco-DJ + Tesseract examples.

- [Universe in a container](discodj_tesseract_example/embedded_tesseract/README.md): a field-level inverse problem that reconstructs Fourier-space white-noise modes for a late-time density target using L-BFGS.
![Optimized forward evolution](discodj_tesseract_example/embedded_tesseract/outputs/optimized_forward_evolution_3d.gif)

- [Lyman-alpha Enzyme demo](discodj_tesseract_example/enzyme_fgpa_loga/README.md): a Fortran + Enzyme Lyman-alpha skewer example involving two Tesseracts that jointly samples an initial condition field and a global parameter with Hamiltonian Monte Carlo.
![Posterior samples](discodj_tesseract_example/enzyme_fgpa_loga/posterior_example.gif)

Original repos:
* Tesseract: https://github.com/pasteurlabs/tesseract-core
* Disco-DJ: https://github.com/cosmo-sims/DISCO-DJ

Parts of this project are derived from https://github.com/pasteurlabs/tesseract-core by PasteurLabs, which is licensed under the Apache License 2.0.
Disco-DJ is licensed under the GPLv3. This repository is distributed under GPLv3; Apache-2.0 code can be combined into GPLv3-covered work. See https://www.apache.org/licenses/GPL-compatibility.
