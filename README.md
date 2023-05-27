# multilayer-brains

A pipeline for the construction and multilayer motif analysis of temporal fMRI brain networks. Consists of multiple modular parts, each of which can be applied separately - no need to restrict yourself to brain data only.

### Network construction

Tools for creating multilayer networks from four-dimensional data arrays. Ready-made implementation for reading data arrays from fMRI files in the NIfTI data format, but naturally any data arrays can be used.

### Motif analysis

Tools for identifying distributions of isomorphic subgraphs and motifs from any kind of multilayer network. Uses the [pymnet](http://www.mkivela.com/pymnet/) library as the multilayer network back-end.
